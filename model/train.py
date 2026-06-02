"""
HydroPINN training loop.

Two-phase optimisation:
  Phase 1 — Adam  (lr=1e-3, 5 000 epochs)  : broad exploration
  Phase 2 — L-BFGS (1 000 steps)           : sharp convergence

Total loss:
  L = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc
  λ_data = 1.0,  λ_pde = 0.1,  λ_bc = 0.5

Collocation points : 10 000 random (x,y,t) samples — physics loss evaluated here
Observation points : 200 sparse sensor readings — data loss evaluated here
Boundary points    : grid boundary cells — h forced to 0
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup so we can import from project root
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.pinn import HydroPINN, make_input_tensor
from model.physics_loss import physics_loss, physics_loss_scaled
from model.data_loss import generate_observation_points, data_loss

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_COLLOC     = 10_000   # collocation points for physics loss
N_OBS        = 200      # sparse observation points
N_BC         = 500      # boundary condition points
ADAM_LR      = 1e-3
ADAM_EPOCHS  = 5_000
LBFGS_STEPS  = 1_000
LAMBDA_DATA  = 1.0
LAMBDA_PDE   = 0.1
LAMBDA_BC    = 0.5
CHECKPOINT   = os.path.join(_ROOT, "outputs", "pinn_checkpoint.pt")
SEED         = 42


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collocation_points(n: int, device: torch.device, seed: int = 0):
    """Sample random (x,y,t,z,R) collocation points. x,y,t have requires_grad."""
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(0, 1, n).astype(np.float32)
    y_np = rng.uniform(0, 1, n).astype(np.float32)
    t_np = rng.uniform(0, 1, n).astype(np.float32)

    from data.rainfall.generate_rainfall import rainfall_rate
    R_np = rainfall_rate(x_np, y_np, t_np)

    # We'll fill z from DEM after loading
    return x_np, y_np, t_np, R_np


def _bc_points(n: int, device: torch.device, seed: int = 1):
    """Sample points on the grid boundary where h should be 0."""
    rng = np.random.default_rng(seed)
    t_np = rng.uniform(0, 1, n).astype(np.float32)
    edge = rng.integers(0, 4, n)   # 0=left,1=right,2=bottom,3=top
    x_np = np.where(edge == 0, 0.0,
           np.where(edge == 1, 1.0, rng.uniform(0, 1, n))).astype(np.float32)
    y_np = np.where(edge == 2, 0.0,
           np.where(edge == 3, 1.0, rng.uniform(0, 1, n))).astype(np.float32)
    return x_np, y_np, t_np


def _to_tensor(arr: np.ndarray, device, grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    if grad:
        t = t.requires_grad_(True)
    return t


def _bc_loss(model, x_bc, y_bc, t_bc, z_bc, R_bc, device) -> torch.Tensor:
    """h must be 0 at domain boundaries (no in-flow from outside)."""
    inp = torch.stack([x_bc, y_bc, t_bc, z_bc, R_bc], dim=1)
    h, _, _ = model(inp)
    return h.pow(2).mean()


def load_dem_grid():
    """Load DEM, generating it if not yet on disk."""
    npy_path = os.path.join(_ROOT, "data", "terrain", "dem.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32)
    try:
        from data.terrain.fetch_real_dem import fetch_srtm_dem
        return fetch_srtm_dem().astype(np.float32)
    except Exception:
        from data.terrain.generate_dem import generate_synthetic_dem
        return generate_synthetic_dem().astype(np.float32)


def sample_z_from_dem(dem: np.ndarray,
                      x_np: np.ndarray,
                      y_np: np.ndarray) -> np.ndarray:
    """Sample terrain elevation at arbitrary (x,y) in [0,1]^2."""
    H, W = dem.shape
    xi = (x_np * (W - 1)).clip(0, W - 1).astype(int)
    yi = (y_np * (H - 1)).clip(0, H - 1).astype(int)
    return dem[yi, xi]


def compute_dem_gradients(dem: np.ndarray):
    """
    Pre-compute terrain slopes dz/dx and dz/dy via finite differences.
    Returns two arrays same shape as dem, normalised to [0,1]^2 coordinates.
    """
    H, W = dem.shape
    # Central differences, forward/backward at boundaries
    dz_dy_grid = np.gradient(dem, axis=0) * H   # chain rule: d/d(y_norm) = d/d(row) * H
    dz_dx_grid = np.gradient(dem, axis=1) * W   # chain rule: d/d(x_norm) = d/d(col) * W
    return dz_dx_grid.astype(np.float32), dz_dy_grid.astype(np.float32)


def sample_dem_grads(dz_dx_grid: np.ndarray,
                     dz_dy_grid: np.ndarray,
                     x_np: np.ndarray,
                     y_np: np.ndarray):
    """Sample pre-computed terrain gradient grids at collocation points."""
    H, W = dz_dx_grid.shape
    xi = (x_np * (W - 1)).clip(0, W - 1).astype(int)
    yi = (y_np * (H - 1)).clip(0, H - 1).astype(int)
    return dz_dx_grid[yi, xi], dz_dy_grid[yi, xi]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    adam_epochs: int = ADAM_EPOCHS,
    lbfgs_steps: int = LBFGS_STEPS,
    lambda_data: float = LAMBDA_DATA,
    lambda_pde:  float = LAMBDA_PDE,
    lambda_bc:   float = LAMBDA_BC,
    verbose: bool = True,
) -> HydroPINN:

    device = _device()
    print(f"[train] Device: {device}")

    os.makedirs(os.path.join(_ROOT, "outputs"), exist_ok=True)

    # Load terrain
    dem = load_dem_grid()
    print(f"[train] DEM loaded: shape={dem.shape}  "
          f"min={dem.min():.1f}m  max={dem.max():.1f}m")

    # Build model
    model = HydroPINN().to(device)
    print(f"[train] Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ---- Observation points (fixed throughout training) ----
    obs = generate_observation_points(n_obs=N_OBS, dem=dem)

    # Pre-compute terrain gradients (finite differences on DEM grid)
    dz_dx_grid, dz_dy_grid = compute_dem_gradients(dem)

    # ---- Collocation points ----
    x_np, y_np, t_np, R_np = _collocation_points(N_COLLOC, device)
    z_np = sample_z_from_dem(dem, x_np, y_np)
    dzdx_np, dzdy_np = sample_dem_grads(dz_dx_grid, dz_dy_grid, x_np, y_np)

    # ---- Boundary condition points ----
    xb_np, yb_np, tb_np = _bc_points(N_BC, device)
    zb_np = sample_z_from_dem(dem, xb_np, yb_np)
    from data.rainfall.generate_rainfall import rainfall_rate
    Rb_np = rainfall_rate(xb_np, yb_np, tb_np)

    # Convert BC points (no grad needed)
    def _t(arr): return torch.tensor(arr, dtype=torch.float32, device=device)
    x_bc, y_bc, t_bc = _t(xb_np), _t(yb_np), _t(tb_np)
    z_bc, R_bc       = _t(zb_np), _t(Rb_np)

    # ================================================================
    # Phase 1a — Long data-only warmup (50% of Adam budget)
    # Run at high lr (3e-3) with NO physics — very fast (no autograd
    # through network for derivatives).  Gets data_loss < 0.005 before
    # physics is ever introduced, so the combined phase has a strong
    # starting point and physics gradients cannot dominate.
    # ================================================================
    warmup_epochs = adam_epochs // 2
    print(f"\n[train] Phase 1a — Data-only warmup ({warmup_epochs} epochs, lr=3e-3) …")
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)

    for epoch in tqdm(range(1, warmup_epochs + 1), disable=not verbose, ncols=80):
        optimiser.zero_grad()
        L_data = data_loss(model, obs, device)
        L_data.backward()
        optimiser.step()

        if verbose and epoch % max(1, warmup_epochs // 5) == 0:
            tqdm.write(f"  warmup {epoch:5d}  data={L_data.item():.4e}")

    # ================================================================
    # Phase 1b — Combined Adam with SCALED continuity physics loss
    # Uses physics_loss_scaled() which produces O(1) residuals instead
    # of O(10^3) — prevents physics gradients from overwhelming data.
    # lambda_pde=0.05 is now safely balanced against data loss.
    # ================================================================
    combined_epochs = adam_epochs - warmup_epochs
    print(f"\n[train] Phase 1b — Combined Adam ({combined_epochs} epochs, scaled PDE) …")
    optimiser = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

    for epoch in tqdm(range(1, combined_epochs + 1), disable=not verbose, ncols=80):
        optimiser.zero_grad()

        # Only x, y, t, R need to be tracked; z is fixed terrain
        x = _to_tensor(x_np, device, grad=True)
        y = _to_tensor(y_np, device, grad=True)
        t = _to_tensor(t_np, device, grad=True)
        z = _to_tensor(z_np, device, grad=False)
        R = _to_tensor(R_np, device, grad=False)

        # Scaled continuity-only PDE loss (O(1) residuals — safe to use)
        L_pde  = physics_loss_scaled(model, x, y, t, z, R)
        L_data = data_loss(model, obs, device)
        L_bc   = _bc_loss(model, x_bc, y_bc, t_bc, z_bc, R_bc, device)

        loss = lambda_data * L_data + lambda_pde * L_pde + lambda_bc * L_bc
        loss.backward()
        optimiser.step()

        if verbose and epoch % 500 == 0:
            tqdm.write(
                f"  epoch {epoch:5d}  "
                f"loss={loss.item():.4e}  "
                f"data={L_data.item():.4e}  "
                f"pde_scaled={L_pde.item():.4e}  "
                f"bc={L_bc.item():.4e}"
            )

    # ================================================================
    # Phase 2 — L-BFGS data-only fine-tuning
    # CRITICAL: physics is excluded here. L-BFGS with unnormalised SWE
    # residuals previously destroyed the good Adam data fit (MSE went
    # from 0.019 → 0.86). Data-only L-BFGS locks in the Adam solution.
    # ================================================================
    print(f"\n[train] Phase 2 — L-BFGS data-only  ({lbfgs_steps} steps) …")
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    step_count = [0]

    def closure():
        if step_count[0] >= lbfgs_steps:
            return torch.tensor(0.0, device=device)
        lbfgs.zero_grad()

        # DATA ONLY — preserves the good data fit from Adam
        # Physics was already baked in during Phase 1b; L-BFGS just
        # sharpens the sensor-data fit without re-introducing the
        # large PDE gradient that previously caused regression.
        L_data = data_loss(model, obs, device)
        loss   = lambda_data * L_data
        loss.backward()

        step_count[0] += 1
        if verbose and step_count[0] % 50 == 0:
            print(f"  L-BFGS step {step_count[0]:4d}  data={L_data.item():.4e}")
        return loss

    lbfgs.step(closure)

    # ================================================================
    # Save checkpoint
    # ================================================================
    torch.save({"model_state": model.state_dict(),
                "dem_shape": dem.shape}, CHECKPOINT)
    print(f"\n[train] Checkpoint saved -> {CHECKPOINT}")

    return model


if __name__ == "__main__":
    start = time.time()
    model = train()
    elapsed = time.time() - start
    print(f"\n[train] Done in {elapsed/60:.1f} min")
