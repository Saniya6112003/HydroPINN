"""
Run trained HydroPINN over a 64×64 spatial grid for T=20 timesteps.

Saves per-timestep arrays to outputs/flood_frames/frame_{t:02d}.npy
Each file contains a dict: {"h": (64,64), "u": (64,64), "v": (64,64)}

Also saves outputs/flood_frames/all_h.npy  — shape (T, 64, 64) for quick access.
"""

from __future__ import annotations


import os
import sys
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.pinn import HydroPINN

CHECKPOINT  = os.path.join(_ROOT, "outputs", "pinn_checkpoint.pt")
FRAMES_DIR  = os.path.join(_ROOT, "outputs", "flood_frames")
GRID_SIZE   = 64
N_TIMESTEPS = 20


def _device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str = CHECKPOINT) -> HydroPINN:
    device = _device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = HydroPINN()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_grid(
    model: HydroPINN,
    dem: np.ndarray,
    t_norm: float,
    grid_size: int = GRID_SIZE,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict h, u, v on the full spatial grid at a single normalised time t_norm.

    Returns
    -------
    h, u, v : each ndarray shape (grid_size, grid_size)
    """
    device = next(model.parameters()).device

    x_lin = np.linspace(0, 1, grid_size, dtype=np.float32)
    y_lin = np.linspace(0, 1, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x_lin, y_lin)    # shape (H, W)
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    t_flat = np.full_like(x_flat, t_norm)

    # Sample terrain from DEM
    xi = (x_flat * (dem.shape[1] - 1)).clip(0, dem.shape[1] - 1).astype(int)
    yi = (y_flat * (dem.shape[0] - 1)).clip(0, dem.shape[0] - 1).astype(int)
    z_flat = dem[yi, xi]

    # Sample rainfall
    from data.rainfall.generate_rainfall import rainfall_rate
    R_flat = rainfall_rate(x_flat, y_flat, t_flat)

    # Run inference in batches
    all_h, all_u, all_v = [], [], []
    N = len(x_flat)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            def _t(a): return torch.tensor(a[start:end], dtype=torch.float32, device=device)
            inp = torch.stack([_t(x_flat), _t(y_flat), _t(t_flat),
                               _t(z_flat), _t(R_flat)], dim=1)
            h, u, v = model(inp)
            all_h.append(h.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_v.append(v.cpu().numpy())

    h_grid = np.concatenate(all_h).reshape(grid_size, grid_size)
    u_grid = np.concatenate(all_u).reshape(grid_size, grid_size)
    v_grid = np.concatenate(all_v).reshape(grid_size, grid_size)
    return h_grid, u_grid, v_grid


def run_inference(
    n_timesteps: int = N_TIMESTEPS,
    grid_size:   int = GRID_SIZE,
    checkpoint:  str = CHECKPOINT,
) -> np.ndarray:
    """
    Generate all flood frames and save to disk.
    Returns all_h array of shape (T, H, W).
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Load model
    if not os.path.exists(checkpoint):
        print("[predict] No checkpoint found — running training first …")
        from model.train import train
        train()

    model = load_model(checkpoint)
    print(f"[predict] Model loaded from {checkpoint}")

    # Load DEM
    dem_path = os.path.join(_ROOT, "data", "terrain", "dem.npy")
    if os.path.exists(dem_path):
        dem = np.load(dem_path).astype(np.float32)
    else:
        from data.terrain.generate_dem import generate_synthetic_dem
        dem = generate_synthetic_dem().astype(np.float32)
    print(f"[predict] DEM shape: {dem.shape}")

    times = np.linspace(0, 1, n_timesteps)
    all_h = np.zeros((n_timesteps, grid_size, grid_size), dtype=np.float32)

    for i, t_norm in enumerate(times):
        h, u, v = predict_grid(model, dem, t_norm, grid_size)
        all_h[i] = h

        frame_data = {"h": h, "u": u, "v": v}
        frame_path = os.path.join(FRAMES_DIR, f"frame_{i:02d}.npy")
        np.save(frame_path, frame_data)

        hr = t_norm * 24
        print(f"  frame {i:02d}  t={hr:.1f}hr  "
              f"max_h={h.max():.3f}m  mean_h={h.mean():.4f}m")

    all_h_path = os.path.join(FRAMES_DIR, "all_h.npy")
    np.save(all_h_path, all_h)
    print(f"\n[predict] Saved {n_timesteps} frames to {FRAMES_DIR}")
    print(f"[predict] all_h.npy shape: {all_h.shape}  "
          f"peak flood depth: {all_h.max():.3f}m")

    return all_h


if __name__ == "__main__":
    run_inference()
