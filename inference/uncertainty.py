"""
MC Dropout uncertainty quantification.

Runs 50 stochastic forward passes with Dropout ON at inference time.
Returns mean and standard deviation of predicted flood depth h(x, y, t).

Saves: outputs/uncertainty_map.npy  — shape (64, 64) std dev at peak time
"""

from __future__ import annotations


import os
import sys
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.pinn import HydroPINN, enable_dropout

N_MC_PASSES = 50
CHECKPOINT  = os.path.join(_ROOT, "outputs", "pinn_checkpoint.pt")


def mc_dropout_predict(
    model: HydroPINN,
    inp: torch.Tensor,
    n_passes: int = N_MC_PASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run n_passes stochastic forward passes on inp with dropout active.

    Parameters
    ----------
    model    : HydroPINN (loaded checkpoint)
    inp      : Tensor (N, 5)
    n_passes : number of MC samples

    Returns
    -------
    h_mean : ndarray (N,)
    h_std  : ndarray (N,)
    """
    enable_dropout(model)   # keep dropout ON during inference

    samples = []
    with torch.no_grad():
        for _ in range(n_passes):
            h, _, _ = model(inp)
            samples.append(h.cpu().numpy().squeeze())

    samples = np.stack(samples, axis=0)   # (n_passes, N)
    return samples.mean(axis=0), samples.std(axis=0)


def compute_uncertainty_map(
    t_norm: float = 0.5,        # timestep for uncertainty estimate (default: peak, t=12hr)
    grid_size: int = 64,
    checkpoint: str = CHECKPOINT,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate mean and std flood depth over full grid using MC Dropout.

    Returns
    -------
    h_mean : ndarray (grid_size, grid_size)
    h_std  : ndarray (grid_size, grid_size)
    """
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    # Load model
    ckpt = torch.load(checkpoint, map_location=device)
    model = HydroPINN()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Build grid inputs
    x_lin = np.linspace(0, 1, grid_size, dtype=np.float32)
    y_lin = np.linspace(0, 1, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x_lin, y_lin)
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    t_flat = np.full_like(x_flat, t_norm)

    dem_path = os.path.join(_ROOT, "data", "terrain", "dem.npy")
    dem = (np.load(dem_path).astype(np.float32)
           if os.path.exists(dem_path)
           else np.zeros((grid_size, grid_size), dtype=np.float32))
    xi = (x_flat * (dem.shape[1] - 1)).clip(0, dem.shape[1] - 1).astype(int)
    yi = (y_flat * (dem.shape[0] - 1)).clip(0, dem.shape[0] - 1).astype(int)
    z_flat = dem[yi, xi]

    from data.rainfall.generate_rainfall import rainfall_rate
    R_flat = rainfall_rate(x_flat, y_flat, t_flat)

    def _t(a): return torch.tensor(a, dtype=torch.float32, device=device)
    inp = torch.stack([_t(x_flat), _t(y_flat), _t(t_flat),
                       _t(z_flat), _t(R_flat)], dim=1)

    print(f"[uncertainty] Running {N_MC_PASSES} MC Dropout passes …")
    h_mean_flat, h_std_flat = mc_dropout_predict(model, inp, N_MC_PASSES)

    h_mean = h_mean_flat.reshape(grid_size, grid_size)
    h_std  = h_std_flat.reshape(grid_size, grid_size)

    print(f"[uncertainty] Mean depth  : {h_mean.mean():.4f}m ± {h_mean.std():.4f}m")
    print(f"[uncertainty] Mean std dev: {h_std.mean():.4f}m  (confidence spread)")

    if save:
        out_path = os.path.join(_ROOT, "outputs", "uncertainty_map.npy")
        np.save(out_path, h_std)
        print(f"[uncertainty] Saved std dev map -> {out_path}")

    return h_mean, h_std


if __name__ == "__main__":
    mean, std = compute_uncertainty_map()
    print(f"\nUncertainty map: shape={std.shape}  "
          f"max_std={std.max():.4f}m  mean_std={std.mean():.4f}m")
