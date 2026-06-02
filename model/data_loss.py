"""
Data loss: MSE between PINN predictions and sparse "observed" flood depth points.

Simulates 200 IoT river gauge / rain sensor readings scattered across Mumbai.
Ground-truth h values are generated analytically from rainfall + terrain — in a
real deployment these would come from actual sensor telemetry or a high-fidelity
hydraulic simulation used as a surrogate.

L_data = mean( (h_pred - h_obs)² )
"""

import torch
import numpy as np


def _analytical_h(x: np.ndarray,
                  y: np.ndarray,
                  t: np.ndarray,
                  z: np.ndarray) -> np.ndarray:
    """
    Simple physics-inspired analytical approximation of flood depth.

    h ∝ (rainfall accumulation) / (terrain elevation + ε)
    — low terrain + high rainfall -> deep flooding (consistent with Dharavi/Santacruz 2005)
    — high terrain (Colaba ridge) -> very shallow despite any rain

    This is intentionally simplified; the PINN learns to match it while also
    satisfying the full SWE — a harder, more physically consistent constraint.
    """
    from data.rainfall.generate_rainfall import rainfall_rate

    R = rainfall_rate(x, y, t)   # mm/hr

    # Accumulate rain over [0, t] — rough integral: R_avg × t × 24hr (in mm)
    rain_accum_mm = R * t * 24.0
    rain_accum_m  = rain_accum_mm / 1000.0   # convert to metres

    # Terrain drainage: low terrain retains more water (Mumbai's low-lying areas)
    # ε = 0.5m so very flat terrain (z<1m) gets deepest flooding
    terrain_factor = 1.0 / (z * 0.3 + 0.5)

    # Scale to produce realistic 2005 depths: 0-2m in worst zones
    h_obs = rain_accum_m * terrain_factor * 8.0
    h_obs = np.clip(h_obs, 0.0, 2.5)   # cap at 2.5m (extreme urban flood)
    return h_obs.astype(np.float32)


def generate_observation_points(
    n_obs: int = 200,
    seed: int = 42,
    dem=None,   # np.ndarray or None
) -> dict:
    """
    Generate n_obs sparse observation points with ground-truth flood depths.

    Returns a dict with keys: x, y, t, z, R, h_obs  — all numpy float32 arrays.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 1, n_obs).astype(np.float32)
    y = rng.uniform(0, 1, n_obs).astype(np.float32)
    t = rng.uniform(0, 1, n_obs).astype(np.float32)

    # Bilinear sample of DEM at (x, y) observation locations
    if dem is not None:
        xi = (x * (dem.shape[1] - 1)).astype(int).clip(0, dem.shape[1] - 1)
        yi = (y * (dem.shape[0] - 1)).astype(int).clip(0, dem.shape[0] - 1)
        z = dem[yi, xi].astype(np.float32)
    else:
        z = (5.0 + 10.0 * y).astype(np.float32)   # simple linear fallback

    from data.rainfall.generate_rainfall import rainfall_rate
    R = rainfall_rate(x, y, t).astype(np.float32)

    h_obs = _analytical_h(x, y, t, z)

    return {"x": x, "y": y, "t": t, "z": z, "R": R, "h_obs": h_obs}


def data_loss(
    model,
    obs: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute MSE between PINN-predicted h and observed h at sparse sensor points.

    Parameters
    ----------
    model : HydroPINN
    obs   : dict returned by generate_observation_points()
    device: torch.device

    Returns
    -------
    L_data : scalar Tensor
    """
    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    inp = torch.stack([
        _t(obs["x"]), _t(obs["y"]), _t(obs["t"]),
        _t(obs["z"]), _t(obs["R"]),
    ], dim=1)

    h_pred, _, _ = model(inp)
    h_target = _t(obs["h_obs"]).unsqueeze(1)

    return torch.nn.functional.mse_loss(h_pred, h_target)
