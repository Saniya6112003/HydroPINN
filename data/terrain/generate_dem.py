"""
Synthetic DEM fallback for Mumbai terrain.

Mimics real Mumbai geography:
- Low-lying flat center (Dharavi / Santacruz area) where water pools
- Gentle south-to-north tilt so Colaba (south) stays higher/drier
- Two Gaussian depressions representing the Mithi River basin
Output: data/terrain/dem.npy  shape=(64, 64), values in metres
"""

import numpy as np
import os


def generate_synthetic_dem(grid_size: int = 64, save: bool = True) -> np.ndarray:
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Base elevation: 0–15m, sloping gently UP from south (y=0, Colaba) to north (y=1)
    # Colaba stayed dry in 2005 — it sits slightly higher relative to central Mumbai
    base = 5.0 + 10.0 * yy

    # Depression 1: Dharavi / Santacruz basin (central-north, low-lying)
    cx1, cy1 = 0.5, 0.65
    depression1 = 8.0 * np.exp(-((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2 * 0.08 ** 2))

    # Depression 2: Mithi River mouth near Bandra-Kurla (slightly west-centre)
    cx2, cy2 = 0.38, 0.55
    depression2 = 5.0 * np.exp(-((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2 * 0.07 ** 2))

    dem = base - depression1 - depression2

    # Clip to realistic Mumbai range: 0–30m above sea level
    dem = np.clip(dem, 0.0, 30.0)

    if save:
        os.makedirs("data/terrain", exist_ok=True)
        out_path = os.path.join("data", "terrain", "dem.npy")
        np.save(out_path, dem)
        print(f"[generate_dem] Saved synthetic DEM -> {out_path}  "
              f"shape={dem.shape}  min={dem.min():.1f}m  max={dem.max():.1f}m")

    return dem


if __name__ == "__main__":
    dem = generate_synthetic_dem()
    print("Elevation stats:")
    print(f"  min  : {dem.min():.2f} m")
    print(f"  max  : {dem.max():.2f} m")
    print(f"  mean : {dem.mean():.2f} m")
    print(f"  shape: {dem.shape}")
