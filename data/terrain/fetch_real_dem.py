"""
Downloads the real NASA SRTM 30m DEM for Mumbai via the `elevation` package.

Bounding box (west, south, east, north):
  (72.75, 18.85, 73.05, 19.15)
  Covers: Colaba (south, stayed dry 2005) -> Santacruz / Bandra (north, worst hit)

Output: data/terrain/dem.npy  shape=(64, 64)

Falls back to generate_dem.py automatically if download fails.
"""

import os
import sys
import numpy as np

BOUNDS = (72.75, 18.85, 73.05, 19.15)   # (west, south, east, north)
GRID_SIZE = 64
TIF_PATH = os.path.join("data", "terrain", "mumbai_dem.tif")
NPY_PATH = os.path.join("data", "terrain", "dem.npy")


def fetch_srtm_dem() -> np.ndarray:
    """Download SRTM DEM, resample to 64×64, save as .npy. Returns the array."""
    try:
        import elevation
        from scipy.ndimage import zoom
        import rasterio

        os.makedirs("data/terrain", exist_ok=True)

        print("[fetch_real_dem] Downloading SRTM DEM for Mumbai …")
        elevation.clip(bounds=BOUNDS, output=os.path.abspath(TIF_PATH))
        elevation.clean()   # remove cached tiles to save disk space

        with rasterio.open(TIF_PATH) as src:
            raw = src.read(1).astype(float)

        print(f"[fetch_real_dem] Raw DEM shape: {raw.shape}  "
              f"min={raw.min():.1f}m  max={raw.max():.1f}m")

        # Resample to 64×64
        zoom_factors = (GRID_SIZE / raw.shape[0], GRID_SIZE / raw.shape[1])
        dem_64 = zoom(raw, zoom_factors)
        dem_64 = np.clip(dem_64, 0.0, 500.0)   # clip sea/void fill artefacts

        np.save(NPY_PATH, dem_64)
        print(f"[fetch_real_dem] Saved real SRTM DEM -> {NPY_PATH}  "
              f"shape={dem_64.shape}  min={dem_64.min():.1f}m  max={dem_64.max():.1f}m")
        return dem_64

    except Exception as exc:
        print(f"[fetch_real_dem] WARNING: SRTM download failed ({exc})")
        print("[fetch_real_dem] Falling back to synthetic DEM …")
        return _fallback_synthetic()


def _fallback_synthetic() -> np.ndarray:
    """Run generate_dem.py fallback and return the array."""
    # Make sure we can import from the same package root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from data.terrain.generate_dem import generate_synthetic_dem
    return generate_synthetic_dem(save=True)


def load_dem() -> np.ndarray:
    """
    Load DEM from disk if it exists, otherwise fetch/generate it.
    This is the single entry-point used by train.py and predict.py.
    """
    if os.path.exists(NPY_PATH):
        dem = np.load(NPY_PATH)
        print(f"[load_dem] Loaded cached DEM from {NPY_PATH}  shape={dem.shape}")
        return dem
    return fetch_srtm_dem()


if __name__ == "__main__":
    dem = fetch_srtm_dem()
    print("\nFinal DEM stats:")
    print(f"  shape : {dem.shape}")
    print(f"  min   : {dem.min():.2f} m")
    print(f"  max   : {dem.max():.2f} m")
    print(f"  mean  : {dem.mean():.2f} m")
