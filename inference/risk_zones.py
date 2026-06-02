"""
Map flood depth grid -> 4-level risk zone grid.

Risk levels (integer codes):
  0 — Safe      (h < 0.3m)   green   — walkable
  1 — Moderate  (0.3–0.8m)   yellow  — caution
  2 — High      (0.8–1.5m)   orange  — evacuate
  3 — Danger    (h ≥ 1.5m)   red     — impassable

Saves: outputs/risk_map.npy  (64×64 int array, values 0–3)
"""

import os
import sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Depth thresholds in metres
THRESHOLDS = [0.3, 0.8, 1.5]

RISK_LABELS  = ["Safe", "Moderate", "High", "Danger"]
RISK_COLORS  = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]   # green/yellow/orange/red
RISK_CODES   = [0, 1, 2, 3]


def depth_to_risk(h: np.ndarray) -> np.ndarray:
    """
    Convert a flood depth array (any shape) to integer risk codes 0–3.

    Parameters
    ----------
    h : ndarray — water depth in metres

    Returns
    -------
    risk : ndarray int8, same shape as h
    """
    risk = np.zeros_like(h, dtype=np.int8)
    risk[h >= THRESHOLDS[0]] = 1
    risk[h >= THRESHOLDS[1]] = 2
    risk[h >= THRESHOLDS[2]] = 3
    return risk


def compute_risk_map(
    all_h_path=None,   # str path or None
    save: bool = True,
) -> np.ndarray:
    """
    Load flood frames, compute peak depth per cell, map to risk codes.

    Returns
    -------
    risk_map : ndarray int8, shape (64, 64)
    """
    if all_h_path is None:
        all_h_path = os.path.join(_ROOT, "outputs", "flood_frames", "all_h.npy")

    if not os.path.exists(all_h_path):
        print("[risk_zones] No flood frames found — running inference …")
        from inference.predict import run_inference
        all_h = run_inference()
    else:
        all_h = np.load(all_h_path)   # shape (T, H, W)

    # Peak flood depth over all timesteps
    peak_h = all_h.max(axis=0)        # shape (H, W)

    risk_map = depth_to_risk(peak_h)

    if save:
        out_path = os.path.join(_ROOT, "outputs", "risk_map.npy")
        np.save(out_path, risk_map)
        print(f"[risk_zones] Saved risk map -> {out_path}  shape={risk_map.shape}")
        _print_stats(risk_map)

    return risk_map


def _print_stats(risk_map: np.ndarray):
    total = risk_map.size
    for code, label in zip(RISK_CODES, RISK_LABELS):
        count = (risk_map == code).sum()
        pct = 100 * count / total
        print(f"  {label:10s} (code={code}): {count:5d} cells  ({pct:.1f}%)")


def risk_summary(risk_map: np.ndarray) -> dict:
    """Return a dict with cell counts and percentages per risk level."""
    total = risk_map.size
    return {
        label: {
            "count": int((risk_map == code).sum()),
            "pct":   float(100 * (risk_map == code).sum() / total),
            "color": color,
        }
        for code, label, color in zip(RISK_CODES, RISK_LABELS, RISK_COLORS)
    }


if __name__ == "__main__":
    rm = compute_risk_map()
    print(f"\nRisk map stats:")
    _print_stats(rm)
