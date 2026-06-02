"""
Evacuation route finder using Dijkstra on a flood-penalised grid graph.

Grid graph:
  Nodes  : (row, col) cells on a 64×64 terrain grid
  Edges  : 4-directional adjacency (N/S/E/W)
  Weight : 1.0 + 10.0 × h(node)   — flooded cells are expensive
           Danger cells (h ≥ 1.5m): weight = 1 000 000  (impassable)

Start presets (real 2005 flood hotspots, normalised grid coords):
  Santacruz : grid row≈42, col≈32  (worst flooded, central-north)
  Dharavi   : grid row≈35, col≈25  (low-lying slum, Mithi River)
  Khar      : grid row≈38, col≈20  (documented hotspot)

Exit direction: south (row -> 0), toward Colaba (stayed dry in 2005).

Saves: outputs/evacuation_path.json  [{x, y, risk_level, h}, ...]
"""

from __future__ import annotations


import os
import sys
import json
import numpy as np
import networkx as nx

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inference.risk_zones import depth_to_risk, RISK_LABELS

DANGER_WEIGHT  = 1_000_000.0
FLOOD_PENALTY  = 10.0
DANGER_THRESH  = 1.5   # metres

# Real 2005 hotspot presets — (row, col) on 64×64 grid
PRESETS = {
    "Santacruz": (41, 32),
    "Dharavi":   (35, 24),
    "Khar":      (37, 19),
}


def build_graph(peak_h: np.ndarray) -> nx.DiGraph:
    """Build a directed grid graph with flood-penalised edge weights."""
    H, W = peak_h.shape
    G = nx.DiGraph()

    for r in range(H):
        for c in range(W):
            h = peak_h[r, c]
            node_w = (DANGER_WEIGHT if h >= DANGER_THRESH
                      else 1.0 + FLOOD_PENALTY * h)
            # Add 4-directional edges to neighbours
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    nh = peak_h[nr, nc]
                    target_w = (DANGER_WEIGHT if nh >= DANGER_THRESH
                                else 1.0 + FLOOD_PENALTY * nh)
                    edge_w = (node_w + target_w) / 2.0
                    G.add_edge((r, c), (nr, nc), weight=edge_w)

    return G


def find_safe_exits(grid_size: int = 64, n_exits: int = 8) -> list[tuple[int, int]]:
    """Return cells along the southern boundary (Colaba direction)."""
    row = 2   # near south edge
    cols = np.linspace(grid_size // 4, 3 * grid_size // 4, n_exits, dtype=int)
    return [(row, int(c)) for c in cols]


def find_evacuation_route(
    start: tuple[int, int],
    peak_h: np.ndarray,
) -> tuple[list[tuple[int, int]], float]:
    """
    Run Dijkstra from start toward the best southern exit.

    Returns
    -------
    path   : list of (row, col) cells
    length : total path cost
    """
    G     = build_graph(peak_h)
    exits = find_safe_exits(peak_h.shape[0])

    best_path, best_cost = None, float("inf")
    for exit_node in exits:
        try:
            path = nx.dijkstra_path(G, start, exit_node, weight="weight")
            cost = nx.dijkstra_path_length(G, start, exit_node, weight="weight")
            if cost < best_cost:
                best_cost = cost
                best_path = path
        except nx.NetworkXNoPath:
            continue

    if best_path is None:
        raise RuntimeError(f"No passable route found from {start} to any exit. "
                           "Try a different start location.")

    return best_path, best_cost


def path_to_waypoints(
    path: list[tuple[int, int]],
    peak_h: np.ndarray,
) -> list[dict]:
    """Convert grid (row, col) path to normalised (x, y) waypoints with metadata."""
    H, W   = peak_h.shape
    risk_m = depth_to_risk(peak_h)
    waypoints = []
    for r, c in path:
        h_val    = float(peak_h[r, c])
        risk_lvl = int(risk_m[r, c])
        waypoints.append({
            "x":          round(c / (W - 1), 4),
            "y":          round(r / (H - 1), 4),
            "row":        r,
            "col":        c,
            "h":          round(h_val, 4),
            "risk_level": risk_lvl,
            "risk_label": RISK_LABELS[risk_lvl],
        })
    return waypoints


def compute_evacuation(
    start_preset: str = "Santacruz",
    start_rc=None,   # tuple(row,col) or None
    all_h_path: str | None = None,
    save: bool = True,
) -> list[dict]:
    """
    Main entry point. Loads peak flood depth, runs Dijkstra, returns waypoints.

    Parameters
    ----------
    start_preset : one of "Santacruz", "Dharavi", "Khar"
    start_rc     : override (row, col) — takes precedence over start_preset
    """
    if all_h_path is None:
        all_h_path = os.path.join(_ROOT, "outputs", "flood_frames", "all_h.npy")

    if not os.path.exists(all_h_path):
        print("[evacuate] No flood frames — running inference first …")
        from inference.predict import run_inference
        run_inference()

    all_h  = np.load(all_h_path)
    peak_h = all_h.max(axis=0)

    start = start_rc if start_rc is not None else PRESETS[start_preset]
    print(f"[evacuate] Start: {start_preset} -> grid cell {start}  "
          f"h={peak_h[start]:.3f}m")

    path, cost = find_evacuation_route(start, peak_h)
    waypoints  = path_to_waypoints(path, peak_h)

    # Stats
    risk_counts = {lbl: sum(1 for w in waypoints if w["risk_label"] == lbl)
                   for lbl in RISK_LABELS}
    print(f"[evacuate] Route length: {len(path)} cells  "
          f"cost={cost:.1f}")
    print(f"[evacuate] Risk breakdown: {risk_counts}")

    if save:
        out_path = os.path.join(_ROOT, "outputs", "evacuation_path.json")
        with open(out_path, "w") as f:
            json.dump(waypoints, f, indent=2)
        print(f"[evacuate] Saved -> {out_path}")

    return waypoints


if __name__ == "__main__":
    for preset in ["Santacruz", "Dharavi", "Khar"]:
        print(f"\n{'='*60}")
        print(f"Computing route from {preset} …")
        wps = compute_evacuation(start_preset=preset, save=(preset == "Santacruz"))
        print(f"Route has {len(wps)} waypoints")
