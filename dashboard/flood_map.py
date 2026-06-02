"""
Generate an interactive Folium map of Mumbai flood data.

Maps the 64×64 synthetic grid onto real Mumbai GPS coordinates:
  row=0  → south (Colaba, stayed dry in 2005)
  row=63 → north (Borivali, Santacruz flood belt)
  col=0  → west (Arabian Sea coast)
  col=63 → east (Dharavi / harbour side)
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Mumbai bounding box (flood-affected suburban strip)
LAT_MIN = 18.905   # Colaba / Nariman Point (south, stayed dry)
LAT_MAX = 19.285   # Borivali (north)
LON_MIN = 72.790   # Arabian Sea coastline (west)
LON_MAX = 72.960   # Dharavi / harbour (east)

# Named locations for markers
LOCATION_MARKERS = {
    "Santacruz": {"row": 41, "col": 32, "color": "red",    "icon": "exclamation-sign"},
    "Dharavi":   {"row": 35, "col": 24, "color": "red",    "icon": "exclamation-sign"},
    "Khar":      {"row": 37, "col": 19, "color": "orange", "icon": "exclamation-sign"},
    "Colaba":    {"row":  3, "col": 32, "color": "green",  "icon": "ok-sign"},
}


def grid_to_latlng(row: float, col: float, grid_size: int = 64):
    """Convert (row, col) grid coordinates to (lat, lng)."""
    lat = LAT_MIN + (row / (grid_size - 1)) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (col / (grid_size - 1)) * (LON_MAX - LON_MIN)
    return lat, lon


def normalized_to_latlng(x: float, y: float, grid_size: int = 64):
    """Convert normalized (x,y) ∈ [0,1]² to (lat, lng)."""
    row = y * (grid_size - 1)
    col = x * (grid_size - 1)
    return grid_to_latlng(row, col, grid_size)


def build_flood_map(
    all_h: np.ndarray,
    risk_map: np.ndarray,
    evac_waypoints: list[dict] | None,
    all_routes: dict[str, list[dict]] | None = None,
    timestep: int | None = None,
) -> str:
    """
    Build a Folium map and return its HTML string.

    Parameters
    ----------
    all_h           : (T, H, W) flood depth array
    risk_map        : (H, W) risk zone array (0=safe, 1=moderate, 2=high, 3=danger)
    evac_waypoints  : list of waypoint dicts from evacuation_path.json
    all_routes      : dict of {name: waypoints} for multiple routes
    timestep        : which time frame to show (None = peak)
    """
    import folium
    from folium.plugins import HeatMap

    grid_size = all_h.shape[1]

    # Choose which depth frame to display
    if timestep is None:
        h_frame = all_h.max(axis=0)         # peak depth
        title = "Peak Flood Depth (2005 Mumbai)"
    else:
        h_frame = all_h[timestep]
        hrs = timestep * 24 / (all_h.shape[0] - 1)
        title = f"Flood Depth at Hour {hrs:.1f}"

    # Map center = Mumbai
    center_lat = (LAT_MIN + LAT_MAX) / 2
    center_lon = (LON_MIN + LON_MAX) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    # ----------------------------------------------------------------
    # Layer 1 — Flood depth HeatMap
    # ----------------------------------------------------------------
    heat_data = []
    rows, cols = np.where(h_frame > 0.1)   # only cells with meaningful depth
    for r, c in zip(rows, cols):
        lat, lon = grid_to_latlng(r, c, grid_size)
        depth = float(h_frame[r, c])
        heat_data.append([lat, lon, depth])

    if heat_data:
        HeatMap(
            heat_data,
            name="Flood Depth Heatmap",
            min_opacity=0.3,
            max_val=float(h_frame.max()),
            radius=18,
            blur=15,
            gradient={
                0.0:  "#ffffff",
                0.25: "#aed6f1",
                0.5:  "#2980b9",
                0.75: "#c0392b",
                1.0:  "#7b241c",
            },
        ).add_to(m)

    # ----------------------------------------------------------------
    # Layer 2 — Risk zone circles (sampled to keep HTML small)
    # ----------------------------------------------------------------
    risk_colors  = ["#2ecc71", "#f39c12", "#e74c3c", "#900000"]
    risk_labels  = ["Safe", "Moderate", "High", "Danger"]
    risk_opacities = [0.0, 0.15, 0.25, 0.40]

    risk_group = folium.FeatureGroup(name="Risk Zones", show=True)
    # Sample every 4th cell to avoid thousands of circles
    for r in range(0, grid_size, 4):
        for c in range(0, grid_size, 4):
            level = int(risk_map[r, c])
            if level == 0:
                continue
            lat, lon = grid_to_latlng(r, c, grid_size)
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=risk_colors[level],
                fill=True,
                fill_color=risk_colors[level],
                fill_opacity=risk_opacities[level],
                weight=0,
                tooltip=f"{risk_labels[level]} — {h_frame[r, c]:.2f}m",
            ).add_to(risk_group)
    risk_group.add_to(m)

    # ----------------------------------------------------------------
    # Layer 3 — Evacuation routes
    # ----------------------------------------------------------------
    route_colors = ["#00e676", "#40c4ff", "#ffea00", "#ff6d00"]

    def _add_route(waypoints, color, name):
        if not waypoints:
            return
        coords = [normalized_to_latlng(w["x"], w["y"], grid_size) for w in waypoints]
        route_group = folium.FeatureGroup(name=name, show=True)

        folium.PolyLine(
            coords,
            color=color,
            weight=5,
            opacity=0.9,
            tooltip=name,
        ).add_to(route_group)

        # Start marker
        start_lat, start_lon = coords[0]
        folium.Marker(
            [start_lat, start_lon],
            popup=folium.Popup(
                f"<b>START — {name}</b><br>"
                f"Flood depth: {waypoints[0]['h']:.2f}m<br>"
                f"Risk: {waypoints[0]['risk_label']}",
                max_width=200,
            ),
            icon=folium.Icon(color="red", icon="person-walking", prefix="fa"),
        ).add_to(route_group)

        # End marker
        end_lat, end_lon = coords[-1]
        folium.Marker(
            [end_lat, end_lon],
            popup=folium.Popup(
                f"<b>SAFE EXIT</b><br>"
                f"Flood depth: {waypoints[-1]['h']:.2f}m<br>"
                f"Toward Colaba (stayed dry in 2005)",
                max_width=200,
            ),
            icon=folium.Icon(color="green", icon="flag", prefix="fa"),
        ).add_to(route_group)

        route_group.add_to(m)

    if all_routes:
        for i, (route_name, wps) in enumerate(all_routes.items()):
            _add_route(wps, route_colors[i % len(route_colors)], f"Route: {route_name}")
    elif evac_waypoints:
        _add_route(evac_waypoints, route_colors[0], "Evacuation Route")

    # ----------------------------------------------------------------
    # Layer 4 — Location markers (flood hotspots)
    # ----------------------------------------------------------------
    info_group = folium.FeatureGroup(name="Mumbai 2005 Flood Hotspots", show=True)
    hotspot_info = {
        "Santacruz": "Worst flooded in 2005. Received 944mm in 24hr.",
        "Dharavi":   "World's largest slum. Low-lying, Mithi River overflow.",
        "Khar":      "Documented flood hotspot, July 2005.",
        "Colaba":    "Stayed relatively dry — main evacuation target.",
    }
    for name, cfg in LOCATION_MARKERS.items():
        lat, lon = grid_to_latlng(cfg["row"], cfg["col"])
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(
                f"<b>{name}</b><br>{hotspot_info.get(name, '')}",
                max_width=250,
            ),
            tooltip=name,
            icon=folium.Icon(color=cfg["color"], icon=cfg["icon"], prefix="glyphicon"),
        ).add_to(info_group)
    info_group.add_to(m)

    # ----------------------------------------------------------------
    # Title + legend HTML
    # ----------------------------------------------------------------
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                border: 2px solid #ccc; font-family: Arial, sans-serif; font-size: 13px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b>HydroPINN — Mumbai 2005 Flood</b><br>
        <span style="color:#2980b9;">■</span> Flood Heatmap<br>
        <span style="color:#2ecc71;">■</span> Safe (h &lt; 0.3m)<br>
        <span style="color:#f39c12;">■</span> Moderate (0.3–0.8m)<br>
        <span style="color:#e74c3c;">■</span> High (0.8–1.5m)<br>
        <span style="color:#900000;">■</span> Danger (h &gt; 1.5m)<br>
        <span style="color:#00e676;">―</span> Evacuation Route
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    return m._repr_html_()
