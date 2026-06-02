# -*- coding: utf-8 -*-
from __future__ import annotations
"""
HydroPINN - Google Maps / OSRM real-road routing helper.

Builds interactive Folium HTML maps with:
  - Real Mumbai GPS coordinates for flood hotspots
  - Flood depth HeatMap layer
  - Risk zone CircleMarker layer
  - OSRM real-road routes from flood hotspots to Colaba (safe zone)
  - Google Maps deep-link buttons for turn-by-turn phone navigation
  - MiniMap, Fullscreen, LayerControl plugins
  - HTML legend
"""

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Mumbai bounding box — same as flood_map.py (keep in sync)
# ---------------------------------------------------------------------------
LAT_MIN = 18.905   # Colaba / Nariman Point (south, stayed dry)
LAT_MAX = 19.285   # Borivali (north)
LON_MIN = 72.790   # Arabian Sea coastline (west)
LON_MAX = 72.960   # Dharavi / harbour (east)

# ---------------------------------------------------------------------------
# Real GPS locations for Mumbai flood hotspots
# ---------------------------------------------------------------------------
LOCATIONS: dict[str, dict] = {
    "Santacruz": {
        "lat": 19.0844,
        "lng": 72.8494,
        "desc": "Worst flooded — 944mm/24hr",
        "color": "red",
    },
    "Dharavi": {
        "lat": 19.0411,
        "lng": 72.8544,
        "desc": "Mithi River overflow, low-lying slum",
        "color": "red",
    },
    "Khar": {
        "lat": 19.0728,
        "lng": 72.8347,
        "desc": "Documented 2005 hotspot",
        "color": "orange",
    },
    "Colaba": {
        "lat": 18.9088,
        "lng": 72.8119,
        "desc": "Elevated — stayed dry in 2005 ✓",
        "color": "green",
    },
    "Bandra": {
        "lat": 19.0596,
        "lng": 72.8295,
        "desc": "Partial flooding",
        "color": "orange",
    },
}

# Route colours: neon green / sky blue / yellow
_ROUTE_COLORS = ["#00E676", "#40C4FF", "#FFEA00"]

# ---------------------------------------------------------------------------
# Grid → GPS helpers
# ---------------------------------------------------------------------------

def grid_to_latlng(row: float, col: float, grid_size: int = 64) -> tuple[float, float]:
    """
    Convert (row, col) 64×64 grid coordinates to real Mumbai (lat, lng).

    Grid orientation:
      row 0  → LAT 18.905 (south — Colaba)
      row 63 → LAT 19.285 (north — Borivali)
      col 0  → LON 72.790 (west coast)
      col 63 → LON 72.960 (east — Dharavi)
    """
    lat = LAT_MIN + (row / (grid_size - 1)) * (LAT_MAX - LAT_MIN)
    lng = LON_MIN + (col / (grid_size - 1)) * (LON_MAX - LON_MIN)
    return lat, lng


# ---------------------------------------------------------------------------
# OSRM real-road routing
# ---------------------------------------------------------------------------

def get_osrm_route(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
) -> list[list[float]] | None:
    """
    Fetch a real driving route from OSRM (open-source routing engine).

    URL format:
      https://router.project-osrm.org/route/v1/driving/{lng1},{lat1};{lng2},{lat2}
          ?overview=full&geometries=geojson

    Returns
    -------
    List of [lat, lon] pairs along the route, or None on any failure.
    """
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{start_lng},{start_lat};{end_lng},{end_lat}"
        f"?overview=full&geometries=geojson"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            return None
        coords = data["routes"][0]["geometry"]["coordinates"]
        # OSRM returns [lng, lat] — flip to [lat, lng] for Folium
        return [[c[1], c[0]] for c in coords]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Google Maps URL builders
# ---------------------------------------------------------------------------

def get_google_maps_url(from_name: str, to_name: str = "Colaba") -> str:
    """
    Return a Google Maps directions URL for real phone navigation.

    Parameters
    ----------
    from_name : key in LOCATIONS (e.g. "Santacruz")
    to_name   : key in LOCATIONS (default "Colaba")
    """
    src = LOCATIONS[from_name]
    dst = LOCATIONS[to_name]
    return (
        f"https://www.google.com/maps/dir/"
        f"{src['lat']},{src['lng']}/"
        f"{dst['lat']},{dst['lng']}/"
    )


# ---------------------------------------------------------------------------
# Main map builder
# ---------------------------------------------------------------------------

def build_map_html(
    all_h: np.ndarray,
    risk_map: np.ndarray,
    route_names: list[str],
    timestep: int | None = None,
) -> str:
    """
    Build a Folium interactive map and return its HTML string.

    Parameters
    ----------
    all_h       : (T, H, W) flood depth array
    risk_map    : (H, W) integer risk array  0=safe 1=moderate 2=high 3=danger
    route_names : list of location names (keys in LOCATIONS) to route from Colaba
    timestep    : which time frame to display (None = peak depth)

    Returns
    -------
    HTML string (m._repr_html_())
    """
    import folium
    from folium.plugins import HeatMap, MiniMap, Fullscreen

    # Choose depth frame to display
    if timestep is None:
        h_frame = all_h.max(axis=0)
    else:
        h_frame = all_h[int(timestep)]

    grid_size = h_frame.shape[0]
    h_max = float(h_frame.max()) or 0.01

    # ------------------------------------------------------------------
    # Base map centered on Mumbai
    # ------------------------------------------------------------------
    m = folium.Map(
        location=[19.05, 72.88],
        zoom_start=12,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    # ------------------------------------------------------------------
    # Layer 1 — Flood depth HeatMap
    # ------------------------------------------------------------------
    heat_data: list[list[float]] = []
    rows_idx, cols_idx = np.where(h_frame > 0.05)
    for r, c in zip(rows_idx, cols_idx):
        lat, lng = grid_to_latlng(r, c, grid_size)
        heat_data.append([lat, lng, float(h_frame[r, c])])

    if heat_data:
        HeatMap(
            heat_data,
            name="Flood Depth Heatmap",
            min_opacity=0.3,
            radius=18,
            blur=15,
            gradient={
                "0.0":  "#ffffff",
                "0.25": "#aed6f1",
                "0.5":  "#2980b9",
                "0.75": "#c0392b",
                "1.0":  "#7b241c",
            },
        ).add_to(m)

    # ------------------------------------------------------------------
    # Layer 2 — Risk zone CircleMarkers (every 3rd cell)
    # ------------------------------------------------------------------
    _risk_colors = {
        0: "#2ecc71",   # safe
        1: "#f39c12",   # moderate
        2: "#e74c3c",   # high
        3: "#8b0000",   # danger
    }
    _risk_labels = {0: "Safe", 1: "Moderate", 2: "High", 3: "Danger"}

    risk_group = folium.FeatureGroup(name="Risk Zones", show=True)
    step = 3
    for r in range(0, grid_size, step):
        for c in range(0, grid_size, step):
            level = int(risk_map[r, c])
            color = _risk_colors[level]
            lat, lng = grid_to_latlng(r, c, grid_size)
            depth = float(h_frame[r, c])
            folium.CircleMarker(
                location=[lat, lng],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.35 if level > 0 else 0.10,
                weight=0,
                tooltip=f"{_risk_labels[level]} — {depth:.2f}m",
            ).add_to(risk_group)
    risk_group.add_to(m)

    # ------------------------------------------------------------------
    # Layer 3 — OSRM real-road routes
    # ------------------------------------------------------------------
    colaba = LOCATIONS["Colaba"]

    for idx, name in enumerate(route_names):
        if name not in LOCATIONS:
            continue

        src = LOCATIONS[name]
        route_color = _ROUTE_COLORS[idx % len(_ROUTE_COLORS)]

        coords = get_osrm_route(
            src["lat"], src["lng"],
            colaba["lat"], colaba["lng"],
        )

        route_group = folium.FeatureGroup(name=f"Route: {name} → Colaba", show=True)

        if coords:
            folium.PolyLine(
                coords,
                color=route_color,
                weight=5,
                opacity=0.92,
                tooltip=f"Evacuation: {name} → Colaba",
            ).add_to(route_group)
        else:
            # Fallback: straight line
            folium.PolyLine(
                [[src["lat"], src["lng"]], [colaba["lat"], colaba["lng"]]],
                color=route_color,
                weight=4,
                opacity=0.7,
                dash_array="8",
                tooltip=f"{name} → Colaba (straight-line fallback)",
            ).add_to(route_group)

        # Estimate depth at source grid cell
        src_row = int(round((src["lat"] - LAT_MIN) / (LAT_MAX - LAT_MIN) * (grid_size - 1)))
        src_col = int(round((src["lng"] - LON_MIN) / (LON_MAX - LON_MIN) * (grid_size - 1)))
        src_row = max(0, min(grid_size - 1, src_row))
        src_col = max(0, min(grid_size - 1, src_col))
        src_depth = float(h_frame[src_row, src_col])
        src_risk  = _risk_labels[int(risk_map[src_row, src_col])]

        gmaps_url = get_google_maps_url(name, "Colaba")

        start_popup = folium.Popup(
            f"""
            <div style="font-family:Arial,sans-serif;min-width:220px">
              <b style="font-size:14px;color:#c0392b">🚨 {name}</b><br>
              <span style="color:#555">{src['desc']}</span><br><br>
              <b>Flood depth:</b> {src_depth:.2f}m<br>
              <b>Risk level:</b>
              <span style="color:{_risk_colors[int(risk_map[src_row, src_col])]}">
                {src_risk}
              </span><br><br>
              <a href="{gmaps_url}" target="_blank"
                 style="background:#1a73e8;color:white;padding:6px 12px;
                        border-radius:4px;text-decoration:none;font-size:12px">
                Open in Google Maps
              </a>
            </div>
            """,
            max_width=280,
        )

        end_popup = folium.Popup(
            f"""
            <div style="font-family:Arial,sans-serif;min-width:220px">
              <b style="font-size:14px;color:#27ae60">✅ Colaba — Safe Zone</b><br>
              <span style="color:#555">{colaba['desc']}</span><br><br>
              <b>Flood depth:</b> ~0.0m (stayed dry in 2005)<br>
              <b>Risk level:</b>
              <span style="color:#2ecc71">Safe</span><br><br>
              <a href="{gmaps_url}" target="_blank"
                 style="background:#1a73e8;color:white;padding:6px 12px;
                        border-radius:4px;text-decoration:none;font-size:12px">
                Open in Google Maps
              </a>
            </div>
            """,
            max_width=280,
        )

        # Start marker (flood source)
        folium.Marker(
            [src["lat"], src["lng"]],
            popup=start_popup,
            tooltip=f"START: {name}",
            icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
        ).add_to(route_group)

        # End marker (Colaba)
        folium.Marker(
            [colaba["lat"], colaba["lng"]],
            popup=end_popup,
            tooltip="SAFE EXIT: Colaba",
            icon=folium.Icon(color="green", icon="ok-sign", prefix="glyphicon"),
        ).add_to(route_group)

        route_group.add_to(m)

    # ------------------------------------------------------------------
    # Plugins
    # ------------------------------------------------------------------
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen(position="topright", title="Full screen", title_cancel="Exit").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # ------------------------------------------------------------------
    # Legend HTML
    # ------------------------------------------------------------------
    legend_html = """
    <div style="
        position: fixed;
        bottom: 32px;
        left: 32px;
        z-index: 1000;
        background: rgba(255,255,255,0.96);
        padding: 14px 18px;
        border-radius: 10px;
        border: 1.5px solid #ddd;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 12.5px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.18);
        min-width: 210px;
    ">
      <b style="font-size:14px">🌊 HydroPINN — Mumbai 2005</b><br><br>
      <b>Flood Depth</b><br>
      <span style="color:#2980b9;">&#9632;</span> Heatmap (blue→red = shallow→deep)<br><br>
      <b>Risk Zones</b><br>
      <span style="color:#2ecc71;">&#9632;</span> Safe (&lt;0.3m)<br>
      <span style="color:#f39c12;">&#9632;</span> Moderate (0.3–0.8m)<br>
      <span style="color:#e74c3c;">&#9632;</span> High (0.8–1.5m)<br>
      <span style="color:#8b0000;">&#9632;</span> Danger (&gt;1.5m)<br><br>
      <b>Evacuation Routes</b><br>
      <span style="color:#00E676;">&#9644;</span> Santacruz → Colaba<br>
      <span style="color:#40C4FF;">&#9644;</span> Dharavi → Colaba<br>
      <span style="color:#FFEA00;">&#9644;</span> Khar → Colaba
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()
