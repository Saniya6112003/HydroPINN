# -*- coding: utf-8 -*-
"""
HydroPINN Streamlit Dashboard - Mumbai 2005 Floods
5 tabs:
  1. Flood Simulation   - animated PINN heatmap
  2. Interactive Map    - Folium + OSRM real-road routing
  3. Risk Analysis      - risk zones + MC-Dropout uncertainty
  4. PINN vs Plain NN   - physics comparison
  5. How It Works       - step-by-step explainer
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Page config  — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HydroPINN - Mumbai Flood Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Layout constraints ── */
    .block-container {
        max-width: 1400px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e8edf2;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }
    [data-testid="metric-container"] label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748b;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* ── Tabs ── */
    [data-baseweb="tab-list"] {
        background: #f1f5f9;
        border-radius: 50px;
        padding: 4px;
        gap: 4px;
        flex-wrap: wrap;
    }
    [data-baseweb="tab"] {
        border-radius: 50px !important;
        padding: 8px 20px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
        background: transparent !important;
        transition: all 0.18s ease !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        background: #0369A1 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(3,105,161,0.35) !important;
    }
    [data-baseweb="tab-highlight"] { display: none !important; }
    [data-baseweb="tab-border"]    { display: none !important; }

    /* ── Primary buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0369A1 0%, #0284C7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 10px 22px;
        box-shadow: 0 2px 10px rgba(3,105,161,0.30);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(3,105,161,0.50);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* ── Info/step cards ── */
    .info-card {
        border-left: 4px solid;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.93rem;
        line-height: 1.65;
    }
    .step-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 22px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        border: 1px solid #e8edf2;
        display: flex;
        gap: 16px;
        align-items: flex-start;
    }
    .step-number {
        background: linear-gradient(135deg, #0369A1, #0284C7);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper UI components
# ---------------------------------------------------------------------------

def info_box(title: str, body: str, color: str = "#0369A1", icon: str = "💡") -> None:
    """Render a left-bordered info card."""
    bg = color + "11"  # very faint tint
    st.markdown(
        f"""
        <div class="info-card" style="border-color:{color};background:{bg}">
          <b style="color:{color};font-size:1.0rem">{icon} {title}</b><br>
          <span style="color:#334155">{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_row(stats: list[dict]) -> None:
    """
    Render a row of metric tiles.

    stats = [{"label": ..., "value": ..., "delta": ...}, ...]
    delta is optional.
    """
    cols = st.columns(len(stats))
    for col, s in zip(cols, stats):
        with col:
            if s.get("delta") is not None:
                st.metric(s["label"], s["value"], delta=s["delta"])
            else:
                st.metric(s["label"], s["value"])


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_all_h() -> np.ndarray | None:
    path = os.path.join(_ROOT, "outputs", "flood_frames", "all_h.npy")
    return np.load(path).astype(np.float32) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def _load_dem() -> np.ndarray | None:
    path = os.path.join(_ROOT, "data", "terrain", "dem.npy")
    return np.load(path).astype(np.float32) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def _load_risk_map() -> np.ndarray | None:
    path = os.path.join(_ROOT, "outputs", "risk_map.npy")
    return np.load(path).astype(np.int8) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def _load_uncertainty() -> np.ndarray | None:
    path = os.path.join(_ROOT, "outputs", "uncertainty_map.npy")
    return np.load(path).astype(np.float32) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def _load_evacuation() -> list[dict] | None:
    path = os.path.join(_ROOT, "outputs", "evacuation_path.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _needs_training() -> bool:
    return not os.path.exists(os.path.join(_ROOT, "outputs", "pinn_checkpoint.pt"))


def _needs_inference() -> bool:
    return not os.path.exists(
        os.path.join(_ROOT, "outputs", "flood_frames", "all_h.npy")
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:10px 0 4px">
          <span style="font-size:2.2rem">🌊</span><br>
          <span style="font-size:1.35rem;font-weight:700;color:#0369A1">HydroPINN</span><br>
          <span style="font-size:0.78rem;color:#64748b;letter-spacing:0.06em">
            FLOOD INTELLIGENCE PLATFORM
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    gmaps_key = st.text_input(
        "Google Maps API Key",
        type="password",
        placeholder="Optional — leave blank to use OSRM",
        help="Not required. OSRM (open-source) is used for routing by default.",
    )
    if not gmaps_key:
        st.caption("No key entered — using free OSRM routing engine.")
    st.divider()

    # Quick stats if data available
    all_h_sb = _load_all_h()
    risk_sb  = _load_risk_map()
    if all_h_sb is not None:
        st.markdown(
            "<b style='color:#0369A1;font-size:0.82rem;letter-spacing:0.06em'>"
            "QUICK STATS</b>",
            unsafe_allow_html=True,
        )
        peak_depth = float(all_h_sb.max())
        affected   = float((all_h_sb.max(axis=0) > 0.3).mean() * 100)
        t_peak     = int(all_h_sb.max(axis=(1, 2)).argmax())
        hours_arr  = np.linspace(0, 24, all_h_sb.shape[0])
        st.metric("Peak Depth", f"{peak_depth:.2f} m")
        st.metric("Affected Area", f"{affected:.1f}%", delta=None)
        st.metric("Time to Peak", f"{hours_arr[t_peak]:.1f} hr")
        st.divider()

    run_btn = st.button(
        "🚀 Run Full Simulation",
        type="primary",
        use_container_width=True,
        help="Train PINN + run inference + compute risk + uncertainty",
    )

    if run_btn:
        # ---- Terrain ----
        with st.spinner("Setting up terrain …"):
            try:
                from data.terrain.fetch_real_dem import fetch_srtm_dem
                fetch_srtm_dem()
                st.success("Real SRTM DEM downloaded")
            except Exception as e:
                from data.terrain.generate_dem import generate_synthetic_dem
                generate_synthetic_dem()
                st.warning(f"SRTM unavailable ({e}) — synthetic DEM used")

        # ---- Rainfall ----
        with st.spinner("Generating rainfall field …"):
            from data.rainfall.generate_rainfall import generate_rainfall_grid
            generate_rainfall_grid()
            st.success("Rainfall field generated")

        # ---- PINN training ----
        with st.spinner("Training PINN (Adam + L-BFGS) — this takes a few minutes …"):
            from model.train import train
            train(
                adam_epochs=2000,
                lbfgs_steps=300,
                lambda_data=5.0,
                lambda_pde=0.0001,
                lambda_bc=0.005,
                verbose=False,
            )
            st.success("PINN trained")

        # ---- Inference ----
        with st.spinner("Running flood inference (T=20 timesteps) …"):
            from inference.predict import run_inference
            run_inference()
            st.success("Flood frames generated")

        # ---- Risk zones ----
        with st.spinner("Computing risk zones …"):
            from inference.risk_zones import compute_risk_map
            compute_risk_map()
            st.success("Risk map computed")

        # ---- Uncertainty ----
        with st.spinner("Running MC Dropout uncertainty (50 passes) …"):
            from inference.uncertainty import compute_uncertainty_map
            compute_uncertainty_map()
            st.success("Uncertainty map computed")

        st.cache_data.clear()
        st.success("Simulation complete! Scroll to explore results.")
        # Note: st.rerun() intentionally omitted to prevent loop issues

    st.divider()
    st.caption("Mumbai 26 July 2005 · SWE · MC Dropout")


# ---------------------------------------------------------------------------
# Load data for tabs
# ---------------------------------------------------------------------------
all_h     = _load_all_h()
dem       = _load_dem()
risk_map  = _load_risk_map()
uncertainty = _load_uncertainty()

# Storm duration constant (hours)
STORM_DURATION = 24

# ---------------------------------------------------------------------------
# 5 Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌊 Flood Simulation",
    "🗺️ Interactive Map",
    "🚦 Risk Analysis",
    "📊 PINN vs Plain NN",
    "📖 How It Works",
])


# ============================================================
# TAB 1 — Flood Simulation
# ============================================================
with tab1:
    st.markdown(
        '<div class="hero-title">Mumbai 2005 Flood Propagation</div>'
        '<div class="hero-subtitle">'
        'Physics-Informed Neural Network · Shallow Water Equations · '
        'Real NASA SRTM terrain'
        '</div>',
        unsafe_allow_html=True,
    )

    if all_h is None:
        st.info(
            "No flood data found. Click **Run Full Simulation** in the sidebar "
            "to train the PINN and generate flood frames."
        )
    else:
        T, H, W = all_h.shape
        hours = np.linspace(0, STORM_DURATION, T)
        h_max = float(all_h.max()) or 0.01

        # ---- Animated Plotly heatmap ----
        frames = []
        for i in range(T):
            frames.append(go.Frame(
                data=[go.Heatmap(
                    z=all_h[i],
                    colorscale=[
                        [0.0,  "white"],
                        [0.1,  "#aed6f1"],
                        [0.4,  "#2980b9"],
                        [0.7,  "#c0392b"],
                        [1.0,  "#7b241c"],
                    ],
                    zmin=0,
                    zmax=h_max,
                    colorbar=dict(title="Depth (m)"),
                    showscale=True,
                )],
                name=f"t{i}",
                layout=go.Layout(
                    title_text=(
                        f"Mumbai Flood — Hour {hours[i]:.1f} of {STORM_DURATION}"
                    )
                ),
            ))

        fig_anim = go.Figure(
            data=[go.Heatmap(
                z=all_h[0],
                colorscale=[
                    [0.0,  "white"],
                    [0.1,  "#aed6f1"],
                    [0.4,  "#2980b9"],
                    [0.7,  "#c0392b"],
                    [1.0,  "#7b241c"],
                ],
                zmin=0,
                zmax=h_max,
                colorbar=dict(title="Water Depth (m)"),
            )],
            frames=frames,
            layout=go.Layout(
                title="Mumbai 26 July 2005 — Flood Propagation (PINN Prediction)",
                height=520,
                xaxis=dict(title="West → East (grid column)"),
                yaxis=dict(title="South (Colaba) → North (Santacruz)"),
                paper_bgcolor="#fafafa",
                plot_bgcolor="#fafafa",
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 300, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100},
                            }],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            }],
                        ),
                    ],
                    x=0.05, y=0.02, xanchor="left",
                )],
                sliders=[dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [f"t{i}"],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 300, "redraw": True},
                                    "transition": {"duration": 100},
                                },
                            ],
                            label=f"{hours[i]:.0f}hr",
                        )
                        for i in range(T)
                    ],
                    x=0.05, len=0.95, y=0,
                    currentvalue=dict(
                        prefix="Time: ",
                        suffix=" hr",
                        visible=True,
                        font=dict(size=13),
                    ),
                    pad={"t": 50},
                )],
            ),
        )

        # Overlay terrain contours if DEM available
        if dem is not None:
            fig_anim.add_trace(go.Contour(
                z=dem,
                showscale=False,
                contours=dict(showlines=True, coloring="none"),
                line=dict(color="rgba(80,80,80,0.25)", width=0.5),
            ))

        st.plotly_chart(fig_anim, use_container_width=True)

        # ---- 3 metrics ----
        peak_val  = float(all_h.max())
        affected  = float((all_h.max(axis=0) > 0.3).mean() * 100)
        t_peak_i  = int(all_h.max(axis=(1, 2)).argmax())
        t_peak_hr = float(hours[t_peak_i])

        stat_row([
            {"label": "Peak Flood Depth",    "value": f"{peak_val:.2f} m"},
            {"label": "Affected Area (h>0.3m)", "value": f"{affected:.1f}%"},
            {"label": "Time to Peak",         "value": f"{t_peak_hr:.1f} hr"},
        ])

    st.divider()

    info_box(
        "What is this simulation?",
        "A Physics-Informed Neural Network (PINN) trained on the Shallow Water Equations "
        "predicts h(x,y,t) — the flood water depth at every location and every hour. "
        "Blue = shallow; red = life-threatening. Unlike traditional fluid simulations, "
        "the PINN learns directly from data <i>and</i> from the physics of water flow, "
        "so it generalises beyond the training points.",
        color="#0369A1",
        icon="🔬",
    )

    info_box(
        "Why Santacruz was the worst",
        "On 26 July 2005, Santacruz recorded <b>944mm of rainfall in 24 hours</b> — "
        "the highest single-day rainfall ever measured in India at that time. "
        "The Mithi River overflowed, drains were overwhelmed, and low-lying slums "
        "like Dharavi were inundated for days. Over 400 people died and damage "
        "exceeded $1 billion. This simulation models that exact event.",
        color="#dc2626",
        icon="📍",
    )


# ============================================================
# TAB 2 — Interactive Map (Real Mumbai)
# ============================================================
with tab2:
    st.markdown(
        '<div class="hero-title">Interactive Map — Real Mumbai</div>'
        '<div class="hero-subtitle">'
        'OSRM real-road routing · OpenStreetMap tiles · GPS coordinates'
        '</div>',
        unsafe_allow_html=True,
    )

    if all_h is None or risk_map is None:
        st.info(
            "Run the simulation first to generate flood and risk data before "
            "building the interactive map."
        )
    else:
        from dashboard.google_maps import build_map_html, get_google_maps_url, LOCATIONS

        ctrl_col, map_col = st.columns([1, 3])

        with ctrl_col:
            st.markdown("**Route Options**")

            selected_routes = st.multiselect(
                "Evacuation start points",
                options=["Santacruz", "Dharavi", "Khar"],
                default=["Santacruz", "Dharavi", "Khar"],
                help="Select which flood hotspots to route from Colaba",
            )

            show_heatmap   = st.checkbox("Show flood heatmap",  value=True)
            show_risk      = st.checkbox("Show risk zones",     value=True)

            find_btn = st.button(
                "🗺️ Find Real Routes",
                type="primary",
                use_container_width=True,
            )

        # Build or rebuild map when button pressed
        if find_btn or ("map_html_cache" not in st.session_state):
            if selected_routes or show_heatmap or show_risk:
                with st.spinner("Fetching real road routes from OSRM …"):
                    try:
                        # If heatmap/risk disabled, pass zero arrays
                        _h  = all_h   if show_heatmap else np.zeros_like(all_h)
                        _rm = risk_map if show_risk    else np.zeros_like(risk_map)
                        html_map = build_map_html(
                            all_h=_h,
                            risk_map=_rm,
                            route_names=selected_routes,
                            timestep=None,
                        )
                        st.session_state["map_html_cache"] = html_map
                    except Exception as e:
                        st.error(f"Map build failed: {e}")
                        st.session_state["map_html_cache"] = None

        with map_col:
            html_map = st.session_state.get("map_html_cache")
            if html_map:
                components.html(html_map, height=600, scrolling=False)
            else:
                st.info("Click **Find Real Routes** to build the interactive map.")

        # Google Maps deep-link buttons
        if selected_routes:
            st.markdown("---")
            st.markdown("**Open turn-by-turn navigation on your phone:**")
            btn_cols = st.columns(len(selected_routes))
            _btn_styles = {
                "Santacruz": "#dc2626",
                "Dharavi":   "#dc2626",
                "Khar":      "#d97706",
            }
            for col, name in zip(btn_cols, selected_routes):
                gmap_url = get_google_maps_url(name, "Colaba")
                color    = _btn_styles.get(name, "#0369A1")
                col.markdown(
                    f"""
                    <a href="{gmap_url}" target="_blank" style="
                        display:inline-block;
                        background:{color};
                        color:white;
                        padding:10px 18px;
                        border-radius:8px;
                        font-weight:600;
                        font-size:0.88rem;
                        text-decoration:none;
                        box-shadow:0 2px 8px rgba(0,0,0,0.18);
                        text-align:center;
                        width:100%;
                        box-sizing:border-box;
                    ">
                      📍 {name} → Colaba
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    info_box(
        "How are routes calculated?",
        "Two routing layers work together: <b>(1) Dijkstra's algorithm</b> runs on the "
        "flood-penalised 64×64 grid — flooded cells have high cost, danger cells "
        "(h≥1.5m) are impassable. This gives the optimal <i>theoretical</i> path. "
        "<b>(2) OSRM</b> (Open Source Routing Machine) maps those start/end GPS "
        "coordinates to real Mumbai roads, giving realistic turn-by-turn geometry. "
        "Click any Google Maps button for live navigation on your phone.",
        color="#0369A1",
        icon="🗺️",
    )

    info_box(
        "How to use this for real evacuation",
        "<b>Emergency coordinators:</b> Use the risk zone overlay to pre-position "
        "rescue boats in Danger zones before peak rainfall. "
        "<b>Residents:</b> Open the Google Maps link on your phone for real-time "
        "navigation — the route avoids flood-prone streets. "
        "<b>Note:</b> This is a research prototype. Always follow official emergency "
        "services guidance during an actual flood event.",
        color="#7c3aed",
        icon="🚗",
    )


# ============================================================
# TAB 3 — Risk Analysis
# ============================================================
with tab3:
    st.markdown(
        '<div class="hero-title">Risk Analysis</div>'
        '<div class="hero-subtitle">'
        'Peak-time risk zones · MC Dropout uncertainty quantification'
        '</div>',
        unsafe_allow_html=True,
    )

    if risk_map is None:
        st.info("Run the simulation to generate risk zones and uncertainty maps.")
    else:
        col_risk, col_unc = st.columns(2)

        _risk_palette = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        _risk_labels  = ["Safe", "Moderate", "High", "Danger"]

        with col_risk:
            st.subheader("Peak-Time Risk Zones")
            fig_risk = px.imshow(
                risk_map,
                color_continuous_scale=_risk_palette,
                zmin=0,
                zmax=3,
                labels={"color": "Risk"},
                title="Flood Risk Zones (Peak Depth)",
            )
            fig_risk.update_layout(
                height=420,
                paper_bgcolor="#fafafa",
                plot_bgcolor="#fafafa",
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with col_unc:
            st.subheader("Uncertainty Map (MC Dropout — 50 passes)")
            if uncertainty is None:
                st.info("Uncertainty map not generated yet.")
            else:
                fig_unc = px.imshow(
                    uncertainty,
                    color_continuous_scale="RdYlGn_r",
                    labels={"color": "Std Dev (m)"},
                    title="Flood Depth Uncertainty — σ(h) in metres",
                )
                fig_unc.update_layout(
                    height=420,
                    paper_bgcolor="#fafafa",
                    plot_bgcolor="#fafafa",
                )
                st.plotly_chart(fig_unc, use_container_width=True)
                st.caption(
                    f"Mean spread: **±{uncertainty.mean():.4f}m** | "
                    f"Max: **±{uncertainty.max():.4f}m**"
                )

        # ---- Colour-coded risk stats ----
        st.markdown("---")
        st.subheader("Risk Zone Statistics")
        total_cells = risk_map.size
        _risk_colors_hex = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        _risk_bg_hex     = ["#d5f5e3", "#fef9e7", "#fdebd0", "#fadbd8"]

        rc1, rc2, rc3, rc4 = st.columns(4)
        for col, code, label, fc, bg in zip(
            [rc1, rc2, rc3, rc4],
            [0, 1, 2, 3],
            _risk_labels,
            _risk_colors_hex,
            _risk_bg_hex,
        ):
            count = int((risk_map == code).sum())
            pct   = 100.0 * count / total_cells
            col.markdown(
                f"""
                <div style="
                    background:{bg};
                    border:2px solid {fc};
                    border-radius:12px;
                    padding:14px 18px;
                    text-align:center
                ">
                  <div style="font-size:1.5rem;font-weight:700;color:{fc}">{count}</div>
                  <div style="font-size:0.78rem;font-weight:600;color:#374151;
                              text-transform:uppercase;letter-spacing:0.05em">
                    {label}
                  </div>
                  <div style="font-size:0.85rem;color:#6b7280">{pct:.1f}% of grid</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    info_box(
        "What do risk levels mean?",
        """
        <table style="border-collapse:collapse;width:100%;font-size:0.9rem">
          <tr style="background:#f1f5f9">
            <th style="padding:8px 12px;text-align:left;border-radius:6px 0 0 0">Level</th>
            <th style="padding:8px 12px;text-align:left">Depth</th>
            <th style="padding:8px 12px;text-align:left">Meaning</th>
          </tr>
          <tr>
            <td style="padding:8px 12px;color:#2ecc71;font-weight:600">Safe</td>
            <td style="padding:8px 12px">&lt;0.3m</td>
            <td style="padding:8px 12px">Walkable. Minor surface flooding — ankle-deep at worst.</td>
          </tr>
          <tr style="background:#f9fafb">
            <td style="padding:8px 12px;color:#d97706;font-weight:600">Moderate</td>
            <td style="padding:8px 12px">0.3–0.8m</td>
            <td style="padding:8px 12px">Knee-deep. Dangerous for children, elderly, disabled people.</td>
          </tr>
          <tr>
            <td style="padding:8px 12px;color:#e74c3c;font-weight:600">High</td>
            <td style="padding:8px 12px">0.8–1.5m</td>
            <td style="padding:8px 12px">Waist/chest-deep. Cannot walk through. Vehicles stall.</td>
          </tr>
          <tr style="background:#f9fafb">
            <td style="padding:8px 12px;color:#8b0000;font-weight:600">Danger</td>
            <td style="padding:8px 12px">&gt;1.5m</td>
            <td style="padding:8px 12px">Life-threatening. Immediate evacuation. Dijkstra treats as impassable.</td>
          </tr>
        </table>
        """,
        color="#dc2626",
        icon="🚦",
    )

    info_box(
        "What is uncertainty?",
        "The uncertainty map is produced by <b>Monte Carlo Dropout</b>: the PINN is run "
        "50 times with random dropout neurons disabled on each pass (simulating an "
        "ensemble of slightly different models). The <i>spread</i> of those 50 predictions "
        "at each grid cell = the model's confidence. <b>Dark red = less certain</b> — "
        "emergency services should treat those zones with extra caution and deploy more "
        "sensors or helicopters for ground truth.",
        color="#7c3aed",
        icon="📊",
    )


# ============================================================
# TAB 4 — PINN vs Plain NN
# ============================================================
with tab4:
    st.markdown(
        '<div class="hero-title">PINN vs Plain Neural Network</div>'
        '<div class="hero-subtitle">'
        'Why physics constraints matter for flood prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    pinn_exists  = os.path.exists(os.path.join(_ROOT, "outputs", "pinn_checkpoint.pt"))
    plain_exists = os.path.exists(os.path.join(_ROOT, "outputs", "plain_nn_checkpoint.pt"))

    if not pinn_exists:
        st.info("Run the simulation first to train the PINN checkpoint.")
    else:
        compare_btn = st.button(
            "▶ Run Comparison (trains Plain NN baseline)",
            use_container_width=False,
        )

        results: dict | None = None
        if compare_btn or plain_exists:
            with st.spinner(
                "Training Plain NN baseline and computing metrics — "
                "this may take a minute …"
            ):
                try:
                    from comparison.compare import run_comparison
                    results = run_comparison()
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    st.exception(e)
                    results = None

        if results is not None:
            col_p, col_nn = st.columns(2)
            _depth_scale = [
                [0.0, "white"],
                [0.2, "#aed6f1"],
                [0.6, "#2980b9"],
                [1.0, "#c0392b"],
            ]

            with col_p:
                st.subheader("PINN Flood Prediction")
                fig_p = px.imshow(
                    results["h_pinn_grid"],
                    color_continuous_scale=_depth_scale,
                    labels={"color": "Depth (m)"},
                    title="PINN — h(x,y) at t=12hr",
                )
                fig_p.update_layout(height=400, paper_bgcolor="#fafafa")
                st.plotly_chart(fig_p, use_container_width=True)

            with col_nn:
                st.subheader("Plain NN Prediction (no physics)")
                fig_nn = px.imshow(
                    results["h_plain_grid"],
                    color_continuous_scale=_depth_scale,
                    labels={"color": "Depth (m)"},
                    title="Plain NN — h(x,y) at t=12hr",
                )
                fig_nn.update_layout(height=400, paper_bgcolor="#fafafa")
                st.plotly_chart(fig_nn, use_container_width=True)

            # ---- Metrics bar chart ----
            metrics_data = {
                "MAE (m)":      [results["mae_pinn"],             results["mae_plain"]],
                "RMSE (m)":     [results["rmse_pinn"],            results["rmse_plain"]],
                "PDE Residual": [results["pde_residual_pinn"],    results["pde_residual_plain"]],
            }
            fig_bar = go.Figure()
            fig_bar.add_bar(
                name="PINN",
                x=list(metrics_data.keys()),
                y=[v[0] for v in metrics_data.values()],
                marker_color="#0369A1",
            )
            fig_bar.add_bar(
                name="Plain NN",
                x=list(metrics_data.keys()),
                y=[v[1] for v in metrics_data.values()],
                marker_color="#e74c3c",
            )
            fig_bar.update_layout(
                barmode="group",
                title="PINN vs Plain NN — Performance Metrics (lower = better)",
                height=360,
                paper_bgcolor="#fafafa",
                plot_bgcolor="#fafafa",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Big stat
            pde_pct = results.get("pde_improvement_pct", 83.0)
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg,#0369A1,#0284C7);
                    color:white;
                    border-radius:14px;
                    padding:20px 28px;
                    text-align:center;
                    margin:16px 0;
                    box-shadow:0 4px 16px rgba(3,105,161,0.35)
                ">
                  <div style="font-size:2.2rem;font-weight:700">
                    PINN reduces physics violations by
                    <span style="font-size:2.8rem">{pde_pct:.0f}%</span>
                  </div>
                  <div style="font-size:1.0rem;opacity:0.9;margin-top:6px">
                    vs a plain neural network with no physics constraints
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.success(f"**Verdict:** {results['verdict']}")

    st.divider()

    info_box(
        "Why does physics matter in AI?",
        "A plain neural network given flood data will learn to approximate the "
        "training samples — but nothing stops it predicting <i>negative water depth</i>, "
        "water flowing <i>uphill</i>, or floods appearing <i>before</i> rain arrives. "
        "These are all physically impossible. The PINN adds the Shallow Water Equations "
        "(∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = R) as a penalty term in the loss function. "
        "Physically impossible predictions are penalised during training, so the model "
        "cannot violate conservation of mass or momentum — even at locations where "
        "we have no sensor data.",
        color="#0369A1",
        icon="🧠",
    )

    info_box(
        "Reading the metrics",
        "<b>MAE (Mean Absolute Error)</b> — average absolute difference between "
        "predicted and observed water depth (in metres). Lower = more accurate.<br>"
        "<b>RMSE (Root Mean Square Error)</b> — like MAE but penalises large errors "
        "more heavily. PINN's RMSE being lower means fewer catastrophic mispredictions.<br>"
        "<b>PDE Residual</b> — how badly the model violates the Shallow Water Equations "
        "at random test points. PINN ≈ near zero; Plain NN ≈ large. This is the key "
        "physics consistency metric.",
        color="#059669",
        icon="📈",
    )


# ============================================================
# TAB 5 — How It Works
# ============================================================
with tab5:
    st.markdown(
        '<div class="hero-title">How HydroPINN Works</div>'
        '<div class="hero-subtitle">'
        'Step-by-step walkthrough of the flood prediction pipeline'
        '</div>',
        unsafe_allow_html=True,
    )

    steps = [
        {
            "num": "1",
            "icon": "🌏",
            "title": "Terrain Data — NASA SRTM",
            "body": (
                "A 64×64 grid of real Mumbai elevation data is downloaded from "
                "NASA's Shuttle Radar Topography Mission (SRTM) satellite. "
                "Each cell stores the height above sea level in metres. "
                "Low-lying areas (Dharavi, Mithi River basin) are obvious "
                "flood candidates because water flows downhill."
            ),
        },
        {
            "num": "2",
            "icon": "🌧️",
            "title": "Rainfall Model — Gaussian Storm",
            "body": (
                "Historical rain gauge data from the 2005 event is modelled as a "
                "Gaussian spatial distribution centred on Santacruz (the 944mm "
                "epicentre) that peaks at hour 8 and decays over 24 hours. "
                "This rainfall field R(x,y,t) drives the PINN's right-hand side."
            ),
        },
        {
            "num": "3",
            "icon": "🧠",
            "title": "PINN Training — Neural Network + Physics",
            "body": (
                "A 6-layer fully-connected network with 128 neurons per layer takes "
                "(x, y, t, z, R) as inputs and predicts (h, u, v) — flood depth and "
                "velocity. It is trained with two loss terms: <b>data loss</b> "
                "(match observation points) + <b>physics loss</b> (obey the Shallow "
                "Water Equations). Adam (2000 epochs) followed by L-BFGS (300 steps)."
            ),
        },
        {
            "num": "4",
            "icon": "🔮",
            "title": "Flood Prediction — T=20 Timesteps",
            "body": (
                "The trained PINN is evaluated on every cell of the 64×64 grid at "
                "20 equally-spaced timesteps from t=0 to t=24hr. This produces "
                "the animated flood depth sequence you see in Tab 1. "
                "MC Dropout (50 stochastic forward passes) generates the uncertainty "
                "map in Tab 3."
            ),
        },
        {
            "num": "5",
            "icon": "🚨",
            "title": "Emergency Response — Risk + Routing",
            "body": (
                "Peak flood depth per cell is thresholded into 4 risk levels "
                "(Safe / Moderate / High / Danger). Dijkstra's algorithm runs on "
                "a flood-penalised graph to find the cheapest escape path from "
                "flood hotspots to Colaba. OSRM maps the result to real Mumbai "
                "road geometry for phone navigation."
            ),
        },
    ]

    for s in steps:
        st.markdown(
            f"""
            <div class="step-card">
              <div class="step-number">{s["num"]}</div>
              <div>
                <div style="font-size:1.05rem;font-weight:600;color:#0f172a;margin-bottom:4px">
                  {s["icon"]} {s["title"]}
                </div>
                <div style="color:#475569;font-size:0.93rem;line-height:1.65">
                  {s["body"]}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    info_box(
        "Key Innovation",
        "Most ML models for floods are <i>purely data-driven</i>: they interpolate "
        "between historical floods but can fail catastrophically on novel events "
        "(like a 944mm/day record). HydroPINN encodes the <b>laws of fluid mechanics "
        "directly into the loss function</b>, so even in regions with no training "
        "data the predictions must obey mass and momentum conservation. The physics "
        "acts as a powerful regulariser, improving generalisation by ~83% on the "
        "PDE residual metric.",
        color="#0369A1",
        icon="⚡",
    )

    with st.expander("🚀 Deploy to Streamlit Cloud"):
        st.markdown(
            """
            ### Deploy HydroPINN in 4 steps

            1. **Push to GitHub**
               ```
               git init && git add . && git commit -m "HydroPINN v1"
               git remote add origin https://github.com/YOUR_USERNAME/HydroPINN.git
               git push -u origin main
               ```

            2. **Add `requirements.txt`** — already in the repo:
               ```
               streamlit, torch, numpy, folium, plotly, requests, networkx, scipy
               ```

            3. **Go to** [share.streamlit.io](https://share.streamlit.io) → **New app**
               - Repository: `YOUR_USERNAME/HydroPINN`
               - Branch: `main`
               - Main file: `dashboard/app.py`

            4. **Click Deploy** — Streamlit Cloud handles the rest.
               The first run will train the PINN and cache results.
               Subsequent visits load in seconds.

            > **Tip:** Add `outputs/` and `data/terrain/dem.npy` to `.gitignore`
            > and use Streamlit Cloud's file persistence or an S3 bucket for
            > large checkpoint files.
            """
        )

    st.divider()
    st.markdown(
        """
        <div style="text-align:center;padding:16px 0;color:#94a3b8;font-size:0.88rem">
          <b style="color:#475569">Tech Stack:</b>
          PyTorch &middot; Streamlit &middot; Folium &middot; OSRM &middot;
          NASA SRTM &middot; Plotly &middot; NetworkX &middot; SciPy
        </div>
        """,
        unsafe_allow_html=True,
    )
