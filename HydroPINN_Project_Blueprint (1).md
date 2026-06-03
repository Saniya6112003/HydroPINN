# HydroPINN — Complete Project Blueprint
> Paste this entire file as context when starting any Claude session in VS Code.

---

## What This Project Is

**HydroPINN** is a Physics-Informed Neural Network (PINN) system that predicts urban flood propagation, maps flood risk zones, and recommends safe evacuation routes — using rainfall data, real Mumbai SRTM terrain elevation, and the Shallow Water Equations (SWE) as physics constraints.

This is a **final, production-quality prototype** — not a demo. Every component is real, runnable, and resume-ready.

**The real-world event being modeled:** The Mumbai floods of 26 July 2005 — 944mm of rainfall in 24 hours, 400+ deaths, $1 billion in damage, 12 million people affected. One of the most documented urban flood disasters in history.

### The 4 Wow Factors (non-negotiable, all must be in final build)

| # | Wow Factor | Why It Matters |
|---|---|---|
| 1 | **Animated flood propagation** | Frame-by-frame Plotly animation with time slider — this is the demo centrepiece |
| 2 | **PINN vs Plain NN comparison** | Side-by-side metrics + physical consistency plot — proves you understand why PINNs exist |
| 3 | **Real Mumbai SRTM DEM terrain** | Uses NASA SRTM 30m elevation data for the 2005 Mumbai flood area — grounds the project in real impact |
| 4 | **Evacuation route overlay** | Green Dijkstra path drawn over red flood zone — routes toward southern Mumbai (Colaba) which stayed dry during 2005 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| PINN Framework | PyTorch (custom PINN, no DeepXDE) |
| Terrain Data | NASA SRTM 30m DEM for Mumbai via `elevation` pip package (fallback: synthetic numpy DEM) |
| Rainfall Data | NASA GPM IMERG (July 25–27, 2005) or synthetic calibrated to 944mm/24hr peak |
| Visualization | Streamlit + Plotly |
| Routing | NetworkX (Dijkstra on flood-penalized grid graph) |
| Uncertainty | MC Dropout (50 forward passes → confidence interval on flood depth) |
| Environment | `venv` + `requirements.txt` |

---

## Project Folder Structure

```
HydroPINN/
│
├── data/
│   ├── terrain/
│   │   ├── generate_dem.py          # Synthetic DEM fallback (numpy gaussian hills)
│   │   ├── fetch_real_dem.py        # Downloads SRTM Mumbai DEM via elevation package
│   │   └── dem.npy                  # Final terrain grid (real or synthetic, 64×64)
│   └── rainfall/
│       └── generate_rainfall.py     # Synthetic rainfall R(x,y,t) calibrated to 944mm/24hr peak
│
├── model/
│   ├── pinn.py                      # PINN architecture (PyTorch MLP + MC Dropout)
│   ├── physics_loss.py              # SWE residuals via torch.autograd.grad
│   ├── data_loss.py                 # MSE on sparse observed flood depth points
│   └── train.py                     # Full training loop (Adam → L-BFGS), saves checkpoint
│
├── inference/
│   ├── predict.py                   # Load model, generate flood depth grid per timestep
│   ├── risk_zones.py                # Threshold h → 4-level risk map
│   └── uncertainty.py              # MC Dropout: 50 passes → mean + std dev flood depth
│
├── routing/
│   └── evacuate.py                  # Grid graph, edge weight = distance + α·flood_depth, Dijkstra
│
├── comparison/
│   ├── plain_nn.py                  # Same MLP architecture, physics loss disabled (λ_pde=0)
│   └── compare.py                   # Metrics: MAE, RMSE, % negative depth, generalization gap
│
├── dashboard/
│   └── app.py                       # Streamlit app — all 5 sections listed below
│
├── outputs/
│   ├── flood_frames/                # .npy arrays per timestep (h, u, v)
│   ├── risk_map.npy                 # Peak-time risk zone grid
│   ├── uncertainty_map.npy          # Std dev of h across MC Dropout passes
│   └── evacuation_path.json         # [{x, y, risk_level}, ...] waypoints
│
├── requirements.txt
└── README.md
```

---

## Real-World Data Sources

### 1. Terrain / Elevation — NASA SRTM 30m DEM

**Source:** NASA / USGS Shuttle Radar Topography Mission (SRTM), Version 3 (void-filled)
**Resolution:** 1 arc-second (~30m per pixel)
**Cost:** Free, no account needed via `elevation` package

**Mumbai bounding box:** `(72.75, 18.85, 73.05, 19.15)` — covers Mumbai peninsula + Dharavi + Bandra + Santacruz

```python
# fetch_real_dem.py — key logic
import elevation
import numpy as np
from scipy.ndimage import zoom
import rasterio

# Mumbai bounding box (west, south, east, north)
BOUNDS = (72.75, 18.85, 73.05, 19.15)
elevation.clip(bounds=BOUNDS, output='data/terrain/mumbai_dem.tif')

# Load, resample to 64×64, save
with rasterio.open('data/terrain/mumbai_dem.tif') as src:
    dem = src.read(1).astype(float)
dem_64 = zoom(dem, (64 / dem.shape[0], 64 / dem.shape[1]))
np.save('data/terrain/dem.npy', dem_64)
```

**Alternative download methods:**
- OpenTopography REST API: https://opentopography.org/developers (free account)
- USGS EarthExplorer: https://earthexplorer.usgs.gov → SRTM 1-Arc-Second Global (free account)

### 2. Rainfall — NASA GPM IMERG (Primary) or Synthetic (Fallback)

**Source:** NASA Integrated Multi-satellitE Retrievals for GPM (IMERG)
**Resolution:** 0.1° × 0.1° (~10km), 30-minute intervals
**Coverage:** Covers 2000–present (includes July 2005 event via TRMM-era data)
**Cost:** Free, requires NASA Earthdata account at https://urs.earthdata.nasa.gov
**Download URL:** https://gpm.nasa.gov/data/imerg
**Event dates to download:** July 25–27, 2005, same bounding box as DEM

**Synthetic fallback (default — no account needed):**
`generate_rainfall.py` generates R(x, y, t) calibrated to the documented 944mm/24hr peak at Santacruz station. Gaussian spatial distribution centred on central Mumbai, tapering toward Colaba (which received only 84mm — a real spatial contrast built into the synthetic field).

**In interviews:** *"Rainfall is synthetic but physically calibrated to the documented 944mm/24hr peak recorded at Santacruz on 26 July 2005, with the correct spatial gradient toward Colaba."*

### 3. Flood Extent Validation

A digitized flood extent map for the 2005 event (based on Gupta 2007) exists in published research showing flooded city wards. Available via ResearchGate. Use this as visual validation — show your PINN's predicted flood zones match the documented 2005 extent. This is a very strong result slide.

### 4. Known Flood Hotspots (baked into routing context)

Real documented flood zones from 26 July 2005:
- **Khar, Gandhi Market, Hindmata** — major flood hotspots, low-lying areas in central Mumbai
- **Santacruz (central)** — 900mm in 25 hours, worst affected
- **Colaba (southern tip)** — only 84mm, stayed largely dry

The evacuation routing exploits this real geography: safe exits route from central Mumbai southward toward Colaba.

---

## Core Architecture

### 1. Physics — Shallow Water Equations (SWE)

The governing PDE enforced during training:

```
Continuity:   ∂h/∂t  +  ∂(hu)/∂x  +  ∂(hv)/∂y  =  R(x, y, t)

x-momentum:   ∂(hu)/∂t  +  ∂(hu² + ½gh²)/∂x  =  -gh · ∂z/∂x

y-momentum:   ∂(hv)/∂t  +  ∂(hv² + ½gh²)/∂y  =  -gh · ∂z/∂y
```

Variables:
- `h(x, y, t)` — water depth (primary output, must be ≥ 0)
- `u(x, y, t)` — x-direction flow velocity
- `v(x, y, t)` — y-direction flow velocity
- `R(x, y, t)` — rainfall source term (input)
- `z(x, y)` — terrain elevation from Mumbai SRTM DEM (input)
- `g = 9.81` m/s²

### 2. PINN Model (`model/pinn.py`)

**Input:** `[x, y, t, z, R]` — 5 features
**Output:** `[h, u, v]` — water depth + velocities
**Architecture:** Fully connected MLP, 6 hidden layers × 128 neurons, tanh activation
**Dropout:** p=0.1 on every hidden layer (enables MC Dropout at inference)
**Output constraint:** h = softplus(raw_h) — guarantees h ≥ 0 always

**Total loss:**
```
L_total = λ_data · L_data  +  λ_pde · L_pde  +  λ_bc · L_bc

λ_data = 1.0   (weight on observed data points)
λ_pde  = 0.1   (weight on physics residuals)
λ_bc   = 0.5   (weight on boundary conditions)
```

### 3. Training Strategy

- **Collocation points:** 10,000 randomly sampled (x, y, t) — physics loss evaluated here
- **Observation points:** 200 sparse points — simulates real IoT rain gauge / river sensor data
- **Optimizer:** Adam (lr=1e-3, 5,000 epochs) → L-BFGS fine-tuning (1,000 steps)
- **Checkpoint:** saved as `outputs/pinn_checkpoint.pt` after training

### 4. WOW FACTOR 1 — Animated Flood Propagation (`dashboard/app.py` Section 2)

- Run `predict.py` for T=20 timesteps → 20 flood depth grids h(x,y,tᵢ)
- Plotly `animation_frame` heatmap — user scrubs through time slider
- Color scale: white (dry, h=0) → blue → deep red (h≥1.5m)
- Mumbai terrain contour lines overlaid in thin gray
- **This is the first thing shown in every demo**

### 5. WOW FACTOR 2 — PINN vs Plain NN Comparison (`comparison/`)

- `plain_nn.py`: identical MLP, training loop identical, but `λ_pde = 0` (physics loss off)
- `compare.py` produces:
  - Side-by-side heatmap: PINN flood depth vs Plain NN flood depth at peak time
  - Bar chart with 3 metrics:
    - MAE on test points
    - RMSE on test points
    - **Physical consistency score** = % of predictions where h < 0 (impossible physically — PINN ≈ 0%, Plain NN ≈ 8-15%)
  - One-line verdict: "PINN reduces physically impossible predictions by X%"
- **This is the slide that wins interviews** — it proves you know *why* you used PINNs

### 6. WOW FACTOR 3 — Real Mumbai SRTM DEM (`data/terrain/fetch_real_dem.py`)

- Uses `elevation` Python package to fetch NASA SRTM 30m DEM
- Bounding box: `(72.75, 18.85, 73.05, 19.15)` — Mumbai peninsula to Bandra/Santacruz
- Captures the real low-lying central Mumbai terrain that caused catastrophic 2005 flooding
- Resampled to 64×64 grid using `scipy.ndimage.zoom`
- Fallback: if download fails, `generate_dem.py` creates synthetic terrain
- **In interviews:** *"I used the actual NASA SRTM terrain for Mumbai — the same low-lying central geography that caused 944mm of rain to pool catastrophically in 2005"*

### 7. Flood Risk Zones (`inference/risk_zones.py`)

| Water Depth | Zone | Colour | Meaning |
|---|---|---|---|
| h < 0.3m | Safe | Green | Walkable |
| 0.3 ≤ h < 0.8m | Moderate | Yellow | Caution |
| 0.8 ≤ h < 1.5m | High | Orange | Evacuate |
| h ≥ 1.5m | Danger | Red | Impassable |

### 8. WOW FACTOR 4 — Evacuation Routing (`routing/evacuate.py`)

- Build a 64×64 grid graph (nodes = terrain cells, edges = 4-directional adjacency)
- Edge weight formula: `w = 1.0 + 10.0 · h(node)` — flooded cells become expensive to traverse
- Danger-zone nodes (h ≥ 1.5m): weight = 1,000,000 (effectively impassable)
- Run Dijkstra from user-specified start cell (e.g. Santacruz / Dharavi) toward southern exits (Colaba direction)
- Output: JSON list of waypoints `[{x, y, risk_level}]`
- **Dashboard overlay:** green path drawn on top of the red/orange risk map
- Shows: total path length, number of moderate-risk cells crossed, estimated safety score
- **Real geography context:** Santacruz (start, flooded) → Colaba (exit, dry) mirrors what actually happened in 2005
- **This is what makes it product-grade** — prediction alone is science; prediction + routing is a tool

### 9. Uncertainty Quantification (`inference/uncertainty.py`)

- At inference, keep dropout layers active (`model.train()` mode)
- Run 50 forward passes on same input
- Compute mean(h) and std(h) across passes
- Dashboard shows confidence interval: "flood depth = 1.2m ± 0.3m"
- **Interview line:** *"Almost no student adds uncertainty quantification — I added MC Dropout so emergency responders know not just the predicted depth but how confident the model is"*

---

## Streamlit Dashboard (`dashboard/app.py`) — 5 Sections

**Section 1 — Simulation Setup (sidebar)**
- Sliders: rainfall intensity (mm/hr), storm duration (hrs)
- Toggle: use real Mumbai SRTM DEM / synthetic terrain
- Button: "Run Simulation" → triggers training + inference pipeline

**Section 2 — Animated Flood Map** *(WOW FACTOR 1)*
- Plotly animated heatmap, time slider, Mumbai terrain contours
- Color: white → blue → red with depth
- Title: "Mumbai 26 July 2005 — Flood Propagation"

**Section 3 — Risk Zone Map + Uncertainty**
- Static peak-time risk heatmap (4-colour)
- Uncertainty overlay toggle: shows std dev as opacity mask
- Legend with depth thresholds

**Section 4 — Evacuation Routes** *(WOW FACTOR 4)*
- Number inputs: start X, start Y coordinates
- Preset buttons: "Start from Santacruz", "Start from Dharavi", "Start from Khar"
- Button: "Find Safe Route" → runs Dijkstra
- Green path drawn on top of risk map toward safe southern exit
- Stats: path length, risk zones crossed, safety score

**Section 5 — PINN vs Plain NN** *(WOW FACTOR 2)*
- Side-by-side flood prediction heatmaps
- Bar chart: MAE, RMSE, physical consistency
- Plain English verdict text

---

## Key Interview Q&A (memorise these)

**"Why PINNs instead of CNN/LSTM?"**
→ Flood data is sparse and expensive. Plain NNs can predict negative water depth — physically impossible. SWE physics loss enforces mass and momentum conservation at every collocation point. The model cannot violate physics even where there's no training data.

**"What PDE did you use?"**
→ The depth-averaged Shallow Water Equations — the standard in computational hydrodynamics for free-surface flow on complex terrain.

**"How is physics loss computed?"**
→ At each collocation point, I compute ∂h/∂t, ∂(hu)/∂x, ∂(hv)/∂y using `torch.autograd.grad`. The SWE continuity residual is the physics loss term. Both data loss and physics residual are minimised simultaneously during backprop.

**"What real-world data did you use?"**
→ NASA SRTM 30m elevation data for Mumbai, clipped to the peninsula bounding box covering Santacruz and Dharavi — the areas worst affected in 2005. Rainfall is synthetic but calibrated to the documented 944mm/24hr peak at Santacruz station.

**"Why Mumbai?"**
→ The 2005 Mumbai floods are one of the most documented urban flood disasters in history — 944mm in 24 hours, 400+ deaths, $1 billion damage. There's a published digitized flood extent map I used for visual validation. It's also a textbook case of how flat coastal terrain + overwhelmed drainage = catastrophic urban flooding.

**"How did you validate?"**
→ Held-out test set never seen during training. Physical consistency check: % of predictions with h < 0 (PINN ≈ 0%, plain NN ≈ 8-15%). Visual comparison against the published 2005 flood extent map.

**"What's the MC Dropout for?"**
→ Uncertainty quantification. Running 50 stochastic forward passes gives a distribution over flood depth predictions. Emergency services need to know not just the prediction but how confident the model is — a 1.2m ± 0.1m prediction is very different from 1.2m ± 0.8m.

**"How would this scale to a real city?"**
→ The SRTM DEM is already real. For live deployment: replace synthetic rainfall with real-time IMD / GPM IMERG feeds, replace sparse observations with IoT river gauge data, swap Dijkstra on a grid with A* on the actual Mumbai road network via OSMnx. The PINN architecture itself is unchanged.

**"What are PINN limitations?"**
→ Training is slow vs pure data-driven models. L-BFGS is memory-intensive. SWE is simplified — ignores turbulence, 3D flow, and drainage infrastructure. Real Mumbai flooding is worsened by its 150-year-old drainage system, which this model doesn't capture.

---

## Build Order (2 Days)

### Day 1 — Core Pipeline
1. `data/terrain/fetch_real_dem.py` — Mumbai SRTM DEM download + `generate_dem.py` fallback
2. `data/rainfall/generate_rainfall.py` — synthetic R(x,y,t) calibrated to 944mm/24hr
3. `model/pinn.py` — architecture with MC Dropout + softplus output
4. `model/physics_loss.py` — SWE residuals via autograd
5. `model/train.py` — Adam + L-BFGS loop, checkpoint saving
6. `inference/predict.py` — T=20 timestep flood frames
7. `inference/risk_zones.py` — depth thresholds → risk grid

### Day 2 — Wow Factors + Dashboard
8. `inference/uncertainty.py` — MC Dropout 50-pass inference
9. `routing/evacuate.py` — grid graph + Dijkstra (Santacruz → Colaba routing)
10. `comparison/plain_nn.py` + `comparison/compare.py`
11. `dashboard/app.py` — all 5 sections
12. `README.md` — architecture diagram, setup, screenshots
13. GitHub push — clean commits, demo GIF, badge

---

## Requirements (`requirements.txt`)

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
plotly>=5.14.0
streamlit>=1.25.0
networkx>=3.1
scikit-learn>=1.3.0
tqdm>=4.65.0
elevation>=1.0.6
rasterio>=1.3.0
```

---

## How to Use This File in VS Code

At the start of every Claude session in VS Code, paste:

```
Here is my full HydroPINN project blueprint. I am modeling the 2005 Mumbai floods.
Build exactly what is described — do not change architecture, scope, or wow factors.
Help me build [specific file] now.

[paste this entire document]
```

Then tell Claude exactly which file to build next, in the Day 1 / Day 2 order above.

---

*Project: HydroPINN | Event: Mumbai 2005 Floods (944mm/24hr) | Stack: PyTorch + Streamlit + NetworkX | Data: NASA SRTM DEM + GPM IMERG | Wow: Animation + Comparison + Real DEM + Evacuation | Goal: Resume / Interviews*
