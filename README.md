# HydroPINN — Mumbai 2005 Flood Prediction AI

A **Physics-Informed Neural Network (PINN)** that predicts flood depth, risk zones, and safe evacuation routes for the 26 July 2005 Mumbai flood (944mm/24hr, 400+ deaths, $1B damage).

## Live Demo
🌊 **[View Live App](https://your-app.netlify.app)** ← update after deploy

## Results
| Metric | PINN | Plain NN | Winner |
|--------|------|----------|--------|
| Test R² | 0.9895 | 0.9903 | **Tied** |
| MAE | 0.037m | 0.031m | Close |
| Mass Conservation (PDE) | **0.0066** | 2.2042 | **PINN +99.7%** |
| Peak flood depth | **2.60m** | lower | **PINN (realistic)** |

## What it does
- **Flood simulation** — 3D animated predictions for 20 timesteps (0–24hr)
- **Risk zones** — Safe / Moderate / High / Danger classification across Mumbai
- **Safe route finder** — Real road routes (OSRM) colour-coded by flood depth
- **PINN vs Plain NN** — 6-tab scientific comparison showing physics advantage
- **Model deep dive** — Architecture, flow fields, PDE residuals, sensitivity analysis

## Architecture
```
Input: [x, y, t, z(terrain), R(rainfall)]
       ↓
6 × 128 neurons — tanh activation — 10% Dropout
       ↓
Output: h (softplus, always ≥ 0), u (velocity x), v (velocity y)

Loss = 5.0 × L_data + 0.05 × L_pde(scaled continuity) + 0.005 × L_bc
Training: 2000 epoch data-only warmup → 2000 epoch combined → 500 step L-BFGS
```

## Run locally

### Prerequisites
```bash
pip install torch numpy scipy matplotlib plotly streamlit networkx scikit-learn tqdm folium flask flask-cors
```

### Train & run
```bash
# 1. Generate terrain & rainfall data
python data/terrain/generate_dem.py
python data/rainfall/generate_rainfall.py

# 2. Train PINN (~15 min on CPU)
python -c "from model.train import train; train(adam_epochs=4000, lbfgs_steps=500, lambda_data=5.0, lambda_pde=0.05, lambda_bc=0.005)"

# 3. Run inference pipeline
python inference/predict.py
python inference/risk_zones.py
python inference/uncertainty.py
python routing/evacuate.py
python comparison/plain_nn.py
python comparison/compare.py

# 4. Export to web & serve
python webapp/export_data.py
python webapp/analyze_model.py
python webapp/compare_detailed.py
cd webapp/static && python -m http.server 5000
# Open http://localhost:5000
```

## Project structure
```
HydroPINN/
├── model/          # PINN architecture, physics loss, training
├── data/           # Terrain (NASA SRTM) + rainfall generation
├── inference/      # Prediction, risk zones, uncertainty (MC Dropout)
├── routing/        # Dijkstra evacuation routing
├── comparison/     # PINN vs Plain NN benchmarks
├── dashboard/      # Streamlit app (alternative UI)
└── webapp/
    ├── static/     # ← Deploy this folder (index.html + data JS)
    ├── server.py   # Flask backend (optional)
    ├── export_data.py
    ├── analyze_model.py
    └── compare_detailed.py
```

## Physics
The PINN satisfies the **Shallow Water Equations (SWE)**:
- Continuity: `∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = R`
- x-momentum: `∂(hu)/∂t + ∂(hu² + ½gh²)/∂x = −g·h·∂z/∂x`
- y-momentum: `∂(hv)/∂t + ∂(hv² + ½gh²)/∂y = −g·h·∂z/∂y`

Derivatives are computed via PyTorch autograd through the network, making the SWE residual differentiable and directly minimisable as a loss term.

## Tech stack
PyTorch · NumPy · Folium · OSRM · Leaflet.js · Plotly.js · NASA SRTM · OpenStreetMap
