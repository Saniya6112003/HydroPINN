# -*- coding: utf-8 -*-
"""
HydroPINN Model Analysis
Run:  python webapp/analyze_model.py
Writes webapp/static/analysis.js with all deep-dive data.

Computes:
  1. Velocity flow field (u, v) at peak flood time
  2. PDE residual maps (continuity + momentum) via finite differences
  3. h_pred vs h_obs scatter at sensor points
  4. Input sensitivity: dh/dz and dh/dR across grid (autograd)
  5. Layer weight distributions (histogram bins per layer)
  6. Layer activations distribution (mean / std per layer for a batch)
  7. Training summary stats
"""
from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CKPT      = os.path.join(ROOT, "outputs", "pinn_checkpoint.pt")
FRAMES    = os.path.join(ROOT, "outputs", "flood_frames")
DEST      = os.path.join(ROOT, "webapp", "static", "analysis.js")

# ── guard ─────────────────────────────────────────────────────────────────────
if not os.path.exists(CKPT):
    print("[analyze] No checkpoint found — run training first."); sys.exit(1)

import torch
from model.pinn import HydroPINN
from model.train import load_dem_grid, sample_z_from_dem, compute_dem_gradients, sample_dem_grads
from model.data_loss import generate_observation_points
from data.rainfall.generate_rainfall import rainfall_rate

print("[analyze] Loading model…")
device = torch.device("cpu")
ckpt   = torch.load(CKPT, map_location=device, weights_only=False)
model  = HydroPINN()
model.load_state_dict(ckpt["model_state"])
model.eval()

dem = load_dem_grid()
print(f"  DEM: {dem.shape}  min={dem.min():.1f}m  max={dem.max():.1f}m")

all_h = np.load(os.path.join(FRAMES, "all_h.npy")).astype(np.float32)
T, G, _ = all_h.shape
peak_idx = int(np.argmax(all_h.max(axis=(1,2))))
peak_hr  = round(peak_idx * 24 / (T-1), 1)
print(f"  Peak frame: {peak_idx}  ({peak_hr}hr)  max_h={all_h[peak_idx].max():.2f}m")

# ── 1. Velocity field ─────────────────────────────────────────────────────────
print("[analyze] Loading velocity fields…")
STEP = 4   # subsample: keep every 4th cell -> 16x16 arrows
u_all, v_all, h_all = [], [], []
for i in range(T):
    fp = os.path.join(FRAMES, f"frame_{i:02d}.npy")
    if os.path.exists(fp):
        d = np.load(fp, allow_pickle=True).item()
        u_all.append(d["u"].astype(np.float32))
        v_all.append(d["v"].astype(np.float32))
        h_all.append(d["h"].astype(np.float32))

u_peak = u_all[peak_idx]   # 64x64
v_peak = v_all[peak_idx]

# Subsample + normalise for display
rows, cols = np.arange(0, G, STEP), np.arange(0, G, STEP)
lat_grid = np.array([18.905 + (r/(G-1))*(19.285-18.905) for r in rows])
lon_grid = np.array([72.790 + (c/(G-1))*(72.960-72.790) for c in cols])
U_sub = u_peak[::STEP, ::STEP]
V_sub = v_peak[::STEP, ::STEP]
mag   = np.sqrt(U_sub**2 + V_sub**2) + 1e-9
# Normalise arrow length to max 1 for display
scale = 1.0 / (mag.max() + 1e-9)
U_n = (U_sub * scale).round(4).tolist()
V_n = (V_sub * scale).round(4).tolist()
mag_n = (mag / mag.max()).round(3).tolist()

# Build flat list of arrows for Plotly quiver
arrow_lats, arrow_lons, arrow_u, arrow_v, arrow_mag = [], [], [], [], []
SZ = len(rows)
for ri, r in enumerate(rows):
    for ci, c in enumerate(cols):
        lat = 18.905 + (r/(G-1))*(19.285-18.905)
        lon = 72.790 + (c/(G-1))*(72.960-72.790)
        u_val = float(U_n[ri][ci])
        v_val = float(V_n[ri][ci])
        m_val = float(mag_n[ri][ci])
        arrow_lats.append(round(lat,5))
        arrow_lons.append(round(lon,5))
        arrow_u.append(u_val)
        arrow_v.append(v_val)
        arrow_mag.append(m_val)

# Per-timestep max velocity for animation chart
vel_curve = [round(float(np.sqrt(u_all[i]**2+v_all[i]**2).max()),3) for i in range(T)]
depth_curve = [round(float(all_h[i].max()),3) for i in range(T)]
time_axis   = [round(i*24/(T-1),1) for i in range(T)]

# ── 2. PDE residuals (finite differences on grid) ────────────────────────────
print("[analyze] Computing PDE residuals…")

h_pk = all_h[peak_idx].astype(np.float64)
u_pk = u_peak.astype(np.float64)
v_pk = v_peak.astype(np.float64)

dt_norm = 1.0/(T-1)
if 0 < peak_idx < T-1:
    dh_dt = (all_h[peak_idx+1] - all_h[peak_idx-1]).astype(np.float64) / (2*dt_norm)
else:
    dh_dt = np.zeros_like(h_pk)

hu = h_pk * u_pk
hv = h_pk * v_pk
d_hu_dx = np.gradient(hu, axis=1) * G   # d/d(x_norm) * G
d_hv_dy = np.gradient(hv, axis=0) * G

x_lin = np.linspace(0,1,G,dtype=np.float32)
y_lin = np.linspace(0,1,G,dtype=np.float32)
XX,YY = np.meshgrid(x_lin,y_lin)
R_grid = rainfall_rate(XX.ravel(),YY.ravel(),
                       np.full(G*G, peak_idx/(T-1), dtype=np.float32)).reshape(G,G)
R_ms = R_grid / (1000*3600)

res_cont = np.abs(dh_dt + d_hu_dx + d_hv_dy - R_ms).astype(np.float32)
rc_norm  = (res_cont / (res_cont.max()+1e-9)).round(4).tolist()

# momentum residuals (approximate via h-field only)
g = 9.81
dz_dx_g, dz_dy_g = compute_dem_gradients(dem)
dz_dx = dz_dx_g.astype(np.float64)
dz_dy = dz_dy_g.astype(np.float64)

dhu_dt = np.gradient(h_pk*u_pk, axis=0)*G if T>2 else np.zeros_like(h_pk)
press_x = g * h_pk * np.gradient(h_pk,axis=1)*G
terrain_x = g * h_pk * dz_dx
res_mom_x = np.abs(dhu_dt + press_x + terrain_x).astype(np.float32)
rmx_norm  = (res_mom_x/(res_mom_x.max()+1e-9)).round(4).tolist()

dhv_dt = np.gradient(h_pk*v_pk, axis=1)*G if T>2 else np.zeros_like(h_pk)
press_y = g * h_pk * np.gradient(h_pk,axis=0)*G
terrain_y = g * h_pk * dz_dy
res_mom_y = np.abs(dhv_dt + press_y + terrain_y).astype(np.float32)
rmy_norm  = (res_mom_y/(res_mom_y.max()+1e-9)).round(4).tolist()

pde_max = {"cont":round(float(res_cont.max()),4),
           "mom_x":round(float(res_mom_x.max()),4),
           "mom_y":round(float(res_mom_y.max()),4)}
print(f"  Residuals max — cont:{pde_max['cont']:.4f}  mom_x:{pde_max['mom_x']:.4f}  mom_y:{pde_max['mom_y']:.4f}")

# ── 3. h_pred vs h_obs scatter ────────────────────────────────────────────────
print("[analyze] Scatter: h_pred vs h_obs…")
obs = generate_observation_points(n_obs=200, dem=dem)
def _t(a): return torch.tensor(a, dtype=torch.float32)
inp_obs = torch.stack([_t(obs["x"]),_t(obs["y"]),_t(obs["t"]),_t(obs["z"]),_t(obs["R"])],dim=1)
with torch.no_grad():
    h_pred_obs, _, _ = model(inp_obs)
h_pred_list = h_pred_obs.squeeze().numpy().tolist()
h_obs_list  = obs["h_obs"].tolist()
scatter_mse = round(float(np.mean((np.array(h_pred_list)-np.array(h_obs_list))**2)),4)
scatter_mae = round(float(np.mean(np.abs(np.array(h_pred_list)-np.array(h_obs_list)))),4)
print(f"  MSE={scatter_mse:.4f}  MAE={scatter_mae:.4f}")

# ── 4. Input sensitivity dh/dz  dh/dR ────────────────────────────────────────
print("[analyze] Computing input sensitivity (autograd)…")
SG = 32   # 32x32 sensitivity grid
xs = np.linspace(0,1,SG,dtype=np.float32)
ys = np.linspace(0,1,SG,dtype=np.float32)
XXs,YYs = np.meshgrid(xs,ys,indexing='ij')
t_peak = np.float32(peak_idx/(T-1))
z_np = sample_z_from_dem(dem, XXs.ravel(), YYs.ravel()).astype(np.float32)
R_np = rainfall_rate(XXs.ravel(),YYs.ravel(),np.full(SG*SG,t_peak)).astype(np.float32)

x_s = torch.tensor(XXs.ravel(), requires_grad=False)
y_s = torch.tensor(YYs.ravel(), requires_grad=False)
t_s = torch.full((SG*SG,), float(t_peak))
z_s = torch.tensor(z_np, requires_grad=True)
R_s = torch.tensor(R_np, requires_grad=True)

inp_s = torch.stack([x_s,y_s,t_s,z_s,R_s],dim=1)
h_s, _, _ = model(inp_s)

dh_dz = torch.autograd.grad(h_s.sum(), z_s, retain_graph=True)[0].detach().numpy().reshape(SG,SG)
dh_dR = torch.autograd.grad(h_s.sum(), R_s)[0].detach().numpy().reshape(SG,SG)

# normalise to [-1,1] for display
def norm_sens(a):
    mx = max(abs(a.max()),abs(a.min()),1e-9)
    return (a/mx).round(4).tolist()

dh_dz_n = norm_sens(dh_dz)
dh_dR_n = norm_sens(dh_dR)
sens_stats = {
    "dh_dz_max": round(float(abs(dh_dz).max()),4),
    "dh_dR_max": round(float(abs(dh_dR).max()),4),
}
print(f"  |dh/dz| max={sens_stats['dh_dz_max']:.4f}  |dh/dR| max={sens_stats['dh_dR_max']:.4f}")

# also compute sensitivity to x and y (spatial gradients of prediction)
x_s2 = torch.tensor(XXs.ravel(), requires_grad=True)
y_s2 = torch.tensor(YYs.ravel(), requires_grad=True)
inp_s2 = torch.stack([x_s2,y_s2,t_s,z_s.detach(),R_s.detach()],dim=1)
h_s2, _, _ = model(inp_s2)
dh_dx = torch.autograd.grad(h_s2.sum(), x_s2, retain_graph=True)[0].detach().numpy().reshape(SG,SG)
dh_dy = torch.autograd.grad(h_s2.sum(), y_s2)[0].detach().numpy().reshape(SG,SG)
dh_dx_n = norm_sens(dh_dx)
dh_dy_n = norm_sens(dh_dy)

# ── 5. Layer weight distributions ────────────────────────────────────────────
print("[analyze] Layer weight distributions…")
layer_weights = []
layer_names   = []
for name, param in model.named_parameters():
    if "weight" in name:
        w = param.detach().numpy().ravel()
        # histogram: 30 bins
        counts, edges = np.histogram(w, bins=30)
        layer_weights.append({
            "name":   name,
            "counts": counts.tolist(),
            "edges":  [round(float(e),4) for e in edges[:-1]],
            "mean":   round(float(w.mean()),4),
            "std":    round(float(w.std()),4),
            "shape":  list(param.shape),
        })
print(f"  {len(layer_weights)} weight tensors")

# ── 6. Layer activations (forward pass on a batch) ───────────────────────────
print("[analyze] Layer activations…")
activations = {}
hooks = []
def make_hook(nm):
    def hook(m, inp, out):
        activations[nm] = out.detach().numpy()
    return hook

layer_idx = 0
for m in model.hidden:
    if isinstance(m, torch.nn.Tanh):
        hooks.append(m.register_forward_hook(make_hook(f"hidden_tanh_{layer_idx}")))
        layer_idx += 1

# batch of 512 random points
rng = np.random.default_rng(42)
bx = rng.uniform(0,1,512).astype(np.float32)
by = rng.uniform(0,1,512).astype(np.float32)
bt = rng.uniform(0,1,512).astype(np.float32)
bz = sample_z_from_dem(dem,bx,by).astype(np.float32)
bR = rainfall_rate(bx,by,bt).astype(np.float32)
b_inp = torch.stack([torch.tensor(bx),torch.tensor(by),torch.tensor(bt),
                     torch.tensor(bz),torch.tensor(bR)],dim=1)
with torch.no_grad():
    _ = model(b_inp)
for h_ in hooks: h_.remove()

act_stats = []
for nm, act in activations.items():
    act_flat = act.ravel()
    counts, edges = np.histogram(act_flat, bins=40, range=(-1,1))
    act_stats.append({
        "layer": nm,
        "mean":  round(float(act_flat.mean()),4),
        "std":   round(float(act_flat.std()),4),
        "saturation_pct": round(float((np.abs(act_flat)>0.95).mean()*100),1),
        "counts": counts.tolist(),
        "edges":  [round(float(e),3) for e in edges[:-1]],
    })
print(f"  {len(act_stats)} activation layers captured")

# ── 7. Model architecture summary ────────────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

arch = {
    "total_params": total_params,
    "trainable":    trainable,
    "layers": [
        {"name":"Input","neurons":5,"activation":"—"},
        {"name":"Hidden 1","neurons":128,"activation":"tanh"},
        {"name":"Hidden 2","neurons":128,"activation":"tanh"},
        {"name":"Hidden 3","neurons":128,"activation":"tanh"},
        {"name":"Hidden 4","neurons":128,"activation":"tanh"},
        {"name":"Hidden 5","neurons":128,"activation":"tanh"},
        {"name":"Hidden 6","neurons":128,"activation":"tanh"},
        {"name":"Output h","neurons":1,"activation":"softplus"},
        {"name":"Output u","neurons":1,"activation":"linear"},
        {"name":"Output v","neurons":1,"activation":"linear"},
    ]
}

# ── Assemble & write ──────────────────────────────────────────────────────────
analysis = {
    "meta": {
        "peak_idx": peak_idx,
        "peak_hr":  peak_hr,
        "total_params": total_params,
        "trainable": trainable,
        "grid_size": G,
    },
    "velocity": {
        "arrow_lats": arrow_lats,
        "arrow_lons": arrow_lons,
        "arrow_u":    arrow_u,
        "arrow_v":    arrow_v,
        "arrow_mag":  arrow_mag,
        "u_grid":     [[round(float(v),3) for v in row] for row in u_peak[::2,::2].tolist()],
        "v_grid":     [[round(float(v),3) for v in row] for row in v_peak[::2,::2].tolist()],
    },
    "curves": {
        "time":  time_axis,
        "depth": depth_curve,
        "vel":   vel_curve,
    },
    "pde": {
        "cont":     rc_norm,
        "mom_x":    rmx_norm,
        "mom_y":    rmy_norm,
        "max":      pde_max,
    },
    "scatter": {
        "h_pred":    [round(v,4) for v in h_pred_list],
        "h_obs":     [round(v,4) for v in h_obs_list],
        "mse":       scatter_mse,
        "mae":       scatter_mae,
    },
    "sensitivity": {
        "dh_dz":      dh_dz_n,
        "dh_dR":      dh_dR_n,
        "dh_dx":      dh_dx_n,
        "dh_dy":      dh_dy_n,
        "stats":      sens_stats,
        "grid_size":  SG,
    },
    "weights":     layer_weights,
    "activations": act_stats,
    "arch":        arch,
}

os.makedirs(os.path.dirname(DEST), exist_ok=True)
js = "// Auto-generated by analyze_model.py\nconst ANALYSIS = " + json.dumps(analysis, separators=(',',':')) + ";\n"
with open(DEST,"w",encoding="utf-8") as f:
    f.write(js)

size_kb = os.path.getsize(DEST)/1024
print(f"\n[analyze] Written: {DEST}  ({size_kb:.0f} KB)")
print(f"  Arrows: {len(arrow_lats)}")
print(f"  Scatter points: {len(h_pred_list)}")
print(f"  Weight layers: {len(layer_weights)}")
print(f"  Activation layers: {len(act_stats)}")
print(f"  Model params: {total_params:,}")
