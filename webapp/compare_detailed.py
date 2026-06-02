# -*- coding: utf-8 -*-
"""
Detailed PINN vs Plain NN comparison for the web dashboard.
Generates rich comparison data: full timeseries, PDE maps, divergence,
mass conservation, extrapolation test, and radar chart metrics.

Run: python webapp/compare_detailed.py   (or called from export_data.py)
"""
from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from model.pinn import HydroPINN
from model.train import (load_dem_grid, sample_z_from_dem,
                          compute_dem_gradients, sample_dem_grads)
from data.rainfall.generate_rainfall import rainfall_rate

PINN_CKPT  = os.path.join(ROOT, "outputs", "pinn_checkpoint.pt")
PLAIN_CKPT = os.path.join(ROOT, "outputs", "plain_nn_checkpoint.pt")
FRAMES_DIR = os.path.join(ROOT, "outputs", "flood_frames")


def _load(ckpt):
    m = HydroPINN()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def _full_grid(model, dem, t_val, G=64):
    """Run model on G×G grid at normalised time t_val."""
    x = np.linspace(0,1,G,dtype=np.float32)
    y = np.linspace(0,1,G,dtype=np.float32)
    XX,YY = np.meshgrid(x,y)
    xf,yf = XX.ravel(),YY.ravel()
    tf = np.full_like(xf,t_val)
    xi = (xf*(dem.shape[1]-1)).clip(0,dem.shape[1]-1).astype(int)
    yi = (yf*(dem.shape[0]-1)).clip(0,dem.shape[0]-1).astype(int)
    zf = dem[yi,xi]
    Rf = rainfall_rate(xf,yf,tf).astype(np.float32)
    def _t(a): return torch.tensor(a,dtype=torch.float32)
    inp = torch.stack([_t(xf),_t(yf),_t(tf),_t(zf.astype(np.float32)),_t(Rf)],dim=1)
    with torch.no_grad():
        h,u,v = model(inp)
    return h.squeeze().numpy().reshape(G,G), u.squeeze().numpy().reshape(G,G), v.squeeze().numpy().reshape(G,G)


def _pde_fd(h_prev, h_curr, h_next, u_g, v_g, t_val, dem, T, G=64):
    """Finite-difference SWE residuals (continuity) on grid."""
    dt = 1.0/(T-1)
    dh_dt = (h_next - h_prev)/(2*dt) if h_prev is not None else np.zeros_like(h_curr)
    hu = h_curr*u_g;  hv = h_curr*v_g
    d_hu_dx = np.gradient(hu,axis=1)*G
    d_hv_dy = np.gradient(hv,axis=0)*G
    x = np.linspace(0,1,G,dtype=np.float32)
    y = np.linspace(0,1,G,dtype=np.float32)
    XX,YY=np.meshgrid(x,y)
    R_ms = rainfall_rate(XX.ravel(),YY.ravel(),np.full(G*G,t_val,dtype=np.float32)).reshape(G,G)/(1000*3600)
    return np.abs(dh_dt + d_hu_dx + d_hv_dy - R_ms).astype(np.float32)


def run_detailed_comparison():
    if not os.path.exists(PINN_CKPT) or not os.path.exists(PLAIN_CKPT):
        print("[compare_detail] Missing checkpoint(s) — skipping"); return {}

    print("[compare_detail] Loading models…")
    pinn  = _load(PINN_CKPT)
    plain = _load(PLAIN_CKPT)
    dem   = load_dem_grid()
    T     = 20

    times = np.linspace(0,1,T,dtype=np.float32)
    time_hrs = [round(float(t)*24,1) for t in times]

    # ── Full timeseries for both ──────────────────────────────────────────────
    print("[compare_detail] Running full grid inference for both models…")
    pinn_h_all, plain_h_all = [], []
    pinn_u_all, pinn_v_all  = [], []
    plain_u_all,plain_v_all = [], []

    for i,t_val in enumerate(times):
        ph,pu,pv = _full_grid(pinn,  dem, float(t_val))
        nh,nu,nv = _full_grid(plain, dem, float(t_val))
        pinn_h_all.append(ph);  pinn_u_all.append(pu);  pinn_v_all.append(pv)
        plain_h_all.append(nh); plain_u_all.append(nu); plain_v_all.append(nv)
        if (i+1)%5==0: print(f"  {i+1}/{T} done")

    pinn_h  = np.stack(pinn_h_all)   # (T,G,G)
    plain_h = np.stack(plain_h_all)

    peak_pinn  = int(np.argmax(pinn_h.max(axis=(1,2))))
    peak_plain = int(np.argmax(plain_h.max(axis=(1,2))))

    # ── PDE residual maps at peak ─────────────────────────────────────────────
    print("[compare_detail] PDE residual maps…")
    def pde_map(h_all, u_all, v_all, peak_idx):
        h_prev = h_all[peak_idx-1] if peak_idx>0 else h_all[peak_idx]
        h_next = h_all[peak_idx+1] if peak_idx<T-1 else h_all[peak_idx]
        res = _pde_fd(h_prev, h_all[peak_idx], h_next,
                      u_all[peak_idx], v_all[peak_idx],
                      times[peak_idx], dem, T)
        mx = res.max()+1e-9
        return (res/mx).round(3).tolist(), float(res.max())

    pinn_pde_map,  pinn_pde_mx  = pde_map(pinn_h_all,  pinn_u_all,  pinn_v_all,  peak_pinn)
    plain_pde_map, plain_pde_mx = pde_map(plain_h_all, plain_u_all, plain_v_all, peak_plain)

    # ── Divergence maps (|pinn - plain|) ─────────────────────────────────────
    div_maps = []
    for i in range(T):
        diff = np.abs(pinn_h_all[i] - plain_h_all[i])
        div_maps.append([[round(float(v),3) for v in row] for row in diff.tolist()])
    peak_div_idx = int(np.argmax([np.abs(pinn_h_all[i]-plain_h_all[i]).max() for i in range(T)]))

    # ── Mass conservation curve ───────────────────────────────────────────────
    mass_pinn  = [round(float(pinn_h_all[i].sum()),2)  for i in range(T)]
    mass_plain = [round(float(plain_h_all[i].sum()),2) for i in range(T)]

    # smoothness score = negative of std of second derivative
    def mass_smoothness(m):
        d2 = np.diff(np.diff(m))
        return round(float(np.abs(d2).mean()),3)

    smooth_pinn  = mass_smoothness(mass_pinn)
    smooth_plain = mass_smoothness(mass_plain)

    # ── PDE residual per timestep (single scalar) ─────────────────────────────
    pde_curve_pinn, pde_curve_plain = [], []
    for i in range(T):
        hp = h_all = pinn_h_all[i]
        up = pinn_u_all[i]; vp = pinn_v_all[i]
        h_prev = pinn_h_all[i-1] if i>0 else hp
        h_next = pinn_h_all[i+1] if i<T-1 else hp
        r = _pde_fd(h_prev, hp, h_next, up, vp, times[i], dem, T)
        pde_curve_pinn.append(round(float(r.mean()),4))

        hn = plain_h_all[i]
        un = plain_u_all[i]; vn = plain_v_all[i]
        h_prev2 = plain_h_all[i-1] if i>0 else hn
        h_next2 = plain_h_all[i+1] if i<T-1 else hn
        r2 = _pde_fd(h_prev2, hn, h_next2, un, vn, times[i], dem, T)
        pde_curve_plain.append(round(float(r2.mean()),4))

    # ── Extrapolation test — zero rainfall ───────────────────────────────────
    print("[compare_detail] Extrapolation test (no rainfall)…")
    # Patch rainfall to 0 temporarily by overriding with near-zero
    G=64
    x_lin=np.linspace(0,1,G,dtype=np.float32)
    y_lin=np.linspace(0,1,G,dtype=np.float32)
    XX,YY=np.meshgrid(x_lin,y_lin)
    xf,yf=XX.ravel(),YY.ravel()
    tf=np.full_like(xf,0.5)
    xi2=(xf*(dem.shape[1]-1)).clip(0,dem.shape[1]-1).astype(int)
    yi2=(yf*(dem.shape[0]-1)).clip(0,dem.shape[0]-1).astype(int)
    zf2=dem[yi2,xi2].astype(np.float32)
    Rzero=np.zeros(G*G,dtype=np.float32)   # ZERO rainfall

    def _t2(a): return torch.tensor(a,dtype=torch.float32)
    inp_zero=torch.stack([_t2(xf),_t2(yf),_t2(tf),_t2(zf2),_t2(Rzero)],dim=1)
    with torch.no_grad():
        h_pinn_zero, _,_  = pinn(inp_zero)
        h_plain_zero,_,_  = plain(inp_zero)
    extrap_pinn  = [[round(float(v),3) for v in row]
                    for row in h_pinn_zero.squeeze().numpy().reshape(G,G).tolist()]
    extrap_plain = [[round(float(v),3) for v in row]
                    for row in h_plain_zero.squeeze().numpy().reshape(G,G).tolist()]
    extrap_pinn_max  = round(float(h_pinn_zero.max()),3)
    extrap_plain_max = round(float(h_plain_zero.max()),3)
    print(f"  Zero-rain max: PINN={extrap_pinn_max:.3f}m  Plain={extrap_plain_max:.3f}m")

    # ── Radar chart metrics ───────────────────────────────────────────────────
    # All metrics normalised to [0,1] where 1 = best possible
    pde_pinn_raw  = float(np.mean(pde_curve_pinn))
    pde_plain_raw = float(np.mean(pde_curve_plain))
    pde_max_raw   = max(pde_pinn_raw, pde_plain_raw) + 1e-9

    peak_timing_pinn  = abs(peak_pinn  - 7)/7   # closer to frame 7 (8.8hr) = better
    peak_timing_plain = abs(peak_plain - 7)/7
    timing_pinn  = max(0, 1-peak_timing_pinn)
    timing_plain = max(0, 1-peak_timing_plain)

    max_depth_ref = max(pinn_h.max(), plain_h.max()) + 1e-9
    pinn_max_d  = float(pinn_h.max())
    plain_max_d = float(plain_h.max())
    depth_score_pinn  = min(1, pinn_max_d /max_depth_ref)
    depth_score_plain = min(1, plain_max_d/max_depth_ref)

    # Zero-rainfall score: lower is better (model should predict near 0)
    extrap_score_pinn  = max(0, 1 - extrap_pinn_max  / (extrap_plain_max+1e-9))
    extrap_score_plain = max(0, 1 - extrap_plain_max / (extrap_pinn_max +1e-9))

    # Mass smoothness score: lower irregularity = better
    max_smooth = max(smooth_pinn, smooth_plain)+1e-9
    smooth_score_pinn  = 1 - smooth_pinn /max_smooth
    smooth_score_plain = 1 - smooth_plain/max_smooth

    radar = {
        "labels": ["Physics\nCompliance","Mass\nConservation","Peak\nTiming","Extrapolation","Depth\nRealism"],
        "pinn":  [round(1-pde_pinn_raw/pde_max_raw,3), round(smooth_score_pinn,3),
                  round(timing_pinn,3), round(extrap_score_pinn,3), round(depth_score_pinn,3)],
        "plain": [round(1-pde_plain_raw/pde_max_raw,3), round(smooth_score_plain,3),
                  round(timing_plain,3), round(extrap_score_plain,3), round(depth_score_plain,3)],
    }

    # ── Scorecard ─────────────────────────────────────────────────────────────
    pde_improv = round(100*(1-pde_pinn_raw/(pde_plain_raw+1e-9)),1)
    scorecard = {
        "pde_improvement":    pde_improv,
        "pinn_pde_raw":       round(pde_pinn_raw,4),
        "plain_pde_raw":      round(pde_plain_raw,4),
        "pinn_peak_hr":       round(peak_pinn*24/(T-1),1),
        "plain_peak_hr":      round(peak_plain*24/(T-1),1),
        "pinn_max_depth":     round(pinn_max_d,3),
        "plain_max_depth":    round(plain_max_d,3),
        "smooth_pinn":        round(smooth_pinn,3),
        "smooth_plain":       round(smooth_plain,3),
        "extrap_pinn_max":    extrap_pinn_max,
        "extrap_plain_max":   extrap_plain_max,
        "pinn_wins":          sum(1 for p,n in zip(radar["pinn"],radar["plain"]) if p>n),
    }

    # ── Compact grid frames (downsample to 32×32 for JS) ─────────────────────
    def ds(arr): return [[round(float(v),3) for v in row] for row in arr[::2,::2].tolist()]
    pinn_frames_ds  = [ds(pinn_h_all[i])  for i in range(T)]
    plain_frames_ds = [ds(plain_h_all[i]) for i in range(T)]

    result = {
        "scorecard":      scorecard,
        "radar":          radar,
        "time_hrs":       time_hrs,
        "mass_pinn":      mass_pinn,
        "mass_plain":     mass_plain,
        "pde_curve_pinn": pde_curve_pinn,
        "pde_curve_plain":pde_curve_plain,
        "pinn_pde_map":   pinn_pde_map,
        "plain_pde_map":  plain_pde_map,
        "pinn_pde_mx":    round(pinn_pde_mx,4),
        "plain_pde_mx":   round(plain_pde_mx,4),
        "div_maps":       div_maps,
        "peak_div_idx":   peak_div_idx,
        "extrap_pinn":    extrap_pinn,
        "extrap_plain":   extrap_plain,
        "extrap_max":     {"pinn":extrap_pinn_max,"plain":extrap_plain_max},
        "pinn_frames":    pinn_frames_ds,
        "plain_frames":   plain_frames_ds,
        "peak_pinn":      peak_pinn,
        "peak_plain":     peak_plain,
    }
    print(f"[compare_detail] Done. Physics improvement: {pde_improv:.1f}%")
    return result


if __name__ == "__main__":
    data = run_detailed_comparison()
    DEST = os.path.join(ROOT,"webapp","static","comparison.js")
    os.makedirs(os.path.dirname(DEST),exist_ok=True)
    js = "// Auto-generated by compare_detailed.py\nconst COMPARISON = "+json.dumps(data,separators=(',',':'))+";\n"
    with open(DEST,"w",encoding="utf-8") as f: f.write(js)
    print(f"Written: {DEST}  ({os.path.getsize(DEST)//1024} KB)")
