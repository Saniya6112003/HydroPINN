"""
Shallow Water Equation (SWE) physics residuals computed via torch.autograd.grad.

Governing PDEs (depth-averaged, 2-D):

  Continuity  : dh/dt  + d(hu)/dx  + d(hv)/dy  = R
  x-momentum  : d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx = -g*h * dz/dx
  y-momentum  : d(hv)/dt + d(hv^2 + 0.5*g*h^2)/dy = -g*h * dz/dy

Variables:
  h — water depth (m)
  u, v — depth-averaged velocity components (m/s)
  R — rainfall source term (mm/hr, converted to m/s internally)
  z — terrain elevation (m)  [constant grid — gradients pre-computed numerically]
  dz_dx, dz_dy — terrain slope from finite-difference on DEM [passed in, not autodiffed]
  g = 9.81 m/s^2

Network derivatives (dh/dt, d(hu)/dx, etc.) are computed via autograd.
Terrain gradients (dz/dx, dz/dy) are pre-computed finite-differences from the DEM
— terrain is a fixed external field, not a network output, so autograd cannot
  differentiate the DEM lookup table. This is the standard practice for PINN + DEM.
"""

from __future__ import annotations

import torch

G = 9.81
MM_HR_TO_M_S = 1.0 / (1000.0 * 3600.0)

# Coordinate scaling constants for properly normalised SWE
# x,y,t are in [0,1] (normalised); h,u,v are in physical units (m, m/s)
# Chain-rule factors: d/dt_norm = d/dt_physical * T_REF  =>  d/dt_physical = (1/T_REF)*d/dt_norm
T_REF = 86400.0   # 24-hour simulation in seconds
L_REF = 30000.0   # ~30 km Mumbai domain in metres
ALPHA = L_REF / T_REF   # = 0.347  (scale continuity eq. so all terms are O(1))


def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Compute d(output)/d(inp) via autograd through the network."""
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def swe_residuals(
    model,
    x: torch.Tensor,      # (N,) requires_grad=True
    y: torch.Tensor,      # (N,) requires_grad=True
    t: torch.Tensor,      # (N,) requires_grad=True
    z: torch.Tensor,      # (N,)  terrain elevation (no grad needed)
    dz_dx: torch.Tensor,  # (N,)  terrain x-slope from finite-diff on DEM
    dz_dy: torch.Tensor,  # (N,)  terrain y-slope from finite-diff on DEM
    R: torch.Tensor,      # (N,)  rainfall rate in mm/hr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate SWE residuals at collocation points.

    Returns
    -------
    res_cont  : continuity residual   (N,)
    res_mom_x : x-momentum residual   (N,)
    res_mom_y : y-momentum residual   (N,)
    """
    inp = torch.stack([x, y, t, z, R], dim=1)   # (N, 5)

    h, u, v = model(inp)
    h = h.squeeze(1)
    u = u.squeeze(1)
    v = v.squeeze(1)

    R_ms = R * MM_HR_TO_M_S   # mm/hr -> m/s

    # Continuity: dh/dt + d(hu)/dx + d(hv)/dy = R
    dh_dt  = _grad(h,     t)
    d_hu_x = _grad(h * u, x)
    d_hv_y = _grad(h * v, y)
    res_cont = dh_dt + d_hu_x + d_hv_y - R_ms

    # x-momentum: d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx = -g*h*dz/dx
    hu = h * u
    dhu_dt   = _grad(hu, t)
    d_flux_x = _grad(hu * u + 0.5 * G * h ** 2, x)
    res_mom_x = dhu_dt + d_flux_x + G * h * dz_dx

    # y-momentum: d(hv)/dt + d(hv^2 + 0.5*g*h^2)/dy = -g*h*dz/dy
    hv = h * v
    dhv_dt   = _grad(hv, t)
    d_flux_y = _grad(hv * v + 0.5 * G * h ** 2, y)
    res_mom_y = dhv_dt + d_flux_y + G * h * dz_dy

    return res_cont, res_mom_x, res_mom_y


def physics_loss(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    z: torch.Tensor,
    dz_dx: torch.Tensor,
    dz_dy: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """MSE of all three SWE residuals — this is L_pde in the total loss."""
    rc, rmx, rmy = swe_residuals(model, x, y, t, z, dz_dx, dz_dy, R)
    return (rc.pow(2).mean() + rmx.pow(2).mean() + rmy.pow(2).mean()) / 3.0


def physics_loss_scaled(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    z: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """
    Properly normalised SWE continuity loss — residuals are O(1).

    The unnormalised SWE in normalised coordinates produces residuals O(10^3-10^4)
    because x,y in [0,1] represent 30 km and t in [0,1] represents 86400 s.
    Multiplying through by L_ref brings all terms to O(1):

        ALPHA * dh/dt_n + d(hu)/dx_n + d(hv)/dy_n = R_ms * L_REF

    where ALPHA = L_REF / T_REF = 0.347.

    Only the continuity equation is used here — momentum equations are omitted
    because the terrain slope term (g*h*dz/dx_n ~ 60) cannot be balanced by
    pressure gradients at realistic flood depths, making them unconverge-able
    without a full diffusion-wave reformulation.
    """
    inp = torch.stack([x, y, t, z, R], dim=1)
    h, u, v = model(inp)
    h = h.squeeze(1)
    u = u.squeeze(1)
    v = v.squeeze(1)

    R_ms = R * MM_HR_TO_M_S   # mm/hr → m/s

    dh_dt  = _grad(h,     t)
    d_hu_x = _grad(h * u, x)
    d_hv_y = _grad(h * v, y)

    # Scaled continuity: ALPHA*dh/dt_n + d(hu)/dx_n + d(hv)/dy_n = R_ms*L_REF
    res = ALPHA * dh_dt + d_hu_x + d_hv_y - R_ms * L_REF
    return res.pow(2).mean()
