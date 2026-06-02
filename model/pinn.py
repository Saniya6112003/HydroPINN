"""
HydroPINN — Physics-Informed Neural Network for urban flood prediction.

Input  : [x, y, t, z, R]   (5 features)
          x, y  — normalised spatial coordinates ∈ [0, 1]
          t     — normalised time ∈ [0, 1]
          z     — terrain elevation (m)  from SRTM DEM
          R     — rainfall rate (mm/hr)

Output : [h, u, v]
          h — water depth (m)   always ≥ 0 via softplus
          u — x-velocity (m/s)
          v — y-velocity (m/s)

Architecture: 6 hidden layers × 128 neurons, tanh, Dropout(p=0.1)
Dropout stays ON at inference for MC Dropout uncertainty quantification.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


class HydroPINN(nn.Module):
    def __init__(self,
                 in_features: int = 5,
                 hidden_dim: int = 128,
                 n_layers: int = 6,
                 dropout_p: float = 0.1):
        super().__init__()

        layers = []
        in_dim = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout_p))
            in_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)

        # Separate output heads — makes it easy to apply softplus only on h
        self.head_h = nn.Linear(hidden_dim, 1)   # raw water depth
        self.head_u = nn.Linear(hidden_dim, 1)   # x-velocity
        self.head_v = nn.Linear(hidden_dim, 1)   # y-velocity

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Start h predictions near 0.7m (softplus(0.35) ≈ 0.97) to avoid near-zero local minimum
        nn.init.constant_(self.head_h.bias, 0.35)

    def forward(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyt : Tensor shape (N, 5) — columns are [x, y, t, z, R]

        Returns
        -------
        h : Tensor (N, 1)  water depth in metres, always ≥ 0
        u : Tensor (N, 1)  x-velocity
        v : Tensor (N, 1)  y-velocity
        """
        # Normalize z [2.9, 15m] and R [0, 100 mm/hr] to [0,1] to prevent tanh saturation
        z_norm = (xyt[:, 3:4] - 2.9) / 12.1
        R_norm = xyt[:, 4:5] / 100.0
        inp = torch.cat([xyt[:, :3], z_norm, R_norm], dim=1)
        feat = self.hidden(inp)
        h = F.softplus(self.head_h(feat))   # enforce h ≥ 0
        u = self.head_u(feat)
        v = self.head_v(feat)
        return h, u, v


def make_input_tensor(x: torch.Tensor,
                      y: torch.Tensor,
                      t: torch.Tensor,
                      z: torch.Tensor,
                      R: torch.Tensor) -> torch.Tensor:
    """
    Stack five 1-D tensors into a (N, 5) input matrix.
    All tensors must have the same length N and require_grad where needed.
    """
    return torch.stack([x, y, t, z, R], dim=1)


def enable_dropout(model: nn.Module):
    """Switch all Dropout layers to train mode for MC Dropout inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
