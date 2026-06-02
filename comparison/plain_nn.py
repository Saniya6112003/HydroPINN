"""
Plain NN baseline — identical architecture to HydroPINN but with physics loss disabled.

λ_pde = 0.0  ->  the model sees no Shallow Water Equation constraints.
It can predict negative water depth, violate mass conservation, etc.

This is the control in the PINN vs Plain NN comparison (WOW FACTOR 2).
Saves checkpoint to outputs/plain_nn_checkpoint.pt
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.pinn import HydroPINN
from model.data_loss import generate_observation_points, data_loss

CHECKPOINT  = os.path.join(_ROOT, "outputs", "plain_nn_checkpoint.pt")
ADAM_EPOCHS = 5_000
ADAM_LR     = 1e-3


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def train_plain_nn(
    adam_epochs: int = ADAM_EPOCHS,
    verbose: bool = True,
) -> HydroPINN:
    """Train the plain NN (no physics loss) and save checkpoint."""
    device = _device()
    print(f"[plain_nn] Device: {device}")
    os.makedirs(os.path.join(_ROOT, "outputs"), exist_ok=True)

    # Load DEM
    dem_path = os.path.join(_ROOT, "data", "terrain", "dem.npy")
    dem = (np.load(dem_path).astype(np.float32)
           if os.path.exists(dem_path)
           else np.zeros((64, 64), dtype=np.float32))

    model = HydroPINN().to(device)
    obs   = generate_observation_points(n_obs=200, dem=dem)

    optimiser = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

    print(f"[plain_nn] Training plain NN for {adam_epochs} epochs (no physics loss) …")
    for epoch in tqdm(range(1, adam_epochs + 1), disable=not verbose, ncols=80):
        optimiser.zero_grad()
        L = data_loss(model, obs, device)   # λ_pde = 0 — only data loss
        L.backward()
        optimiser.step()

        if verbose and epoch % 500 == 0:
            tqdm.write(f"  epoch {epoch:5d}  data_loss={L.item():.4e}")

    torch.save({"model_state": model.state_dict()}, CHECKPOINT)
    print(f"[plain_nn] Checkpoint saved -> {CHECKPOINT}")
    return model


if __name__ == "__main__":
    train_plain_nn()
