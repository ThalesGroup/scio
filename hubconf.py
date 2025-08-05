"""Configure the ``torch.hub`` entrypoint."""

dependencies = ["torch"]

from pathlib import Path

import torch

from tiniest import Tiniest as _Tiniest


def tiniest(*, device: str | torch.device = "cpu") -> _Tiniest:
    """Tiniest model, trained on huggingface.co/ego-thales/cifar10."""
    model = _Tiniest().to(device=device)
    path_weights = Path(__file__).with_name("tiniest.pt")
    state_dict = torch.load(path_weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model
