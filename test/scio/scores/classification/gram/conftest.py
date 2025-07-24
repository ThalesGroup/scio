"""Fixtures for Gram tests."""

import pytest
import torch
from torch import Tensor, nn

from scio.recorder import Recorder
from scio.scores import Gram

N_SAMPLES = 5


class NetActDimFactory(nn.Module):
    """Network with predictible first layer activations shape."""

    def __init__(self) -> None:
        super().__init__()
        self.id = nn.Identity()
        self.flat = nn.Flatten()

    def forward(self, x) -> Tensor:
        """Inflate before flattening to handle 1D inputs."""
        return self.flat(self.id(x)[..., None])


@pytest.fixture
def rnet(dtype_device) -> Recorder:
    """Sample recorder net."""
    rnet = Recorder(NetActDimFactory(), input_size=(1, 1)).to(**dtype_device)
    rnet.record((1, 1))
    return rnet


@pytest.fixture(params=[(1,), (1, 2), (1, 2, 3)])
def sample_shape(request) -> tuple[int, ...]:
    """Shape of samples for various ndim."""
    return request.param


@pytest.fixture
def tensor(sample_shape, dtype_device) -> Tensor:
    """Test input tensor with various sample ndim."""
    return torch.rand(N_SAMPLES, *sample_shape, **dtype_device)


@pytest.fixture
def score() -> Gram:
    """Gram score."""
    return Gram()
