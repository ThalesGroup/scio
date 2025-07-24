"""Fixtures for FeatureSqueezing tests."""

import pytest
import torch
from torch import Tensor, nn

from scio.recorder import Recorder

N_SAMPLES = 5


@pytest.fixture
def rnet(dtype_device) -> Recorder:
    """Sample recorder net."""
    return Recorder(nn.Flatten(), input_size=(1, 1)).to(**dtype_device)


@pytest.fixture(params=[(1,), (1, 2), (1, 2, 3)])
def sample_shape(request) -> tuple[int, ...]:
    """Shape of samples for various ndim."""
    return request.param


@pytest.fixture
def tensor(sample_shape, dtype_device) -> Tensor:
    """Test input tensor with various sample ndim."""
    return torch.rand(N_SAMPLES, *sample_shape, **dtype_device)
