"""Fixtures for functional tests of classification scores."""

import pytest
import torch
from torch import Tensor

from scio.recorder import Recorder
from test.scio.conftest import Net3DFactory

N_CALIB = 100  # More than k from test_scores_functional.yaml
N_CLASSES = 3
N_TEST = 8


@pytest.fixture
def net(dtype_device) -> Net3DFactory:
    """Create toy network for 3D samples."""
    return Net3DFactory(height=6, width=6, n_classes=N_CLASSES, **dtype_device)


@pytest.fixture
def calib_data(net, dtype_device) -> Tensor:
    """Generate random calibration data."""
    return torch.randn(N_CALIB, *net.sample_shape, **dtype_device)


@pytest.fixture
def calib_labels(calib_data) -> Tensor:
    """Generate random calibration labels.

    All classes except one are represented, on purpose.
    """
    all_but_one = torch.arange(N_CLASSES - 1)
    remaining = torch.randint(0, N_CLASSES - 1, (N_CALIB - N_CLASSES + 1,))
    return torch.cat([all_but_one, remaining]).to(calib_data.device)


@pytest.fixture
def test_data(net, dtype_device) -> Tensor:
    """Generate random test data."""
    return torch.rand(N_TEST, *net.sample_shape, **dtype_device)


@pytest.fixture
def test_out_expected(net, test_data) -> Tensor:
    """Compute expected out for net on test data."""
    return net(test_data)


@pytest.fixture
def rnet(net, test_data) -> Recorder:
    """Test recorder net."""
    rnet = Recorder(net, input_data=test_data)
    rnet.record((1, 2), (1, 4), (1, 5))  # Arbitrary
    return rnet
