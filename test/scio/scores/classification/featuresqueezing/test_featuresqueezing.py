"""Complementary tests for FeatureSqueezing score."""

import re

import pytest
import torch
from torch import nn

from scio.recorder import Recorder
from scio.scores import FeatureSqueezing


@pytest.mark.parametrize("squeezers", [0, ("bits:1", 0)])
def test_sanitizes_custom_squeezers(squeezers):
    """Check raise on noncallable custom squeezer."""
    score = FeatureSqueezing(squeezers=squeezers)

    msg = "Custom squeezers must be callable (got 0)"
    with pytest.raises(TypeError, match=f"^{re.escape(msg)}$"):
        score.fit(..., [], ...)


@pytest.mark.parametrize("squeezers", ["invalid", ("bits:1", "invalid")])
def test_sanitizes_str_squeezers(squeezers):
    """Check raise on unsupported ``str`` squeezer."""
    score = FeatureSqueezing(squeezers=squeezers)

    msg = "Squeezer 'invalid' is not supported"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score.fit(..., [], ...)


@pytest.mark.parametrize("n_bits", [0, 32])
def test_sanitizes_bits_squeezing_n_bits(n_bits, rnet, tensor):
    """Check that bits squeezing sanitizes ``n_bits`` value."""
    score = FeatureSqueezing(squeezers=f"bits:{n_bits}").fit(rnet, [], ...)

    msg = f"Bits squeezing requires 0<n_bits<32 (got {n_bits!r})"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score(tensor)


def test_sanitizes_median_squeezing_size(rnet, tensor):
    """Check that median squeezing sanitizes ``size`` value."""
    score = FeatureSqueezing(squeezers="median:0").fit(rnet, [], ...)

    msg = "Median squeezing requires a positive size (got 0)"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score(tensor)


@pytest.mark.parametrize("sample_ndim", [0, 4])
def test_sanitizes_median_squeezing_input_ndim(sample_ndim, dtype_device):
    """Check that median squeezing raise on unsupported sample ndim."""
    rnet = Recorder(nn.Identity(), input_size=(1,)).to(**dtype_device)
    score = FeatureSqueezing(squeezers="median:1").fit(rnet, [], ...)
    tensor = torch.rand((1,) * (sample_ndim + 1), **dtype_device)

    msg = f"Individual samples must be 1D, 2D or 3D (got {sample_ndim}D)"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score(tensor)


@pytest.mark.parametrize("size", [1, 2, 3, 100])
def test_median_squeezing_runs_on_1d_2d_3d_samples(size, rnet, tensor):
    """Check that it can handle 1D, 2D and 3D samples."""
    score = FeatureSqueezing(squeezers=f"median:{size}").fit(rnet, [], ...)
    score(tensor)
