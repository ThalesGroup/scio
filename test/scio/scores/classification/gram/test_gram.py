"""Complementary tests for Gram score."""

import re

import pytest
import torch

from scio.utils import LabelKind


def test_raises_unsupported_labelkind(monkeypatch_enum_new_member, score, rnet, tensor):
    """Check raise on unexpected :class:`LabelKind` value."""
    value = monkeypatch_enum_new_member(LabelKind)
    score.calib_labels = value

    msg = f"Unsupported calib_labels_kind: {value!r}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score.fit(rnet, tensor, ...)


@pytest.mark.parametrize("per_sample_act_ndim", [0, 4])
def test_median_squeezing_sanitizes_input_ndim(
    per_sample_act_ndim,
    dtype_device,
    score,
    rnet,
):
    """Check that median squeezing raise on unsupported sample ndim."""
    tensor = torch.rand((1,) * (per_sample_act_ndim + 1), **dtype_device)

    msg = (
        "Gram score only supports per-sample activations as 1D, 2D or 3D tensors (got "
        f"{per_sample_act_ndim}D)"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score.fit(rnet, tensor, ...)


def test_gram_runs_on_1d_2d_3d_samples(score, rnet, tensor):
    """Check that it can handle 1D, 2D and 3D per-sample activations."""
    score.fit(rnet, tensor, ...)
    score(tensor)
