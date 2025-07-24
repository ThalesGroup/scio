"""Complementary tests for GradNorm score."""

import re

import pytest

from scio.scores import GradNorm


def test_raises_when_no_params_for_any_recorded_layers(
    rnet,
    calib_data,
    calib_labels,
    test_data,
):
    """Check raise on paramless-only recorded layers."""
    rnet.record((1, 2), (1, 3), (1, 4))
    score = GradNorm().fit(rnet, calib_data, calib_labels)

    msg = (
        "Recorded layers must have at least one learnable parameters for score GradNorm"
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(msg)}$"):
        score(test_data)
