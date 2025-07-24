"""Test classification base score."""

import re

import pytest
import torch

from scio.scores.classification.base import _postprocess_conformity
from scio.utils import ScoreClassifMode

from ..conftest import N_CLASSES  # noqa: TID252 (relative import)
from .conftest import POSTPROCESS_CONFORMITY_EXPLICIT, POSTPROCESS_CONFORMITY_OUT


def test_raises_on_unexpected_mode(monkeypatch_enum_new_member, score, test_data):
    """Check raise on unsupported mode at urntime."""
    member = monkeypatch_enum_new_member(ScoreClassifMode)
    score.mode = member

    msg = f"Unsupported mode: {member!r}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score(test_data)


def test_accepts_unique_conformity(score, class_aggregation, test_data):
    """Check that the score can output one score only per sample."""
    out, conformity = score(test_data)
    same_conformity_over_classes = (conformity == conformity[:, [0]]).all()

    assert conformity.shape == (len(test_data), N_CLASSES)
    assert same_conformity_over_classes == (class_aggregation != "no")


def test_mode_does_not_unfit(score, test_data):
    """Check that changing mode does not unfit."""
    mode1 = "raw"
    mode2 = "diff"

    score.mode = mode1
    score(test_data)
    score.mode = mode2
    score(test_data)


@pytest.mark.parametrize("mode", ScoreClassifMode)
def test_postprocess_conformity_expectations(conformity, mode, match_array):
    """Test ``_postprocess_conformity`` expectations, plus shape and device."""
    out = _postprocess_conformity(conformity, mode)

    assert out.shape == conformity.shape
    assert out.dtype is conformity.dtype
    assert out.device == conformity.device
    match_array(out)


@pytest.mark.parametrize(
    ("mode", "expected_val"),
    POSTPROCESS_CONFORMITY_OUT.items(),
)
def test_postprocess_conformity_explicit(mode, expected_val, dtype_device):
    """Test ``_postprocess_conformity`` for explicit values."""
    conformity = POSTPROCESS_CONFORMITY_EXPLICIT.to(**dtype_device)
    out = _postprocess_conformity(conformity, mode)

    expected = expected_val.to(**dtype_device)
    torch.testing.assert_close(out, expected)
