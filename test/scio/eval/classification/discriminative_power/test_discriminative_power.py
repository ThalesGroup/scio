"""Test classification benchmarking functions."""

import random
import re
from itertools import chain, product

import numpy as np
import pytest

from scio.eval import AUC, TNR, TPR, BaseDiscriminativePower
from scio.utils import InterpolationKind

from .conftest import EXPECTED, LABELS_AND_SCORES, METRICS


def test_discriminative_power_checks_params_on_init(make_dp_check_params_tester, null):
    """Check that instantiation triggers :meth:`_check_params`."""
    Dp = make_dp_check_params_tester(null)

    # Instantiating with ``null`` should work
    Dp(obj=null)

    # Otherwise should fail
    msg = f"Expected: {null!r}. Got: None"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        Dp(obj=None)


def test_discriminative_power_checks_params_on_update(
    make_dp_check_params_tester,
    null,
):
    """Check that changing parameter triggers :meth:`_check_params`."""
    Dp = make_dp_check_params_tester(null)
    dp = Dp(obj=null)

    msg = f"Expected: {null!r}. Got: None"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        dp.obj = None


def test_discriminative_power_uses_from_roc(
    make_dp_check_params_tester,
    null,
):
    """Check that changing parameter triggers :meth:`_check_params`."""
    val = random.random()  # noqa: S311 (not cryptographically safe)
    Dp = make_dp_check_params_tester(null, val=val)
    dp = Dp(obj=null)

    assert dp([True, False], [0, 1]) == val


def test_discriminative_power_requires_from_roc():
    """Check that abstract class requires implementations."""
    msg = (
        "Can't instantiate abstract class BaseDiscriminativePower without an "
        "implementation for abstract method 'from_roc'"
    )
    with pytest.raises(TypeError, match=f"^{re.escape(msg)}$"):
        BaseDiscriminativePower()


@pytest.mark.parametrize(
    ("labels_and_scores", "metric", "expected"),
    [
        (labels_and_scores, metric, expected)
        for (metric, labels_and_scores), expected in zip(
            product(METRICS, LABELS_AND_SCORES),
            chain.from_iterable(EXPECTED),
            strict=True,
        )
    ],
)
def test_metrics_explicitly(labels_and_scores, metric, expected):
    """Test explicit values for implemented Discriminative Powers."""
    np.testing.assert_allclose(metric(*labels_and_scores), expected, strict=True)


@pytest.mark.parametrize(
    ("DiscriminativePower", "pos_arg", "neg_arg"),
    [
        (TPR, "max_fpr", "min_tnr"),
        (TNR, "min_tpr", "max_fnr"),
    ],
)
def test_TPR_TNR_exactly_one_limit(DiscriminativePower, pos_arg, neg_arg):
    """Test :class:`TPR` and :class:`TNR` min/max checks.

    Exactly one should be defined. Both are authorized if compatible
    (summing to :math:`1`).

    """
    metric = DiscriminativePower()

    # None defined: FAILS
    msg = f"Specify exactly one of {pos_arg!r} or {neg_arg!r}, not none"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        metric([True, False], [0, 1])

    # Both defined but compatible: PASSES
    setattr(metric, pos_arg, 0.3)
    setattr(metric, neg_arg, 0.7)
    metric([True, False], [0, 1])

    # Both defined but incompatible: FAILS
    setattr(metric, pos_arg, 0.4)
    msg = f"Specify exactly one of {pos_arg!r} or {neg_arg!r}, not both"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        metric([True, False], [0, 1])


@pytest.mark.parametrize("metric", [AUC(), TPR(max_fpr=0.2), TNR(max_fnr=0.2)])
def test_catch_new_unexpected_kind(metric, monkeypatch_enum_new_member):
    """Check that metrics catch new unexpected kind."""
    value = monkeypatch_enum_new_member(InterpolationKind)
    metric.kind = value

    msg = f"Unsupported kind of {type(metric).__name__}: {value!r}"
    with pytest.raises(NotImplementedError, match=f"^{re.escape(msg)}$"):
        metric([True, False], [0, 1])
