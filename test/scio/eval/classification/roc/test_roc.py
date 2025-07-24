"""Test ROC utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scio.eval import ROC
from test.conftest import parametrize_bool

from .conftest import EXPECTED, LABELS_SCORES, F, I, T


def test_roc_trivial():
    """Check trivial example works."""
    ROC([T, F], [0, 0])


@parametrize_bool("label")
def test_roc_requires_positive_and_negative(label):
    """Test that both positive and negative samples are required."""
    with pytest.raises(AssertionError, match="^$"):
        ROC([label], [0])


def test_roc_rejects_nan():
    """Test that ``nan`` are forbidden."""
    with pytest.raises(AssertionError, match="^$"):
        ROC([T, F], [0, np.nan])


@pytest.mark.parametrize("sign", [-1, 1])
def test_roc_allows_only_posinf(sign):
    """Test that infinte values must only be positive."""
    if sign > 0:
        ROC([T, F], [0, I])
    else:
        with pytest.raises(AssertionError, match="^$"):
            ROC([T, F], [0, -I])


def test_roc_sorts(rng, labels, scores):
    """Check that order does not matter."""
    shuffler = rng.permutation(len(labels))

    roc = ROC(labels, scores)
    roc_shuffled = ROC(labels[shuffler], scores[shuffler])

    np.testing.assert_allclose(roc.pareto, roc_shuffled.pareto, strict=True)


def test_roc_pareto_skips_unatainable_points_on_equal_score():
    """Check that unatainable points are ignored for Pareto computation.

    Such points may occur when there are equal scores.
    """
    labels = [T, T, F]

    scores = [0, 1, 2]
    pareto = ROC(labels, scores).pareto
    expected = [[0, 2]]
    np.testing.assert_allclose(pareto, expected, strict=True)

    scores = [0, 1, 1]
    pareto = ROC(labels, scores).pareto
    expected = [[0, 1], [1, 2]]
    np.testing.assert_allclose(pareto, expected, strict=True)


@pytest.mark.parametrize(
    ("attr", "labels_scores", "expected"),
    [
        (attr, labels_scores, expected)
        for attr in EXPECTED
        for labels_scores, expected in zip(LABELS_SCORES, EXPECTED[attr], strict=True)
    ],
)
def test_roc_explicit(attr, labels_scores, expected):
    """Check explicitly expected value for every property."""
    roc = ROC(*labels_scores)
    value = getattr(roc, attr)

    np.testing.assert_allclose(value, expected, strict=True)


@parametrize_bool("legend, with_ax")
def test_roc_plot(roc, legend, with_ax, match_plots):  # noqa: ARG001 (unused argument)
    """Check random sample plot."""
    none = roc.plot(legend=legend, ax=plt.subplot(1, 2, 1) if with_ax else None)
    assert none is None
    plt.show()
