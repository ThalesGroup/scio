"""Fixtures for roc tests."""

from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray

from scio.eval import ROC

N_SAMPLES = 100  # >= 2
T = True
F = False
I = np.inf  # noqa: E741 (ambiguous name)


@pytest.fixture
def labels(rng) -> NDArray[np.bool]:
    """Random labels for implicit tests."""
    return np.r_[[T, F], rng.choice([T, F], N_SAMPLES - 2, replace=True)]


@pytest.fixture
def scores(rng, labels) -> NDArray[np.integer]:
    """Random scores for implicit tests."""
    return rng.permutation(len(labels))


@pytest.fixture
def roc(labels, scores) -> ROC:
    """:class:`ROC` instance for implicit tests."""
    return ROC(labels, scores)


LABELS_SCORES = (
    ((T, F), range(2)),
    ((F, T), range(2)),
    ((F, T, F, T), range(4)),
    ((F, T, F, T, T, F, T), (0, 1, 2, 4, 4, 5, 6)),
)
# Compute MANUALLY with ``scores = range(len(labels))``.
EXPECTED = MappingProxyType({
    "pareto": (
        ((0, 1),),
        ((0, 0), (1, 1)),
        ((0, 0), (1, 1), (2, 2)),
        ((0, 0), (1, 1), (2, 3), (3, 4)),
    ),
    "thresholds": (
        ((0.0, 1),),
        ((-I, 0.0), (1.0, I)),
        ((-I, 0.0), (1.0, 2.0), (3.0, I)),
        ((-I, 0.0), (1.0, 2.0), (4.0, 5.0), (6.0, I)),
    ),
    "pareto_ch": (
        ((0, 1),),
        ((0, 0), (1, 1)),
        ((0, 0), (2, 2)),
        ((0, 0), (2, 3), (3, 4)),
    ),
    "thresholds_ch": (
        ((0.0, 1),),
        ((-I, 0.0), (1.0, I)),
        ((-I, 0.0), (3.0, I)),
        ((-I, 0.0), (4.0, 5.0), (6.0, I)),
    ),
    "tot": (
        2,
        2,
        4,
        7,
    ),
    "FP": (
        (0,),
        (0, 1),
        (0, 1, 2),
        (0, 1, 2, 3),
    ),
    "FPch": (
        (0,),
        (0, 1),
        (0, 2),
        (0, 2, 3),
    ),
    "TP": (
        (1,),
        (0, 1),
        (0, 1, 2),
        (0, 1, 3, 4),
    ),
    "TPch": (
        (1,),
        (0, 1),
        (0, 2),
        (0, 3, 4),
    ),
    "FN": (
        (0,),
        (1, 0),
        (2, 1, 0),
        (4, 3, 1, 0),
    ),
    "FNch": (
        (0,),
        (1, 0),
        (2, 0),
        (4, 1, 0),
    ),
    "TN": (
        (1,),
        (1, 0),
        (2, 1, 0),
        (3, 2, 1, 0),
    ),
    "TNch": (
        (1,),
        (1, 0),
        (2, 0),
        (3, 1, 0),
    ),
    "FPR": (
        (0.0,),
        (0.0, 1.0),
        (0.0, 1 / 2, 1.0),
        (0.0, 1 / 3, 2 / 3, 1.0),
    ),
    "FPRch": (
        (0.0,),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 2 / 3, 1.0),
    ),
    "TPR": (
        (1.0,),
        (0.0, 1.0),
        (0.0, 1 / 2, 1.0),
        (0.0, 1 / 4, 3 / 4, 1.0),
    ),
    "TPRch": (
        (1.0,),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 3 / 4, 1.0),
    ),
    "FNR": (
        (0.0,),
        (1.0, 0.0),
        (1.0, 1 / 2, 0.0),
        (1.0, 3 / 4, 1 / 4, 0.0),
    ),
    "FNRch": (
        (0.0,),
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 1 / 4, 0.0),
    ),
    "TNR": (
        (1.0,),
        (1.0, 0.0),
        (1.0, 1 / 2, 0.0),
        (1.0, 2 / 3, 1 / 3, 0.0),
    ),
    "TNRch": (
        (1.0,),
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 1 / 3, 0.0),
    ),
})
