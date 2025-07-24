"""Fixtures for scores utils.utils."""

from math import log, sqrt
from types import MappingProxyType

import pytest
import torch
from torch import Tensor

SHAPES = ((1, 1, 1), (3, 4, 5, 6))


@pytest.fixture(params=SHAPES)
def shape(request) -> torch.Size:
    """Test the following shapes."""
    return torch.empty(*request.param).shape


@pytest.fixture
def tensor(shape, dtype_device_gen) -> Tensor:
    """Test tensor."""
    return torch.rand(*shape, **dtype_device_gen)


@pytest.fixture(params=[(), (10,), (2, 3, 4)])
def batch_shape(request) -> tuple[int, ...]:
    """Shape of the batch dims of test samples."""
    return request.param


N = torch.nan
Z = 0
I = torch.inf  # noqa: E741 (ambiguous name)

NORMALIZE_SAMPLES_EXPLICIT_INPUT = torch.tensor([
    [1, 2, 3, 4],
    [Z, 2, 3, 4],
    [Z, Z, 3, 4],
    [Z, Z, I, 4],
    [Z, Z, I, I],
    [1, Z, I, I],
    [1, 2, I, I],
    [1, 2, 3, I],
    [1, Z, I, 4],
    [Z, Z, Z, Z],
    [I, I, I, I],
])
NORMALIZE_SAMPLES_ORD = [-I, -115, -2, -0.6, -0.15, 0.15, 0.6, 2, 115, I]
NORMALIZE_SAMPLES_ORD_MIN16BIT = 0.6  # Empirical limit in our tests
NORMALIZE_SAMPLES_SAMPLE_START_DIM = [0, 1, 2]  # At most ``len(shape)``
# Keys are ``(ord, same_inf)``. Compute values MANUALLY!
NORMALIZE_SAMPLES_EXPLICIT_EXPECTED = MappingProxyType({
    (1, False): torch.tensor([
        [1 / 10, 1 / 5, 3 / 10, 2 / 5],
        [Z, 2 / 9, 1 / 3, 4 / 9],
        [Z, Z, 3 / 7, 4 / 7],
        [Z, Z, 1, Z],
        [N, N, N, N],
        [N, N, N, N],
        [N, N, N, N],
        [Z, Z, Z, 1],
        [Z, Z, 1, Z],
        [Z, Z, Z, Z],
        [N, N, N, N],
    ]),
    (1, True): torch.tensor([
        [1 / 10, 1 / 5, 3 / 10, 2 / 5],
        [Z, 2 / 9, 1 / 3, 4 / 9],
        [Z, Z, 3 / 7, 4 / 7],
        [Z, Z, 1, Z],
        [Z, Z, 1 / 2, 1 / 2],
        [Z, Z, 1 / 2, 1 / 2],
        [Z, Z, 1 / 2, 1 / 2],
        [Z, Z, Z, 1],
        [Z, Z, 1, Z],
        [Z, Z, Z, Z],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    ]),
    (-1, False): torch.tensor([
        [25 / 12, 25 / 6, 25 / 4, 25 / 3],
        [1, I, I, I],
        [N, N, N, N],
        [N, N, N, N],
        [N, N, N, N],
        [I, 1, I, I],
        [3 / 2, 3, I, I],
        [11 / 6, 11 / 3, 11 / 2, I],
        [I, 1, I, I],
        [N, N, N, N],
        [I, I, I, I],
    ]),
    (-1, True): torch.tensor([
        [25 / 12, 25 / 6, 25 / 4, 25 / 3],
        [1, I, I, I],
        [2, 2, I, I],
        [2, 2, I, I],
        [2, 2, I, I],
        [I, 1, I, I],
        [3 / 2, 3, I, I],
        [11 / 6, 11 / 3, 11 / 2, I],
        [I, 1, I, I],
        [4, 4, 4, 4],
        [I, I, I, I],
    ]),
})

FGM_DIRECTION_EXPLICIT_GRAD = torch.tensor([
    [1, 0],
    [1, 2],
    [-1, 1],
    [0, 0],
])
# Keys are ``p``. Compute values MANUALLY!
FGM_DIRECTION_EXPLICIT_EXPECTED = MappingProxyType({
    1: torch.tensor([
        [1, 0],
        [0, 1],
        [-0.5, 0.5],
        [0, 0],
    ]),
    2: torch.tensor([
        [1, 0],
        [1 / sqrt(5), 2 / sqrt(5)],
        [-1 / sqrt(2), 1 / sqrt(2)],
        [0, 0],
    ]),
    I: torch.tensor([
        [1, 0],
        [1, 1],
        [-1, 1],
        [0, 0],
    ]),
})
FGM_DIRECTION_P = (1, 1.1, 1.4, 20, I)
FGM_DIRECTION_P_MIN16BIT = 1.4  # Empirical limit in our tests

KNN_LABEL_COUNT_EXPLICIT_POPULATION = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
KNN_LABEL_COUNT_EXPLICIT_N_CLASSES = 5
KNN_LABEL_COUNT_EXPLICIT_LABELS = torch.tensor([3, 0, 1, 1])
KNN_LABEL_COUNT_EXPLICIT_QUERY = torch.tensor([
    [0.1, 0.0],
    [0.6, 0.7],
])
# Keys are ``(k, metric)``. Compute values MANUALLY!
KNN_LABEL_COUNT_EXPLICIT_COUNTS = MappingProxyType({
    (1, "l2"): torch.tensor([
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
    ]),
    (1, "ip"): torch.tensor([
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ]),
    (3, "l2"): torch.tensor([
        [1, 1, 0, 1, 0],
        [1, 2, 0, 0, 0],
    ]),
    (3, "ip"): torch.tensor([
        [1, 2, 0, 0, 0],
        [1, 2, 0, 0, 0],
    ]),
    (4, "ip"): torch.tensor([
        [1, 2, 0, 1, 0],
        [1, 2, 0, 1, 0],
    ]),
    (5, "l2"): torch.tensor([
        [N, N, N, N, N],
        [N, N, N, N, N],
    ]),
})

AK_LPE_COUNT_EXPLICIT_POPULATION = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.5, 0.5],
])
AK_LPE_COUNT_EXPLICIT_QUERY = torch.tensor([
    [0.1, 0.0],
    [0.6, 0.7],
    [0.5, 0.5],
])
# Keys are ``(k, metric)``. Compute values MANUALLY!
AK_LPE_COUNT_EXPLICIT_OUT = MappingProxyType({
    (1, "l2"): torch.tensor([0.01, 0.05, 0]),
    (1, "ip"): torch.tensor([0.1, 1.3, 1.0]),
    (2, "l2"): torch.tensor([0.61, 0.35, 0.5]),
    (2, "ip"): torch.tensor([0.075, 0.675, 0.5]),
    (3, "l2"): torch.tensor([2.23 / 3, 0.45, 0.5]),
    (3, "ip"): torch.tensor([0.05, 0.65, 0.5]),
    (4, "ip"): torch.tensor([N, N, N]),
})

# Unfortunate argument naming "expected" from ``kldiv``
KLDIV_EXPLICIT_EXPECTED = torch.tensor([0, 1, 2, 3])
KLDIV_EXPLICIT_INPUTS = torch.tensor([
    [0, 1, 2, 3],
    [0, 100, 200, 300],
    [0, 1, 1, 1],
    [0, 0, 1, 0],
    [1, 1, 2, 3],
    [0, 1, 2, -3],
    [0, 1, 2, N],
])
# Compute values MANUALLY!
KLDIV_EXPLICIT_DIV = torch.tensor([0, 0, log(4 / 3) / 3, log(3), I, N, N])
