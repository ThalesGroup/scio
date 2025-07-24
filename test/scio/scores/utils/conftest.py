"""Fixtures for all utils tests."""

import math
from types import MappingProxyType

import pytest
import torch
from torch import Tensor

from scio.scores.utils import Index
from scio.utils import IndexMetric

SAMPLE_SHAPE = (2, 3)
SAMPLE_DIM = math.prod(SAMPLE_SHAPE)
N_POPULATION = 50
N_QUERY = 5
TOL32BITS = MappingProxyType({"rtol": 1.3e-6, "atol": 1e-5})


@pytest.fixture
def population(dtype_device_gen) -> Tensor:
    """Prepare reference population for future index."""
    return torch.randn(N_POPULATION, *SAMPLE_SHAPE, **dtype_device_gen)


@pytest.fixture
def query(dtype_device_gen) -> Tensor:
    """Prepare query samples for future :meth:`search` calls."""
    return torch.randn(N_QUERY, *SAMPLE_SHAPE, **dtype_device_gen)


@pytest.fixture(params=IndexMetric)
def metric(request) -> IndexMetric:
    """Loop over implemented index metrics."""
    return request.param


@pytest.fixture
def index(metric) -> Index:
    """Empty index for the ``metric``."""
    return Index(dim=SAMPLE_DIM, metric=metric)


@pytest.fixture
def index_population(index, population) -> Index:
    """Index trained on ``population``."""
    index.add(population)
    return index


@pytest.fixture
def search_result(index_population, query) -> tuple[Tensor, Tensor]:
    """Return search result for ``query``, with ``k = n_pop + 1``."""
    return index_population.search(query, index_population.ntotal + 1)
