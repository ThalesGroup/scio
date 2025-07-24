"""Test index utils."""

import re

import pytest
import torch

from scio.scores.utils import Index, make_indexes, normalize_samples
from scio.scores.utils.index import IndexType
from scio.utils import IndexMetric
from test.conftest import parametrize_bool

from ..conftest import (  # noqa: TID252 (relative import)
    N_POPULATION,
    SAMPLE_DIM,
    TOL32BITS,
)


def test_unsupported_metric(monkeypatch_enum_new_member):
    """Check that :func:`IndexType` raises on unexpected metric."""
    metric_like = monkeypatch_enum_new_member(IndexMetric)

    msg = f"Unsupported metric: {metric_like!r}"
    with pytest.raises(NotImplementedError, match=f"^{re.escape(msg)}$"):
        IndexType(metric_like)


def test_add_and_ntotal(index, population):
    """Test :meth:`add` and :attr:`ntotal` together (simpler)."""
    assert index.ntotal == 0

    index.add(population)
    assert index.ntotal == N_POPULATION


def test_dim(index):
    """Test :attr:`dimn`."""
    assert index.dim == SAMPLE_DIM


@pytest.mark.parametrize("k", [1, N_POPULATION, N_POPULATION + 1])
def test_search_shape_dtype_device(index, query, k):
    """Test the shape, dtype and device of search results."""
    D, I = index.search(query, k)  # noqa: E741 (ambiguous I)

    assert D.shape == (len(query), k)
    assert D.dtype is query.dtype
    assert D.device == query.device

    assert I.shape == (len(query), k)
    assert I.dtype is torch.long
    assert I.device == query.device


def test_search_trivial(index, population):
    """Test search result on trivial expectation."""
    # Normalize to ensure ``x`` being the closest sample to ``x``.
    population_unit = normalize_samples(population, ord=2)
    index.add(population_unit)
    I = index.search(population_unit, 1)[1]  # noqa: E741 (ambiguous I)

    I_expected = torch.arange(N_POPULATION).to(I)[:, None]
    torch.testing.assert_close(I, I_expected)


def test_search_expectations(search_result, match_array):
    """Test search result compared to stored expectations."""
    D, I = search_result  # noqa: E741 (Ambiguous I)
    match_array(D, **TOL32BITS)
    match_array(I, rtol=0, atol=0)


def test_search_self_query():
    """Check that self query removes self when closest neighbor."""
    population = torch.tensor([[0.1, 0.1], [1.0, 2.0]])
    index = Index(dim=2, metric="ip")
    index.add(population)

    result_normal = index.search(population, 1)
    result_self_query = index.search(population, 1, self_query=True)

    expected_normal = (torch.tensor([[0.3], [5.0]]), torch.tensor([[1], [1]]))
    expected_self_query = (torch.tensor([[0.3], [0.3]]), torch.tensor([[1], [0]]))
    torch.testing.assert_close(result_normal, expected_normal)
    torch.testing.assert_close(result_self_query, expected_self_query)


def test_remove_ids(index_population, query):
    """Ensure that search result changes after sample removal."""
    single_query = query[[0]]
    D_closest_before, I_closest = index_population.search(single_query, 1)

    index_population.remove_ids(I_closest)
    D_closest_after = index_population.search(single_query, 1)[0]
    assert not torch.allclose(D_closest_after, D_closest_before)


@pytest.mark.parametrize("metric", IndexMetric)
def test_metric_and_D_is_similarity(metric):
    """Test trivial properties."""
    index = Index(dim=0, metric=metric)
    expected_D_is_similarity = metric in {IndexMetric.IP}
    assert index.metric == metric
    assert index.D_is_similarity is expected_D_is_similarity


@pytest.mark.parametrize("metric", IndexMetric)
@pytest.mark.parametrize("n_groups", [None, 1, 2])
@parametrize_bool("samples_as_tuple, with_groups, groups_as_tuple, squeeze")
def test_make_indexes(
    population,
    samples_as_tuple,
    with_groups,
    groups_as_tuple,
    n_groups,
    metric,
    squeeze,
    search_result,
    query,
):
    """Test :func:`make_indexes` utility.

    Complex test, in part because of the nested (potentially squeezed)
    tuple structure of the output. Only builds normal index with
    ``population`` as reference, or empty index, but in all different
    API call setups.
    """
    all_samples = (population,) if samples_as_tuple else population
    if with_groups:
        groups = torch.zeros(len(population), device=population.device)
        all_groups = (groups,) if groups_as_tuple else groups
    else:
        all_groups = n_groups = None

    if n_groups is None and with_groups:
        msg = (
            "Since `all_groups is not None`, `n_groups` must be a positive integer "
            "(got None)"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
            indexes = make_indexes(
                all_samples,
                all_groups,
                n_groups=n_groups,
                metric=metric,
                squeeze=squeeze,
            )

        return

    indexes = make_indexes(
        all_samples,
        all_groups,
        n_groups=n_groups,
        metric=metric,
        squeeze=squeeze,
    )

    expect_squeeze_0 = squeeze and not (with_groups and groups_as_tuple)
    expect_squeeze_1 = squeeze and not samples_as_tuple
    expect_squeeze_2 = squeeze and not with_groups

    if not expect_squeeze_0:
        assert len(indexes) == 1
        indexes = indexes[0]

    if not expect_squeeze_1:
        assert len(indexes) == 1
        indexes = indexes[0]

    if not expect_squeeze_2:
        assert len(indexes) == 1 if n_groups is None else n_groups

        if n_groups == 2:
            empty_index = indexes[1]
            assert empty_index.ntotal == 0

    # Check search result for the index built with (expectedly) ``population``
    index = indexes if expect_squeeze_2 else indexes[0]
    observed_result = index.search(query, index.ntotal + 1)
    torch.testing.assert_close(observed_result, search_result)
