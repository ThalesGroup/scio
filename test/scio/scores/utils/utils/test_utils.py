"""Test scores utils.utils."""

import re
from itertools import chain

import pytest
import torch

from scio.scores.utils import (
    Index,
    ak_lpe,
    batched_grad,
    dirmult_surprise,
    fgm_direction,
    get_aggregator,
    kldiv,
    knn_label_count,
    multinomial_test,
    normalize_samples,
    torch_quantile,
)
from scio.utils import AggrName, MultinomialTestMode
from test.conftest import parametrize_bool

from ..conftest import N_POPULATION, TOL32BITS  # noqa: TID252 (relative import)
from .conftest import (
    AK_LPE_COUNT_EXPLICIT_OUT,
    AK_LPE_COUNT_EXPLICIT_POPULATION,
    AK_LPE_COUNT_EXPLICIT_QUERY,
    FGM_DIRECTION_EXPLICIT_EXPECTED,
    FGM_DIRECTION_EXPLICIT_GRAD,
    FGM_DIRECTION_P,
    FGM_DIRECTION_P_MIN16BIT,
    KLDIV_EXPLICIT_DIV,
    KLDIV_EXPLICIT_EXPECTED,
    KLDIV_EXPLICIT_INPUTS,
    KNN_LABEL_COUNT_EXPLICIT_COUNTS,
    KNN_LABEL_COUNT_EXPLICIT_LABELS,
    KNN_LABEL_COUNT_EXPLICIT_N_CLASSES,
    KNN_LABEL_COUNT_EXPLICIT_POPULATION,
    KNN_LABEL_COUNT_EXPLICIT_QUERY,
    NORMALIZE_SAMPLES_EXPLICIT_EXPECTED,
    NORMALIZE_SAMPLES_EXPLICIT_INPUT,
    NORMALIZE_SAMPLES_ORD,
    NORMALIZE_SAMPLES_ORD_MIN16BIT,
    NORMALIZE_SAMPLES_SAMPLE_START_DIM,
    SHAPES,
    I,
    N,
)


@pytest.mark.parametrize("sample_start_dim", NORMALIZE_SAMPLES_SAMPLE_START_DIM)
def test_normalize_samples_shape_dtype_device(tensor, sample_start_dim):
    """Check the output shape, dtype and device match input tensor."""
    out = normalize_samples(tensor, ord=1, sample_start_dim=sample_start_dim)

    assert out.shape == tensor.shape
    assert out.dtype is tensor.dtype
    assert out.device == tensor.device


@pytest.mark.parametrize(
    ("ord_same_inf", "expected_val"),
    NORMALIZE_SAMPLES_EXPLICIT_EXPECTED.items(),
)
def test_normalize_samples_explicit(ord_same_inf, expected_val, dtype_device):
    """Test returned value on explicit examples."""
    tensor = NORMALIZE_SAMPLES_EXPLICIT_INPUT.to(**dtype_device)
    ord, same_inf = ord_same_inf  # noqa: A001 (shadow builtin)
    observed = normalize_samples(tensor, ord=ord, same_inf=same_inf)

    expected = expected_val.to(**dtype_device)
    torch.testing.assert_close(observed, expected, equal_nan=True)


@pytest.mark.parametrize("ord", NORMALIZE_SAMPLES_ORD)
@pytest.mark.parametrize("sample_start_dim", NORMALIZE_SAMPLES_SAMPLE_START_DIM)
def test_normalize_samples_works(dtype, tensor, ord, sample_start_dim):  # noqa: A002 (shadow builtin)
    """Test returned value on random samples via norm expectation."""
    if dtype is torch.half and abs(ord) < NORMALIZE_SAMPLES_ORD_MIN16BIT:
        pytest.skip("`torch.half` precision limit")

    tensor05_2 = 2 ** (1 - 2 * tensor)
    out = normalize_samples(tensor05_2, ord=ord, sample_start_dim=sample_start_dim)

    # Check norm indeed close to ``1``.
    norms = out.flatten(sample_start_dim).norm(ord, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.long, torch.bool, torch.cfloat])
def test_normalize_samples_expects_floating_dtype(dtype, device):
    """Check floating point sanitization."""
    tensor = torch.tensor(0, dtype=dtype, device=device)

    msg = f"Input tensor should be of floating dtype (got {dtype})"
    with pytest.raises(TypeError, match=f"^{re.escape(msg)}$"):
        normalize_samples(tensor, ord=1)


def test_normalize_samples_ord_is_zero(tensor):
    """Check ``ord`` sanitization."""
    msg = "Normalization order cannot be zero"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        normalize_samples(tensor, ord=0)


def test_normalize_samples_ord_is_none(tensor):
    """Check noop when ``ord is None``."""
    copy = tensor.clone()

    out = normalize_samples(tensor, ord=None)
    assert out is tensor
    torch.testing.assert_close(out, copy)


def test_normalize_samples_sensitive_to_signbit():
    """Check signbit sensitivity."""
    tensor = torch.tensor([[1.0, 0.0]])

    # Positive sign bit
    out = normalize_samples(tensor, ord=-1)
    expected = torch.tensor([[I, 1.0]])
    torch.testing.assert_close(out, expected)

    # Negative sign bit
    out = normalize_samples(-tensor, ord=-1)
    expected *= -1
    torch.testing.assert_close(out, expected)


_NOT_PROVIDED = object()


@pytest.mark.parametrize("dim", [_NOT_PROVIDED, 0, 1])
@pytest.mark.parametrize("aggr_or_ord", chain([-I, 0, 2], AggrName))
def test_get_aggregator_expectations(aggr_or_ord, tensor, dim, match_array):
    """Test expectations, plus ndim, dtype and device."""
    aggregator = get_aggregator(aggr_or_ord)

    out = aggregator(tensor) if dim is _NOT_PROVIDED else aggregator(tensor, dim=dim)

    expected_ndim = 0 if dim is _NOT_PROVIDED else tensor.ndim - 1
    assert out.ndim == expected_ndim
    assert out.dtype is tensor.dtype
    assert out.device == tensor.device
    match_array(out)


def test_batched_grad_trivial(tensor):
    """Test on a trivial case."""
    inputs = tensor.requires_grad_()
    outputs = inputs.flatten(1).sum(1)
    ones = torch.ones_like(inputs)

    grads = batched_grad(outputs, inputs)
    torch.testing.assert_close(grads, ones)


def test_batched_grad_retain_graph(tensor):
    """Test multiple uses with ``retain_graph``."""
    inputs = tensor.requires_grad_()
    intermediate = inputs * 1  # Create intermediate computation node
    outputs = intermediate.flatten(1).sum(1)
    ones = torch.ones_like(inputs)

    grads = batched_grad(outputs, inputs, retain_graph=True)
    torch.testing.assert_close(grads, ones)

    # Works a second time since ``retain_graph=True`` was used
    grads = batched_grad(outputs, inputs, retain_graph=False)
    torch.testing.assert_close(grads, ones)

    # Fails after ``retain_graph=False``
    msg = "Trying to backward through the graph a second time"  # [...]
    with pytest.raises(RuntimeError, match=f"^{re.escape(msg)}"):
        grads = batched_grad(outputs, inputs)


@parametrize_bool("zero")
@pytest.mark.parametrize("p", FGM_DIRECTION_P)
def test_fgm_direction_shape_dtype_device(dtype, tensor, zero, p):
    """Check the output shape, dtype and device match input grad."""
    if dtype is torch.half and p < FGM_DIRECTION_P_MIN16BIT:
        pytest.skip("`torch.half` precision limit")

    grad = torch.zeros_like(tensor) if zero else tensor
    out = fgm_direction(grad, p=p)

    assert out.shape == grad.shape
    assert out.dtype is grad.dtype
    assert out.device == grad.device


@pytest.mark.parametrize(
    ("p", "expected_val"),
    FGM_DIRECTION_EXPLICIT_EXPECTED.items(),
)
def test_fgm_direction_explicit(p, expected_val, dtype_device):
    """Test returned value on explicit examples."""
    grad = FGM_DIRECTION_EXPLICIT_GRAD.to(**dtype_device)
    observed = fgm_direction(grad, p=p)

    expected = expected_val.to(**dtype_device)
    torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize("p", FGM_DIRECTION_P)
def test_fgm_direction_works(dtype, tensor, p, atol_rtol):
    """Test on random ``grad`` with Hölder (in)equality expectation."""
    if dtype is torch.half and p < FGM_DIRECTION_P_MIN16BIT:
        pytest.skip("`torch.half` precision limit")

    grad = tensor
    out = fgm_direction(grad, p=p)

    inner_products = (grad * out).flatten(1).sum(1)
    q = 1 + 1 / (p - 1) if p > 1 else I
    expected = torch.linalg.vector_norm(grad.flatten(1), ord=q, dim=1)
    torch.testing.assert_close(inner_products, expected, **atol_rtol)


def test_fgm_direction_check(device_gen):
    """Test the ``check`` option works."""
    grad = torch.rand(10, 1000, dtype=torch.half, **device_gen)

    # Check may fail on over/underflow
    msg = (
        "This is likely due to computational approximations. Consider using p=1 if p "
        "is too close to 1"
    )
    with pytest.raises(AssertionError, match=f"{re.escape(msg)}$"):
        fgm_direction(grad, p=1.0001)

    # Works without ``check``
    fgm_direction(grad, p=1.0001, check=False)


def test_fgm_direction_raises_on_infinite():
    """Check infinite sanitization."""
    grad = torch.rand(2)
    grad[0] = I

    msg = "Only finite gradients are allowed"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        fgm_direction(grad, p=1)


@pytest.mark.parametrize("p", [N, 0.9999, -I])
def test_fgm_direction_sanitizes_p(p):
    """Check ``p`` sanitization."""
    grad = torch.rand(2)

    msg = f"Parameter p must satisfy `p>=1` (got {p!r})"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        fgm_direction(grad, p=p)


@pytest.mark.parametrize(
    "interpolation",
    ["linear", "lower", "higher", "midpoint", "nearest"],
)
@parametrize_bool("keepdim")
@pytest.mark.parametrize(
    "dim",
    [None, *range(min(map(len, SHAPES)))],
)  # https://github.com/pytest-dev/pytest/discussions/13581
@pytest.mark.parametrize("q", [0, 0.1, 0.5, 1])
def test_torch_quantile_works(tensor, q, dim, keepdim, interpolation, dtype, device):
    """Test out ``torch_quantile`` matches ``torch.quantile``."""
    args = (tensor, q)
    kwargs = {
        "dim": dim,
        "keepdim": keepdim,
        "interpolation": interpolation,
    }
    observed = torch_quantile(*args, **kwargs)

    # Partial check for ``torch.half`` unsupported by ``torch.quantile``
    if dtype is torch.half:
        assert observed.dtype is dtype
        assert observed.device == device
    else:
        expected = torch.quantile(*args, **kwargs)
        torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize("q", [-0.001, 2])
def test_torch_quantile_sanitizes_q(q):
    """Test ``q`` sanitization."""
    tensor = torch.tensor(0)

    msg = f"Only values 0<=q<=1 are supported (got {float(q)!r})"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        torch_quantile(tensor, q)


def test_torch_quantile_sanitizes_interpolation():
    """Test ``interpolation`` sanitization."""
    tensor = torch.tensor(0)
    interpolation = "unsupported"

    msg = (
        "Currently supported interpolations are {'linear', 'lower', 'higher', "
        f"'midpoint', 'nearest'}} (got {interpolation!r})"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        torch_quantile(tensor, 0.5, interpolation=interpolation)


def test_torch_quantile_sanitizes_out():
    """Test ``out`` sanitization."""
    tensor = torch.tensor(0)
    out = torch.empty(())

    msg = f"Only None value is currently supported for out (got {out!r})"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        torch_quantile(tensor, 0.5, out=out)


@parametrize_bool("self_query")
@pytest.mark.parametrize("k", [1, N_POPULATION - 1, N_POPULATION + 1])
@pytest.mark.parametrize("n_classes", [1, N_POPULATION // 2, N_POPULATION * 2])
def test_knn_label_count_expectations(
    population,
    index_population,
    n_classes,
    k,
    query,
    self_query,
    device_gen,
    match_array,
):
    """Test ``knn_label_count`` against expectations.

    Also checks: shape, device and total count (or ``nan``).
    """
    labels = torch.randint(0, n_classes, (len(population),), **device_gen)
    query = population if self_query else query

    counts = knn_label_count(
        index_population,
        labels,
        n_classes,
        k,
        query,
        self_query=self_query,
    )
    assert counts.shape == (len(query), n_classes)
    assert counts.device == query.device
    if k + self_query > index_population.ntotal:
        assert counts.isnan().all()
    else:
        assert (counts.sum(1) == k).all()
    match_array(counts, equal_nan=True)


@pytest.mark.parametrize(
    ("k_metric", "expected_val"),
    KNN_LABEL_COUNT_EXPLICIT_COUNTS.items(),
)
def test_knn_label_count_explicit(k_metric, expected_val, dtype_device, device):
    """Test ``knn_label_count`` for explicit values."""
    population = KNN_LABEL_COUNT_EXPLICIT_POPULATION.to(**dtype_device)
    k, metric = k_metric
    index = Index(dim=population[0].numel(), metric=metric)
    index.add(population)
    labels = KNN_LABEL_COUNT_EXPLICIT_LABELS.to(device=population.device)
    query = KNN_LABEL_COUNT_EXPLICIT_QUERY.to(**dtype_device)

    counts = knn_label_count(
        index,
        labels,
        KNN_LABEL_COUNT_EXPLICIT_N_CLASSES,
        k,
        query,
    )

    expected = expected_val.to(device=device)
    torch.testing.assert_close(counts, expected, equal_nan=True)


def test_knn_label_count_sanitizes_labels():
    """Check ``labels`` sanitization."""
    index = Index(dim=1, metric="l2")
    labels = torch.tensor([0])
    n_classes = 1
    k = 1
    query = torch.empty(1, 1)

    # Shape sanitization
    msg = (
        f"`labels` must have shape (index.ntotal,)={(index.ntotal,)} (got "
        f"{labels.shape})"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        knn_label_count(index, labels, n_classes, k, query)

    # Values sanitization
    index.add(query)
    labels[0] = 1
    msg = f"`labels` value out of range at: {(labels >= n_classes).nonzero()}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        knn_label_count(index, labels, n_classes, k, query)


@parametrize_bool("self_query")
@pytest.mark.parametrize("k", [1, N_POPULATION // 2, N_POPULATION])
def test_ak_lpe_expectations(
    population,
    index_population,
    k,
    query,
    self_query,
    match_array,
):
    """Test ``ak_lpe`` expectations, plus shape, dtype and device."""
    query = population if self_query else query

    out = ak_lpe(index_population, k, query, self_query=self_query)
    assert out.shape == (len(query),)
    assert out.dtype is query.dtype
    assert out.device == query.device
    match_array(out, **TOL32BITS, equal_nan=True)


@pytest.mark.parametrize(
    ("k_metric", "expected_val"),
    AK_LPE_COUNT_EXPLICIT_OUT.items(),
)
def test_ak_lpe_explicit(k_metric, expected_val, dtype_device, atol_rtol):
    """Test ``ak_lpe`` for explicit values."""
    population = AK_LPE_COUNT_EXPLICIT_POPULATION.to(**dtype_device)
    k, metric = k_metric
    index = Index(dim=population[0].numel(), metric=metric)
    index.add(population)
    query = AK_LPE_COUNT_EXPLICIT_QUERY.to(**dtype_device)

    out = ak_lpe(index, k, query)

    expected = expected_val.to(**dtype_device)
    torch.testing.assert_close(out, expected, **atol_rtol, equal_nan=True)


@pytest.mark.parametrize("space_size", [1, 10, 100])
def test_kldiv_expectations(
    batch_shape,
    space_size,
    dtype_device_gen,
    device,
    match_array,
):
    """Test ``kldiv`` expectations, plus shape and device."""
    expected_ = torch.rand(space_size, **dtype_device_gen)
    inputs = torch.rand(*batch_shape, space_size, **dtype_device_gen)

    div = kldiv(inputs, expected_)

    assert div.shape == batch_shape
    assert div.device == device
    match_array(div)


def test_kldiv_explicit(dtype_device):
    """Test ``kldiv`` for explicit values."""
    expected_ = KLDIV_EXPLICIT_EXPECTED.to(**dtype_device)
    inputs = KLDIV_EXPLICIT_INPUTS.to(**dtype_device)

    div = kldiv(inputs, expected_)
    expected = KLDIV_EXPLICIT_DIV.to(**dtype_device)

    torch.testing.assert_close(div, expected, equal_nan=True)


@pytest.mark.parametrize("expected_val", [[0], [1, torch.nan], [1, -2]])
def test_kldiv_nan_for_invalid_expected(expected_val):
    """Check only ``nan`` when invalid ``expected``."""
    expected_ = torch.tensor(expected_val)
    inputs = torch.rand(2, 3, 4, len(expected_))

    div = kldiv(inputs, expected_)

    assert div.isnan().all()


def test_kldiv_sanitizes_expected():
    """Check ``expected`` sanitization."""
    tensor = torch.tensor(0)

    msg = "`expected` must be at least a 1D tensor (got scalar)"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        kldiv(tensor, tensor)


@pytest.mark.parametrize("k", [1, 10, 100])
@pytest.mark.parametrize("n", [1, 5, 20])
def test_dirmult_surprise_expectations(
    batch_shape,
    k,
    n,
    dtype,
    device,
    device_gen,
    dtype_device_gen,
    match_array,
):
    """Test ``dirmult_surprise`` expectations, shape, dtype, device."""
    # `counts` are sampled through their accumulated sums diff, to
    # ensure constant total across last dim
    prepend = torch.full((*batch_shape, 1), 0, device=device)
    append = torch.full((*batch_shape, 1), n, device=device)
    counts_acc = torch.randint(0, n + 1, (*batch_shape, k - 1), **device_gen)
    counts = counts_acc.sort().values.diff(prepend=prepend, append=append)
    alpha = torch.rand(k, **dtype_device_gen)

    res = dirmult_surprise(counts, alpha)

    # The actual test
    assert res.shape == batch_shape
    assert res.dtype is dtype
    assert res.device == device
    match_array(res, equal_nan=True)


@pytest.mark.skip("NotImplemented")
def test_dirmult_surprise_explicit():
    """Test ``dirmult_surprise`` for explicit values."""


def test_dirmult_surprise_sanitizes_counts():
    """Check ``counts`` sanitization."""
    # Nonscalar
    tensor = torch.tensor(0)

    msg = "`counts` must be at least a 1D tensor (got scalar)"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        dirmult_surprise(tensor, tensor)

    # Consistent last axis sum
    tensor = torch.tensor([[0, 1], [1, 1]])

    msg = "`counts` must have constant sum along last axis"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        dirmult_surprise(tensor, tensor)


@pytest.mark.parametrize("k", [1, 10, 100])
@pytest.mark.parametrize("n", [1, 5, 20])
@pytest.mark.parametrize("mode", MultinomialTestMode)
def test_multinomial_test_expectations(
    batch_shape,
    k,
    n,
    mode,
    device,
    device_gen,
    dtype_device_gen,
    dtype,
    match_array,
):
    """Test ``multinomial_test`` expectations, shape, dtype, device."""
    # `counts` and `observed` are sampled through their accumulated
    # sums diff, to ensure constant total across last dim
    prepend = torch.full((*batch_shape, 1), 0, device=device)
    append = torch.full((*batch_shape, 1), n, device=device)
    counts_acc = torch.randint(0, n + 1, (*batch_shape, k - 1), **device_gen)
    counts = counts_acc.sort().values.diff(prepend=prepend, append=append)
    prior = torch.rand(k, **dtype_device_gen)

    assert counts.shape == (*batch_shape, k)
    assert prior.shape == (k,)
    test = multinomial_test(counts, prior, mode)

    observed_acc = torch.randint(0, n + 1, (*batch_shape, k - 1), **device_gen)
    observed = observed_acc.sort().values.diff(prepend=prepend, append=append)

    assert observed.shape == (*batch_shape, k)
    test_result = test(observed)

    # The actual test
    expected_dtype = torch.float32 if mode == MultinomialTestMode.MLE else dtype
    assert test_result.shape == batch_shape
    assert test_result.dtype is expected_dtype
    assert test_result.device == device
    match_array(test_result, equal_nan=True)


@pytest.mark.skip("NotImplemented")
def test_multinomial_test_explicit():
    """Test ``multinomial_test`` for explicit values."""


@pytest.mark.parametrize("mode", MultinomialTestMode)
def test_multinomial_test_sanitizes_counts(mode):
    """Check ``counts`` sanitization."""
    tensor = torch.tensor(0)

    msg = "`counts` must be at least a 1D tensor (got scalar)"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        multinomial_test(tensor, tensor, mode)


def test_multinomial_test_raises_unexpected_mode(monkeypatch_enum_new_member):
    """Check raise on unexpected mode."""
    counts = torch.tensor([1])
    value = monkeypatch_enum_new_member(MultinomialTestMode)

    msg = f"Unsupported mode: {value!r}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        multinomial_test(counts, counts, value)
