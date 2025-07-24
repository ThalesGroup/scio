"""Utility functions."""

__all__ = [
    "ak_lpe",
    "batched_grad",
    "dirmult_surprise",
    "fgm_direction",
    "get_aggregator",
    "kldiv",
    "knn_label_count",
    "multinomial_test",
    "normalize_samples",
    "torch_quantile",
]

import warnings
from collections.abc import Callable
from functools import partial
from math import ceil, floor
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn.functional import kl_div

from scio.utils import (
    AggrName,
    AggrNameLike,
    MultinomialTestMode,
    MultinomialTestModeLike,
)

if TYPE_CHECKING:
    from .index import Index  # pragma: no cover


def normalize_samples(
    tensor: Tensor,
    *,
    ord: float | None,  # noqa: A002 (shadows builtin)
    same_inf: bool = False,
    sample_start_dim: int = 1,
) -> Tensor:
    r"""Vector-normalizes all samples if ``ord`` is provided.

    Arguments
    ---------
    tensor: ``Tensor``
        The samples to normalize. Floating point tensor of shape
        ``(*batch_shape, *sample_shape)``. The norm is computed by
        interpreting every sample as a :math:`1`\ D vector.
    ord: ``float | None``
        If ``None``, the function returns ``tensor`` as is. Otherwise,
        the norm to use. Must be nonzero. Regarding negative ``ord``,
        the following is always true element-wise:
        ``1 / normalize_samples(tensor, ord=ord, **kw) ==
        normalize_samples(1 / tensor, ord=-ord, **kw)``.
    same_inf: ``bool``
        For simplicity, the following description assumes that ``ord >
        0`` and that ``tensor`` has positive coordinates. A
        sample vector norm may be infinite because of a single
        coordinate or several being infinite. In the former case, the
        normalized sample is well-defined as ``[..., 0, 1, 0, ...]``. In
        the latter case, it remains undefined (resulting in ``nan``),
        unless the infinites are given the same importance. This is what
        this option allows, resulting in, for example with ``ord=1`` and
        exactly two infinite coordinates, ``[..., 0, 0.5, 0,..., 0, 0.5,
        0, ...]``. It ensures that the output has unit norm even in
        (nonzero) degenerate cases. Defaults to ``False``.
    sample_start_dim: ``int``
        Defines ``sample_shape = tensor.shape[sample_start_dim:]``.
        Defaults to ``1``.

    Returns
    -------
    out: ``Tensor``
        Normalized samples. Samples with zero (*resp.* infinite when
        ``ord < 0``) vector-norm are unchanged. If ``not same_inf``,
        samples may be changed to ``nan`` in some degenerate cases (see
        ``same_inf``).

    Raises
    ------
    :exc:`TypeError`
        If ``tensor`` is not of floating point dtype.
    :exc:`ValueError`
        If ``ord == 0``.

    Examples
    --------
    ::

        Z, I = 0, torch.inf
        t = torch.tensor([
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

    With ``ord > 0``::

        >>> normalize_samples(t, ord=1)
        tensor([[0.1000, 0.2000, 0.3000, 0.4000],
                [0.0000, 0.2222, 0.3333, 0.4444],
                [0.0000, 0.0000, 0.4286, 0.5714],
                [0.0000, 0.0000, 1.0000, 0.0000],
                [   nan,    nan,    nan,    nan],
                [   nan,    nan,    nan,    nan],
                [   nan,    nan,    nan,    nan],
                [0.0000, 0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [   nan,    nan,    nan,    nan]])
        >>> normalize_samples(t, ord=1, same_inf=True)
        tensor([[0.1000, 0.2000, 0.3000, 0.4000],
                [0.0000, 0.2222, 0.3333, 0.4444],
                [0.0000, 0.0000, 0.4286, 0.5714],
                [0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.5000, 0.5000],
                [0.0000, 0.0000, 0.5000, 0.5000],
                [0.0000, 0.0000, 0.5000, 0.5000],
                [0.0000, 0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.2500, 0.2500, 0.2500, 0.2500]])

    And with ``ord < 0``::

        >>> normalize_samples(t, ord=-1)
        tensor([[2.0833, 4.1667, 6.2500, 8.3333],
                [1.0000,    inf,    inf,    inf],
                [   nan,    nan,    nan,    nan],
                [   nan,    nan,    nan,    nan],
                [   nan,    nan,    nan,    nan],
                [   inf, 1.0000,    inf,    inf],
                [1.5000, 3.0000,    inf,    inf],
                [1.8333, 3.6667, 5.5000,    inf],
                [   inf, 1.0000,    inf,    inf],
                [   nan,    nan,    nan,    nan],
                [   inf,    inf,    inf,    inf]])
        >>> normalize_samples(t, ord=-1, same_inf=True)
        tensor([[2.0833, 4.1667, 6.2500, 8.3333],
                [1.0000,    inf,    inf,    inf],
                [2.0000, 2.0000,    inf,    inf],
                [2.0000, 2.0000,    inf,    inf],
                [2.0000, 2.0000,    inf,    inf],
                [   inf, 1.0000,    inf,    inf],
                [1.5000, 3.0000,    inf,    inf],
                [1.8333, 3.6667, 5.5000,    inf],
                [   inf, 1.0000,    inf,    inf],
                [4.0000, 4.0000, 4.0000, 4.0000],
                [   inf,    inf,    inf,    inf]])


    Note
    ----
    Finite but extreme values of ``ord`` might result in computation
    under/overflow, leading to unexpected ``inf`` or ``nan`` values. For
    32 bits inputs in :math:`[0.5, 2]`, we roughly identified ``0.15 <
    |ord| < 115`` as a *pretty safe* range.

    Note
    ----
    This function meaningfully differentiates between ``0.0`` and
    ``-0.0``::

        >>> normalize_samples(torch.tensor([[1.0, -0.0]]), ord=-1)
        tensor([[inf, -1.]])

    """
    if not tensor.is_floating_point():
        msg = f"Input tensor should be of floating dtype (got {tensor.dtype})"
        raise TypeError(msg)

    if ord is None:
        return tensor
    if ord == 0:
        msg = "Normalization order cannot be zero"
        raise ValueError(msg)
    if ord > 0:
        zero = torch.tensor(0).to(tensor)
        is_singular = torch.isinf
    else:  # ord < 0
        zero = torch.tensor(torch.inf).to(tensor)
        is_singular = lambda input: input == 0  # noqa: E731, A006 (lambda def, shadow builtin)

    tensor_2d = tensor.reshape(tensor.shape[:sample_start_dim].numel(), -1)
    norms = torch.linalg.vector_norm(tensor_2d, ord=ord, dim=1, keepdim=True)
    out_2d = tensor_2d / norms.where(norms != zero, 1)

    # Handle degenerate cases
    mask = is_singular(norms).squeeze(1)
    degenerate = tensor_2d[mask]
    degenerate_sign = torch.where(degenerate.signbit(), -1, 1)
    degenerate_singular = is_singular(degenerate)
    scale = degenerate_singular.sum(1, keepdim=True).pow(1 / ord).type_as(tensor)
    zero_one_replacement = torch.where(degenerate_singular, 1, zero)
    unsigned_replacements = zero_one_replacement / scale

    # Handle `same_inf`
    if not same_inf:
        unsigned_replacements[scale.squeeze(1) != 1] = torch.nan

    out_2d[mask] = unsigned_replacements * degenerate_sign
    return out_2d.reshape(tensor.shape)


def get_aggregator(aggr_or_ord: AggrNameLike | float) -> Callable[..., Tensor]:
    r"""Get a :math:`1`\ D aggregator function.

    Arguments
    ---------
    aggr_or_ord: ``AggrNameLike | float``
        Provide a ``float`` to get a vector norm aggregator, Else, see
        :class:`~scio.utils.AggrName`.

    Returns
    -------
    out: ``Callable[..., Tensor]``
        Aggregation function that accepts an optional ``dim: int``
        argument. Depending on the aggregation method, may require input
        tensor to be of floating type and/or nonempty and/or
        nonnegative. May also accept ``dim: tuple[int, ...] | None``,
        ``keepdim: bool`` and eventually ``dtype``.

    """
    if isinstance(aggr_or_ord, Real):
        return partial(torch.linalg.vector_norm, ord=aggr_or_ord)

    aggr = AggrName(aggr_or_ord)

    fisher_agg = lambda t, *a, **kw: t.log().sum(*a, **kw)  # noqa: E731 (lambda def)
    torch_geom = lambda t, *a, **kw: t.log().mean(*a, **kw).exp()  # noqa: E731 (lambda def)
    torch_harm = lambda t, *a, **kw: 1 / (1 / t).mean(*a, **kw)  # noqa: E731 (lambda def)

    aggrname_map: dict[AggrName, Callable[..., Tensor]] = {
        AggrName.FISHER: fisher_agg,
        AggrName.GEOMETRIC: torch_geom,
        AggrName.HARMONIC: torch_harm,
        AggrName.MAX: torch.amax,
        AggrName.MEAN: torch.mean,
        AggrName.MIN: torch.amin,
        AggrName.PROD: torch.prod,
        AggrName.SUM: torch.sum,
    }

    return aggrname_map[aggr]


def batched_grad(
    outputs: Tensor,
    inputs: Tensor,
    *,
    retain_graph: bool = False,
) -> Tensor:
    """Compute gradients for batched inputs/outputs.

    Arguments
    ---------
    outputs: ``Tensor``
        Shape ``(n_samples,)``.
    inputs: ``Tensor``
        Shape ``(n_samples, *sample_shape)``.
    retain_graph: ``bool``
        Passed to ``torch.autograd.grad``, defaults to ``False``.

    Returns
    -------
    out: ``Tensor``
        Batched gradients, relative to batched inputs. Same shape as
        ``inputs``.

    """
    (grads,) = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=retain_graph,
    )
    return grads


def fgm_direction(grad: Tensor, *, p: float = torch.inf, check: bool = True) -> Tensor:
    r"""Return FGM attack direction.

    When :math:`1 < p < +\infty`, for every sample gradient
    :math:`\nabla`, returns

    .. math::
        \left(\frac{\vert\nabla\vert}{\Vert\nabla\Vert_q}\right)^{q-1}
        \times\text{sgn}(\nabla),

    where :math:`\frac{1}{p}+\frac{1}{q} = 1`. This is a maximizer of
    :math:`\langle\nabla, x\rangle` under the constraint :math:`\Vert
    x\Vert_p=1`. Extends naturally to :math:`p\in\{1, +\infty\}`. When
    :math:`p = 1`, the output is equally distributed amongst extremal
    coordinates of :math:`\nabla`.

    We first presented this approach `here
    <https://github.com/Trusted-AI/adversarial-robustness-toolbox/pull/2382>`_.

    Arguments
    ---------
    grad: ``Tensor``
        See :math:`\nabla` above. Shape ``(n_samples, *sample_shape)``.
        Every sample gradient is treated as a flat vector.
    p: ``float``
        Parameter :math:`p` for the :math:`L^p` norm constraint. Must
        satisfy ``p >= 1``.
    check: ``bool``
        Whether to check that :math:`\Vert`\ ``out``\ :math:`\Vert_p
        \approx 1` sample-wise, up to :math:`10^{-3}` absolute
        tolerance. Defaults to ``True``.

    Returns
    -------
    out: ``Tensor``
        Direction maximizing inner product above, sample-wise. Same
        shape as ``grad``. If the gradient is only zeros, returns zeros.

    Raises
    ------
    :exc:`ValueError`
        If ``grad`` is not finite.
    :exc:`ValueError`
        If not ``p >= 1``.

    """
    if not grad.isfinite().all():
        msg = "Only finite gradients are allowed"
        raise ValueError(msg)

    grad[grad.abs() <= torch.finfo(grad.dtype).eps] *= 0  # Handle subprecision noise
    if p == torch.inf:
        out_abs = torch.ones_like(grad)
    elif p > 1:
        q = p / (p - 1)
        out_abs = normalize_samples(grad.abs(), ord=q) ** (q - 1)
    elif p == 1:
        # Note: out_abs is equally distributed amongst maximizing components
        grad_2d_abs = grad.flatten(1).abs()
        mask = (grad_2d_abs == grad_2d_abs.max(1, keepdim=True)[0]).reshape(grad.shape)
        out_abs = normalize_samples(mask.to(grad), ord=1)
    else:
        msg = f"Parameter p must satisfy `p>=1` (got {p!r})"
        raise ValueError(msg)

    out = out_abs * grad.sign()

    # Check p-norm of output (optional)
    if check:

        def torch_msg(msg: str) -> str:
            """Add error hint."""
            return (
                f"{msg}\n\nThis is likely due to computational approximations. "
                "Consider using p=1 if p is too close to 1"
            )

        norms = out.flatten(1).norm(p, dim=1)
        expected = torch.ones_like(norms).where(norms.to(dtype=bool), 0)
        # Adjust default tolerances for ``torch.half``
        atol, rtol = (1e-3, 3e-3) if grad.dtype is torch.half else (None, None)
        torch.testing.assert_close(norms, expected, atol=atol, rtol=rtol, msg=torch_msg)

    return out


def torch_quantile(  # noqa: PLR0913 (too many arguments)
    tensor: Tensor,
    q: float | Tensor,
    dim: int | None = None,
    *,
    keepdim: bool = False,
    interpolation: str = "nearest",
    out: Tensor | None = None,
) -> Tensor:
    r"""Improved :func:`torch.quantile` for one scalar quantile.

    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    keepdim: ``bool``
        See ``torch.quantile``. Supports only ``False`` currently.
        Defaults to ``False``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.
    out: ``Tensor``, optional
        See ``torch.quantile``. Currently not supported.

    Raises
    ------
    :exc:`ValueError`
        If ``q`` is not in :math:`[0, 1]`.
    :exc:`ValueError`
        If ``interpolation`` value is unsupported.
    :exc:`ValueError`
        If ``out is not None``.

    Notes
    -----
    Uses :func:`torch.kthvalue`. Better than :func:`torch.quantile`
    since:

    #. it has no :math:`2^{24}` tensor `size limit <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes (up to
       :math:`5\times`).

    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Sanitization of: inteporlation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)

    # Sanitization of: out
    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)

    # Logic
    mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode=False)
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    torch.use_deterministic_algorithms(mode=mode)
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    # Rectification of: keepdim
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)


def knn_label_count(  # noqa: PLR0913 (too many arguments)
    index: "Index",
    labels: Tensor,
    n_classes: int,
    k: int,
    query: Tensor,
    *,
    self_query: bool = False,
) -> Tensor:
    """Count labels of neighbors for batched queries.

    Arguments
    ---------
    index: ``Index``
        Search index.
    labels: ``Tensor``
        Labels of reference samples used to build ``index``. Shape
        ``(n_reference,)``.
    n_classes: ``int``
        Number of classes.
    k: ``int``
        Number of neighbors to look up.
    query: ``Tensor``
        The query samples, not necessarily flattened. Shape
        ``(n_query, *sample_shape)``.
    self_query: ``bool``
        See :meth:`Index.search() <scio.scores.utils.Index.search>`.
        Requires one additional reference sample in ``index``. Defaults
        to ``False``.

    Returns
    -------
    counts: ``Tensor``
        Class counts amongst ``k`` nearest neighbors. Shape ``(n_query,
        n_classes)``. Full of ``nan`` if ``k + self_query >
        index.ntotal``.

    Raises
    ------
    :exc:`ValueError`
        If ``labels.shape != (index.ntotal,)``.
    :exc:`ValueError`
        If ``(labels >= n_classes).any()``.

    """
    if labels.shape != (index.ntotal,):
        msg = (
            f"`labels` must have shape (index.ntotal,)={(index.ntotal,)} (got "
            f"{labels.shape})"
        )
        raise ValueError(msg)

    if labels.amax() >= n_classes:
        msg = f"`labels` value out of range at: {(labels >= n_classes).nonzero()}"
        raise ValueError(msg)

    if k + self_query > index.ntotal:
        return torch.full((len(query), n_classes), torch.nan, device=query.device)

    indices = index.search(query, k, self_query=self_query)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return torch.vmap(torch.bincount)(labels[indices], minlength=n_classes)


def ak_lpe(
    index: "Index",
    k: int,
    query: Tensor,
    *,
    self_query: bool = False,
) -> Tensor:
    r"""Mean distance from :math:`\approx k`-th nearest neighbors.

    Arguments
    ---------
    index: ``Index``
        Search index.
    k: ``int``
        Number of neighbors to look up.
    query: ``Tensor``
        The query samples, not necessarily flattened. Shape
        ``(n_query, *sample_shape)``.
    self_query: ``bool``
        See :meth:`Index.search() <scio.scores.utils.Index.search>`. Requires one
        additional reference sample in ``index``. Defaults to ``False``.

    Returns
    -------
    out: ``Tensor``
        Mean distance from ``m``-th nearest neighbors, with ``k / 2 < m
        <= 3 * k // 2``. Shape ``(n_query,)``. Full of ``nan`` if
        ``index.ntotal < 3 * k // 2 + self_query``.

    Attention
    ---------
    If ``index`` uses the :math:`L^2` metric, the "mean distance"
    actually ressembles a **variance**, since :meth:`Index.search()
    <scio.scores.utils.Index.search>` outputs the **squared** euclidian
    distance in that case.

    """
    K = 3 * k // 2  # noqa: N806 (uppercase K)
    if index.ntotal < K + self_query:
        shape = (len(query),)
        return torch.full(shape, torch.nan, dtype=query.dtype, device=query.device)

    D = index.search(query, K, self_query=self_query)[0]  # noqa: N806 (uppercase D)
    return D[:, -k:].mean(1)


def kldiv(inputs: Tensor, expected: Tensor) -> Tensor:
    r"""KL div for (potentially batched) :math:`1`\ D inputs.

    Computes :math:`D_{\text{KL}}(\text{inputs}\Vert\text{expected})`
    where inputs may be batched. Input samples and ``expected`` should
    be vectors of same length, in the probability space (up to
    rescaling).

    It is essentially a wrapper for intuitive use of
    :func:`torch.nn.functional.kl_div`, which should be preferred in
    case of numerical instability, as it can operate in
    :math:`\text{log}` space.

    Arguments
    ---------
    inputs: ``Tensor``
        Batched samples, in probability space (up to rescaling). Shape
        ``(*batch_shape, space_size)``.
    expected: ``Tensor``
        Expectation in probability space (up to rescaling). Shape
        ``(space_size,)``.

    Returns
    -------
    div: ``Tensor``
        The sample-wise divergence. Shape ``batch_shape``. If
        ``expected`` is invalid (*e.g.* contains ``nan``), returns all
        ``nan``. The same is true individually for each input sample.
        The returned ``div.dtype`` is ``torch.result_type(inputs,
        expected)`` if at least one of ``inputs`` of ``expected`` is of
        floating type, ``torch.float`` otherwise.

    Raises
    ------
    :exc:`ValueError`
        If ``expected`` is a scalar.

    Example
    -------
    ::

        >>> inputs = torch.tensor([
        ...     [0, 10, 20],
        ...     [0, 2, 2],
        ...     [0, 0.5, 0.5],
        ...     [0, 1, 0],
        ...     [1, 1, 2],
        ...     [0, -1, 2],
        ...     [0, 1, torch.nan],
        ... ])
        >>> expected = torch.tensor([0, 1, 2])
        >>> kldiv(inputs, expected)
        tensor([0.0000, 0.0589, 0.0589, 1.0986,    inf,    nan,    nan])

    """
    if expected.ndim == 0:
        msg = "`expected` must be at least a 1D tensor (got scalar)"
        raise ValueError(msg)
    expected_prob = expected / expected.sum()
    inputs_prob = inputs / inputs.sum(-1, keepdim=True)

    # Handle dtype
    found_float = torch.is_floating_point(inputs) or torch.is_floating_point(expected)
    dtype = torch.result_type(inputs, expected) if found_float else torch.float
    expected_prob = expected_prob.to(dtype=dtype)
    inputs_prob = inputs_prob.to(dtype=dtype)

    if torch.logical_or(expected_prob.isnan(), expected_prob < 0).any():
        return torch.full_like(inputs_prob[..., 0], torch.nan)

    div_elts = kl_div(expected_prob.log(), inputs_prob, reduction="none")
    div = div_elts.nan_to_num(posinf=torch.inf).sum(-1)  # p*log(p/q) w/ p=q=0 -> 0

    # Incorrect inputs
    mask = torch.logical_or(inputs_prob.isnan(), inputs_prob < 0).any(-1)
    div[mask] = torch.nan

    return div


def dirmult_surprise(counts: Tensor, alpha: Tensor) -> Tensor:
    r"""Compute DCM surprise.

    Entropic surprise for the `Dirichlet Compound Multinomial
    <https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution>`_
    distribution :math:`\mathrm{DirMult}(n, \alpha)`, where :math:`n` is
    implicitly defined by ``counts`` and the prior :math:`\alpha` is
    defined by ``alpha``.

    Arguments
    ---------
    counts: ``Tensor``
        Integers counts that sum up to the same :math:`n\geqslant 1` for
        every sample. Shape ``(*batch_shape, k)``.
    alpha: ``Tensor``
        Dirichlet prior, interpreted as pseudocount. Shape ``(k,)``.

    Returns
    -------
    res: ``Tensor``
        Shape ``batch_shape``. The returned ``div.dtype`` is
        ``torch.result_type(counts, alpha)`` if at least one of
        ``counts`` of ``alpha`` is of floating type, ``torch.float``
        otherwise.

    Raises
    ------
    :exc:`ValueError`
        If ``counts`` is a scalar.
    :exc:`ValueError`
        If ``counts`` has inconsistent sums along the last axis.

    """
    if counts.ndim == 0:
        msg = "`counts` must be at least a 1D tensor (got scalar)"
        raise ValueError(msg)

    ns = counts.sum(-1)
    if (ns != (n := ns.flatten()[0])).any():
        msg = "`counts` must have constant sum along last axis"
        raise ValueError(msg)

    a = alpha.sum()
    gln = torch.special.gammaln
    constant = gln(n + a) - gln(n + 1) - gln(a) + gln(alpha).sum()
    res = constant + gln(counts + 1).sum(-1) - gln(counts + alpha).sum(-1)

    # Handle dtype
    found_float = torch.is_floating_point(counts) or torch.is_floating_point(alpha)
    dtype = torch.result_type(counts, alpha) if found_float else torch.float
    return res.to(dtype=dtype)


def multinomial_test(
    counts: Tensor,
    prior: Tensor,
    mode: MultinomialTestModeLike,
) -> Callable[[Tensor], Tensor]:
    """Prepare test after having observed counts, with prior belief.

    Arguments
    ---------
    counts: ``Tensor``
        Observed counts. Shape ``(*batch_shape, k)``
    prior: ``Tensor``
        Prior pseudo-count. Set to ones for no *a priori*. Shape
        ``(k,)``.
    mode: ``MultinomialTestModeLike``
        See :class:`~scio.utils.MultinomialTestMode`.

    Returns
    -------
    test: ``Callable[[Tensor], Tensor]``
        Deviation test function ``test(observed: Tensor) -> Tensor``,
        where ``observed`` can be batched. Depending on ``mode``, this
        function defines the ``test`` "expectations" by aggregating the
        observed ``counts`` and the ``prior``.

    Raises
    ------
    :exc:`ValueError`
        If ``counts`` is a scalar.
    :exc:`ValueError`
        If ``mode`` value is unsupported.

    Note
    ----
    In principle, if ``counts`` indeed represents counting results, then
    ``counts.sum(-1)`` should hold the same value for every sample. This
    is not checked at runtime. The same applies to ``observed``.

    """
    mode = MultinomialTestMode(mode)
    if counts.ndim == 0:
        msg = "`counts` must be at least a 1D tensor (got scalar)"
        raise ValueError(msg)

    counts_sum = torch.atleast_2d(counts).flatten(end_dim=-2).sum(0)

    if mode == MultinomialTestMode.MAP:
        return partial(kldiv, expected=counts_sum + prior - 1)
    if mode == MultinomialTestMode.MLE:
        return partial(kldiv, expected=counts_sum)
    if mode == MultinomialTestMode.DCM:
        return partial(dirmult_surprise, alpha=counts_sum + prior)

    msg = f"Unsupported mode: {mode!r}"
    raise ValueError(msg)
