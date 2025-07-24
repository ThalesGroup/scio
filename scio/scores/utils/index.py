"""Utilities directly related to index creation."""

__all__ = ["Index", "make_indexes"]

from functools import reduce
from itertools import batched, product, starmap
from typing import TYPE_CHECKING, cast

import faiss  # type: ignore[import-untyped]
import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable

from scio.utils import IndexMetric, IndexMetricLike


def IndexType(metric_like: IndexMetricLike) -> type[faiss.Index]:  # noqa: N802 (uppercase IndexType)
    """Select appropriate index constructor."""
    metric = IndexMetric(metric_like)

    if metric == IndexMetric.L2:
        return faiss.IndexFlatL2
    if metric == IndexMetric.IP:
        return faiss.IndexFlatIP

    msg = f"Unsupported metric: {metric_like!r}"
    raise NotImplementedError(msg)


class Index:
    r"""Practical ``faiss`` index wrapper.

    Eases the use of nearest neighbors search library ``faiss``.

    Arguments
    ---------
    dim: ``int``
        Dimensionality of the data space.
    metric: ``IndexMetricLike``
        See :class:`~scio.utils.IndexMetric`.

    Notes
    -----
    The current implementation uses ``faiss-cpu``, which may involve a
    ``GPU > CPU > GPU`` bottleneck.

    To avoid a ``dtype`` conversion bottleneck, use ``float32`` data.

    Current implementation only supports flat indexes.

    """

    def __init__(self, *, dim: int, metric: IndexMetricLike) -> None:
        """Construct an :class:`Index` instance."""
        self._dim = dim
        self._metric = metric
        self.faiss = IndexType(metric)(dim)

    def add(self, samples: Tensor) -> None:
        """Add reference samples to the index."""
        samples_faiss = samples.flatten(1).numpy(force=True)
        self.faiss.add(samples_faiss)

    def remove_ids(self, ids: Tensor) -> int:
        """Remove selected reference samples from the index.

        Arguments
        ---------
        ids: ``Tensor``
            Indices to remove from reference population.

        """
        ids_faiss = ids.numpy(force=True).flatten()
        return self.faiss.remove_ids(ids_faiss)

    def search(
        self,
        samples: Tensor,
        k: int,
        *,
        self_query: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Run nearest neighbors search.

        Arguments
        ---------
        samples: ``Tensor``
            The query samples.
        k: ``int``
            Number of neighbors to look up.
        self_query: ``bool``
            Set to ``True`` if ``index`` was built using exactly
            ``samples``, unshuffled. In this case, if a sample is itself
            amongst its closest neighbors (it is not always the case,
            *e.g.* inner product without prior normalization), it is
            excluded from the result. Requires looking up ``k + 1``
            neighbors. Defaults to ``False``.

        Returns
        -------
        D, I: ``tuple[Tensor, Tensor]``
            Result from :meth:`faiss.Index.search` with postprocessing,
            including ``torch`` conversion back to ``samples``' device
            (and dtype for ``D``), with self removal if required.

        Note
        ----
        ``D`` is the **squared** *distance* for :math:`L^2` ``metric``,
        and (decreasing) *similarity* for inner product ``metric``.

        """
        device = samples.device
        n_samples = len(samples)
        samples_faiss = samples.flatten(1).numpy(force=True)
        D, I = self.faiss.search(samples_faiss, k + self_query)  # noqa: E741, N806 (uppercase D, I; ambiguous I)
        D_out = torch.as_tensor(D, dtype=samples.dtype, device=device)  # noqa: N806 (uppercase D)
        I_out = torch.as_tensor(I, device=device)  # noqa: N806 (uppercase I)

        # Remove self if present
        if self_query:
            mask = I_out != torch.arange(n_samples, device=device)[:, None]
            mask[:, k] = ~mask[:, :k].all(1)
            D_out = D_out[mask].reshape(n_samples, k)  # noqa: N806 (uppercase D)
            I_out = I_out[mask].reshape(n_samples, k)  # noqa: N806 (uppercase I)

        return D_out, I_out

    @property
    def dim(self) -> int:
        """Dimensionality of the data space."""
        return self._dim

    @property
    def metric(self) -> IndexMetricLike:
        """See :class:`~scio.utils.IndexMetric`."""
        return self._metric

    @property
    def ntotal(self) -> int:
        """Number of samples in reference population."""
        return self.faiss.ntotal

    @property
    def D_is_similarity(self) -> bool:  # noqa: N802 (uppercase D)
        """Whether ``D`` is a similarity measure."""
        return IndexMetric(self.metric) == IndexMetric.IP


type NestedIndexTupleOptionallySqueezed = "NestedIndexTupleOptionallySqueezed"


# Logic
def make_indexes(
    all_samples: tuple[Tensor, ...] | Tensor,
    all_groups: tuple[Tensor, ...] | Tensor | None = None,
    *,
    n_groups: int | None = None,
    metric: IndexMetricLike,
    squeeze: bool = True,
) -> NestedIndexTupleOptionallySqueezed:
    """Prepare multiple search indexes.

    For now only :class:`faiss.IndexFlatL2` or
    :class:`faiss.IndexFlatIP`. Note that distances returned when
    querying these ``faiss`` indexes are respectively the **squared**
    euclidian distance and the inner product -- which is a
    **similarity** measure, not a distance. In the latter case, the
    returned neighbors are still ordered from the "closest" (most
    similar) to the furthest.

    Arguments
    ---------
    all_samples: ``tuple[Tensor, ...] | Tensor``
        If a ``Tensor``, treated as ``(all_samples,)``. Tuple of tensors
        of shape ``(n_samples, *sample_shape)`` of common length
        ``n_samples``. The samples from which to build search indexes,
        treated as vector samples.
    all_groups: ``tuple[Tensor, ...] | Tensor``, optional
        If not provided, treated as ``torch.zeros(n_samples)``. If a
        ``Tensor``, treated as ``(all_groups,)``. Tuple of tensors of
        shape ``(n_samples,)``. The group every sample belongs to.
        Values must be nonnegative integers.
    n_groups: ``int``, optional
        If ``all_groups`` is not provided, treated as ``1``. Else it is
        required to be an ``int``. Group values in ``range(n_groups)``
        are considered. Empty indexes are possible (*e.g.* when there
        are no samples for a given group value).
    metric: ``IndexMetricLike``
        See :class:`~scio.utils.IndexMetric`.
    squeeze: ``bool``
        Whether to apply the postprocessing squeezing steps described
        in ``indexes``. Defaults to ``True``.

    Returns
    -------
    indexes: Nested tuples of :class:`Index`, optionally squeezed
        Every search index. When ``all_groups`` and ``all_samples`` are
        tuples, there are ``len(all_groups) * len(all_samples) *
        n_groups`` indexes. From outermost to innermost, nested tuples
        are respectively along ``all_groups``, ``all_samples`` and
        group values. Finally, the following is applied if ``squeeze``.
        If ``all_groups`` or ``all_samples`` are ``Tensor``, their
        respective tuple is squeezed. Furthermore, if ``all_groups`` was
        not provided the tuple corresponding to group value is squeezed.

    Hint
    ----
    Think of ``all_samples`` as layers activations stacked in a tuple
    along layers for example.

    Think of ``all_groups`` as ``(true_labels, pred_labels)`` for
    example.

    Raises
    ------
    :exc:`ValueError`
        If ``n_groups`` is not an ``int``, despite ``all_groups`` being
        ``None``.

    Note
    ----
    For accurate output type specification, please refer to the
    associated source stub.

    """
    # Args preprocessing
    if isinstance(all_samples, tuple):
        expanded_samples = True
    else:
        expanded_samples = False
        all_samples = (all_samples,)

    if isinstance(all_groups, tuple):
        exapanded_groups = expanded_group_values = True
    elif all_groups is None:
        exapanded_groups = expanded_group_values = False
        samples = all_samples[0]
        all_groups = (
            torch.zeros(len(samples), dtype=torch.int, device=samples.device),
        )
        n_groups = 1
    else:  # isinstance(all_groups, Tensor)
        exapanded_groups, expanded_group_values = False, True
        all_groups = (all_groups,)

    if not isinstance(n_groups, int):
        msg = (
            "Since `all_groups is not None`, `n_groups` must be a positive integer "
            f"(got {n_groups})"
        )
        raise ValueError(msg)  # noqa: TRY004 (prefer TypeError)

    if not squeeze:
        expanded_samples = exapanded_groups = expanded_group_values = True

    # Single index
    def make_index(groups: Tensor, samples: Tensor, current_group: int) -> Index:
        """Create one single index."""
        index = Index(dim=samples.shape[1:].numel(), metric=metric)
        index.add(samples[groups == current_group])
        return index

    # Nested logic
    indexes_map = starmap(make_index, product(all_groups, all_samples, range(n_groups)))
    final_shape = (
        (len(all_groups),) * exapanded_groups
        + (len(all_samples),) * expanded_samples
        + (n_groups,) * expanded_group_values
    )

    # Mypy monkeypatch
    # (github.com/python/cpython/issues/129730#issue-2835898168)
    cast("Callable[[Iterable[Index], int], Iterable[Index]]", batched)

    return next(reduce(batched, final_shape[::-1], indexes_map))
