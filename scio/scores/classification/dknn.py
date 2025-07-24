"""DkNN for classification."""

__all__ = ["DkNN"]

import warnings
from itertools import zip_longest

import torch
from torch import Tensor

from scio.scores.utils import get_aggregator, make_indexes
from scio.utils import AggrNameLike, IndexMetricLike, check

from .base import BaseScoreClassif


class DkNN(BaseScoreClassif):
    """DkNN for classification.

    Vanilla ``DkNN`` is equivalent to nearest neighbors class counting.

    Arguments
    ---------
    k: ``int``
        Number of nearest neighbors to consider.
    per_class: ``bool``
        If ``True``, the quantiles are computed relatively to
        calibration scores from the candidate class. Defaults to
        ``False``.
    per_layer: ``bool``
        If ``True``, scores are separated for each layer, then
        aggregated at the end, using ``aggregation`` scheme. Defaults to
        ``False``.
    aggregation: ``AggrNameLike | float``
        See ``per_layer`` and :class:`~scio.utils.AggrName`. Defaults to
        ``"mean"``.
    index_metric: ``IndexMetricLike``
        Kind of metric to use for nearest neighbors search. See
        :class:`~scio.utils.IndexMetric`. Defaults to ``"l2"``.

    References
    ----------
    .. bibliography::
       :filter: false

       DkNN

    """

    k: int
    per_class: bool = False  # [DkNN]
    per_layer: bool = False  # [DkNN]
    aggregation: AggrNameLike | float = "mean"
    index_metric: IndexMetricLike = "l2"  # [DkNN]

    def _check_params(self, n_calib: int) -> None:
        super()._check_params(n_calib)
        check(0 < self.k <= n_calib)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data."""
        self.calib_labels = calib_labels
        n_samples, n_classes = self.rnet(calib_data).shape  # Records activations
        all_activations = self.activations()
        self.indexes = make_indexes(all_activations, metric=self.index_metric)

        if not (self.per_layer or self.per_class):
            return

        n_layers = len(all_activations) if self.per_layer else 1
        self.calib_knn = torch.zeros(
            (n_layers, n_samples),
            dtype=calib_data.dtype,
            device=calib_data.device,
        )
        for layer, (activations, index) in enumerate(
            zip(all_activations, self.indexes, strict=False),
        ):
            # Search and count knn labels
            indices = index.search(activations, self.k, self_query=True)[1]
            neighbors_labels = calib_labels[indices]
            layer_calib_knn = (neighbors_labels == calib_labels[:, None]).sum(1)
            self.calib_knn[layer if self.per_layer else 0] += layer_calib_knn

        # If per_class, store the labels for sorted knn counts
        # Both ``calib_knn`` and ``calib_labels_sorted`` shapes are
        # ``(n_layers if per_layer else 1, n_samples)``
        if self.per_class:
            sorter = self.calib_knn.argsort(1)
            self.calib_labels_sorted = calib_labels[sorter]
            self.calib_knn = self.calib_knn.gather(1, sorter)
        elif self.per_layer:
            self.calib_labels_sorted = torch.tensor([])
            self.calib_knn = self.calib_knn.sort(1).values

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs)
        all_activations = self.activations()
        n_samples, n_classes = out.shape
        n_layers = len(all_activations) if self.per_layer else 1
        test_knn = torch.zeros(
            (n_layers, n_samples, n_classes),
            dtype=out.dtype,
            device=out.device,
        )

        for layer, (index, activations) in enumerate(
            zip(self.indexes, all_activations, strict=False),
        ):
            # Search knn labels
            indices = index.search(activations, self.k)[1]
            neighbors_labels = self.calib_labels[indices]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                test_knn[layer if self.per_layer else 0] += torch.vmap(torch.bincount)(
                    neighbors_labels,
                    minlength=n_classes,
                )

        # Compute conformity
        if not (self.per_class or self.per_layer):
            conformity = test_knn[0]
            return out, conformity

        quantile_args = zip_longest(self.calib_knn, test_knn, self.calib_labels_sorted)
        all_conformities = torch.stack([
            self.quantiles(*quantile_args_layer)
            for quantile_args_layer in quantile_args
        ])

        # Aggregate
        conformity = get_aggregator(self.aggregation)(all_conformities, dim=0)
        return out, conformity

    @staticmethod
    def quantiles(
        calib_scores: Tensor,
        test_scores: Tensor,
        calib_labels: Tensor | None = None,
    ) -> Tensor:
        """Left quantile of ``test_scores``, potentially class-wise.

        Arguments
        ---------
        calib_scores: ``Tensor``
            **MUST BE SORTED**. Population to compute quantiles against.
            Shape ``(n_calib,)``.
        test_scores: ``Tensor``
            Query scores for quantile computation. There is one score
            per sample and per candidate class. Shape ``(n_test,
            n_classes)``.
        calib_labels: ``Tensor``, optional
            If provided, labels of calibration samples, in which case
            the quantiles are computed class-wise. Shape ``(n_calib,)``.

        Returns
        -------
        out: ``Tensor``
            Left quantile of ``test_scores`` against ``calib_scores``.
            If ``calib_labels`` was provided, computed class-wise. Shape
            ``(n_test, n_classes)``. When ``nan``, returned ``0``.

        """
        # Not class-wise
        if calib_labels is None:
            positions = torch.searchsorted(calib_scores, test_scores)
            return positions / len(calib_scores)

        # Class-wise
        out = torch.empty_like(test_scores)
        for label, test_scores_label in enumerate(test_scores.T):
            calib_scores_label = calib_scores[calib_labels == label]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                positions = torch.searchsorted(calib_scores_label, test_scores_label)
            out[:, label] = positions / len(calib_scores_label)

        return out.nan_to_num()
