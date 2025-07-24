"""KNN for classification."""

__all__ = ["KNN"]

import torch
from torch import Tensor

from scio.scores.utils import make_indexes
from scio.utils import IndexMetricLike, check

from .base import BaseScoreClassif


class KNN(BaseScoreClassif):
    r"""KNN for classification.

    Nonconformity is the distance to :math:`k`\ -th neighbor in the
    chosen latent space, after activations normalization. Note that it
    is independant of the predicted class.

    Arguments
    ---------
    k: ``int``
        Number of nearest neighbors to consider.
    index_metric: ``IndexMetricLike``
        Kind of metric to use for nearest neighbors search. See
        :class:`~scio.utils.IndexMetric`.

    Note
    ----
    In :cite:`KNN`, authors operate solely on the penultimate layer
    activations. Here, we stack all recorded activations, which is
    equivalent if and only if the network only records the penultimate
    layer.

    References
    ----------
    .. bibliography::
       :filter: false

       KNN

    """

    k: int
    act_norm: float | None = 2.0  # [KNN] default
    index_metric: IndexMetricLike = "l2"  # [KNN] default

    def _check_params(self, n_calib: int) -> None:
        super()._check_params(n_calib)
        check(0 < self.k <= n_calib)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, _calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data."""
        self.rnet(calib_data)  # Records activations
        (activations,) = self.activations(concatenate=True)
        self.index = make_indexes(activations, metric=self.index_metric)

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs)  # Records activations
        (activations,) = self.activations(concatenate=True)
        Dk = self.index.search(activations, self.k)[0][:, -1]  # noqa: N806 (uppercase Dk)
        conformity = Dk if self.index.D_is_similarity else -Dk
        return out, conformity
