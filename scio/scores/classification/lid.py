"""LID for classification."""

__all__ = ["LID"]

import torch
from torch import Tensor

from scio.scores.utils import make_indexes
from scio.utils import check

from .base import BaseScoreClassif


class LID(BaseScoreClassif):
    """LID for classification.

    Arguments
    ---------
    k: ``int``
        Number of nearest neighbors used for LID estimation.

    References
    ----------
    .. bibliography::
       :filter: false

       LID

    """

    k: int

    def _check_params(self, n_calib: int) -> None:
        super()._check_params(n_calib)
        check(0 < self.k <= n_calib)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, _calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data."""
        self.rnet(calib_data)  # Records activations
        (activations,) = self.activations(concatenate=True)
        self.index = make_indexes(activations, metric="l2")

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs)  # Records activations
        (activations,) = self.activations(concatenate=True)
        D2 = self.index.search(activations, self.k)[0]  # noqa: N806 (uppercase D2)
        conformity = -2 * self.compute_lid(D2)  # Multiply by 2 since squared distances
        return out, conformity

    def compute_lid(self, sorted_distances: Tensor) -> Tensor:
        """Compute LID estimator for batch.

        Arguments
        ---------
        sorted_distances: ``Tensor``
            Shape ``(n_samples, self.k)``, sorted along ``dim=1``.
            Distances to nearest neighbors.

        Returns
        -------
        LID: ``float``
            Shape ``(n_samples,)``. Estimator for LID from :cite:`LID`.

        """
        ratios = sorted_distances[:, [-1]] / sorted_distances
        sample_in_pop = sorted_distances[:, 0] == 0
        LID_estimator = self.k / ratios.log().sum(1)  # noqa: N806 (uppercase)
        return torch.where(sample_in_pop, 0, LID_estimator).nan_to_num(nan=torch.inf)
