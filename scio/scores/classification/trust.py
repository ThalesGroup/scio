"""Trust for classification."""

__all__ = ["Trust"]

import torch
from torch import Tensor

from scio.scores.utils import Index, make_indexes
from scio.utils import check

from .base import BaseScoreClassif


class Trust(BaseScoreClassif):
    """Trust for classification.

    The score is build by comparing the distances of an input
    representation to those of calibration samples grouped by class,
    after having excluded class-wise outliers.

    Arguments
    ---------
    alpha: ``float``
        Target (and minimum) proportion of outliers to exclude when
        building the alpha-high-density-sets.
    k: ``int``
        The ``k``-th nearest neighbors distance is used to build the
        alpha-high-density-sets. Defaults to ``10``.

    References
    ----------
    .. bibliography::
       :filter: false

       Trust

    """

    alpha: float
    k: int = 10  # [Trust] default

    def _check_params(self, n_calib: int) -> None:
        super()._check_params(n_calib)
        check(0 <= self.alpha < 1)
        check(0 < self.k <= n_calib)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data.

        Prepare per-class alpha-high-density-sets indexes.
        """
        n_classes = self.rnet(calib_data).shape[1]  # Records activations
        (activations,) = self.activations(concatenate=True)

        self.indexes = []
        for c in range(n_classes):
            population = activations[calib_labels == c]
            self.indexes.append(self.alpha_high_density_index(population))

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs)
        (activations,) = self.activations(concatenate=True)
        infinite_distance = torch.full((len(activations), 1), torch.inf).to(out)
        all_D2 = [  # noqa: N806 (uppercase all_D2)
            index.search(activations, 1)[0] if index is not None else infinite_distance
            for index in self.indexes
        ]
        D_to_density_sets = torch.cat(all_D2, dim=1) ** 0.5  # noqa: N806 (uppercase D_to_density_sets)
        conformity = self.trust_score(D_to_density_sets)

        return out, conformity

    def alpha_high_density_index(self, population: Tensor) -> Index | None:
        r"""Build index for alpha-high-density-set of population.

        It is built by excluding a proportion :math:`\alpha` (rounded up
        to the nearest fraction possible) of outliers, for :math:`k`\
        -th
        NN distance.

        Note
        ----
        Because of a ``faiss`` `corner case
        <https://github.com/facebookresearch/faiss/issues/3830>`_, we
        return ``None`` to manually handle the case of empty indexes.

        """
        n_samples = len(population)
        if n_samples == 0:
            return None

        index = make_indexes(population, metric="l2")
        if self.alpha == 0:
            return index

        # Remove proportion alpha (rounded down)
        Dk = index.search(population, self.k, self_query=True)[0][:, -1]  # noqa: N806 (uppercase Dk)
        n_remove = int(self.alpha * n_samples)
        to_remove = torch.topk(Dk, n_remove).indices
        index.remove_ids(to_remove)

        return index

    @staticmethod
    def trust_score(distances: Tensor) -> Tensor:
        """Compute trust score :cite:`Trust{algorithm 2}`.

        Arguments
        ---------
        distances: ``Tensor``
            For the batch, distances to every classes'
            alpha-high-density-set. Shape ``(n_samples, n_classes)``.

        Returns
        -------
        inv_preference: ``Tensor``
            Inverse of multiplicative preference, that is the ratio
            between the the highest other value and the value itself.
            Same shape as distances.

        """
        min1_idx, min2_idx = distances.topk(2, dim=1, largest=False).indices.T
        range_ = range(len(distances))
        preferences = distances / distances[range_, min1_idx, None]
        preferences[range_, min1_idx] = 1 / preferences[range_, min2_idx]
        return (1 / preferences).nan_to_num(nan=1)
