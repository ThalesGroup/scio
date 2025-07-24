"""ROC utility class for Pareto front, convex hull, thresholds, ..."""

# ruff: noqa: N802 (uppercase methods)
# ruff: noqa: N806 (uppercase variables)

__all__ = ["ROC"]

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from scio.utils import check


class ROC:
    """ROC utility for Discriminative Power and visualization.

    We recall that a :ref:`Discriminative Power <discriminative_power>`
    only depends on the Pareto front of all the :math:`(FP, TP)` tuples
    when thresholding with every possible threshold. Per convention:

    #. The thresholding test is ``score <= tau``;
    #. **Positive** (*i.e.* OoD) samples should verify this and thus
       have a **low score**;
    #. Scores must not be ``nan`` or ``-inf`` (ensuring validity of the
       first note in :attr:`pareto`);

    Arguments
    ---------
    labels: ``ArrayLike``
        The label of samples, interpreted as ``bool``. Shape
        ``(n_samples,)``.
    scores: ``ArrayLike``
        The score of samples. Shape ``(n_samples,)``.

    """

    def __init__(self, labels: ArrayLike, scores: ArrayLike) -> None:
        """Compute pareto front and convex hull for ROC."""
        self._preprocess(labels, scores)
        self._compute_front()
        self._compute_convex_hull()

    def _preprocess(self, labels: ArrayLike, scores: ArrayLike) -> None:
        """Cast, run basic checks, sort."""
        scores_np = np.asarray(scores, dtype=float)
        labels_np = np.asarray(labels, dtype=bool)

        check(labels_np.any())
        check(not labels_np.all())
        check(not np.isnan(scores).any())
        check(-np.inf < scores_np.min())

        sorter = np.argsort(scores)
        self._scores = scores_np[sorter]
        self._labels = labels_np[sorter]

    def _compute_front(self) -> None:
        """Compute Pareto front from stored sorted scores.

        Store attributes :attr:`_N`, :attr:`_P`, :attr:`_pareto` and
        :attr:`_thresholds`.
        """
        inf = np.inf
        scores = self._scores
        # ``unique_mask``: keep highest index when equal scores. Handles
        # ``inf`` (considered self equal). Rests on ``scores`` being
        # sorted. Faster than ``np.unique`` which keeps first occurrence
        unique_mask = scores != np.r_[scores[1:], np.nan]
        PP = np.where(unique_mask)[0]
        TP = self._labels.cumsum()[PP]
        FP = PP - TP + 1
        attainable_fptp = np.insert(np.c_[FP, TP], 0, 0, 0)
        pareto_mask = (np.diff(attainable_fptp[:, 0], append=inf) > 0) & (
            np.diff(attainable_fptp[:, 1], prepend=-inf) > 0
        )
        self._pareto = attainable_fptp[pareto_mask]
        pareto_idx = np.where(pareto_mask)[0]
        self._thresholds = np.insert(scores[unique_mask], [0, len(PP)], [-inf, inf])[
            [pareto_idx, pareto_idx + 1]
        ].T
        self._N, self._P = int(FP[-1]), int(TP[-1])

    def _compute_convex_hull(self) -> None:
        """Compute convex hull from stored Pareto front.

        Store attributes :attr:`_pareto_ch` and :attr:`_thresholds_ch`.
        """
        keep: list[int] = []
        for i, new in enumerate(self.pareto):
            while len(keep) > 1 and np.linalg.det(new - self.pareto[keep[-2:]]) >= 0:
                keep.pop()
            keep.append(i)

        self._pareto_ch = self.pareto[keep]
        self._thresholds_ch = self.thresholds[keep]

    @property
    def N(self) -> int:
        """Total number of negative samples."""
        return self._N

    @property
    def P(self) -> int:
        """Total number of positive samples."""
        return self._P

    @property
    def pareto(self) -> NDArray[np.integer]:
        """Ordered :math:`(FP, TP)` tuples defining the Pareto front.

        Returns
        -------
        pareto: ``NDArray[np.integer]``
            Shape ``(n_points_pareto, 2)``.

        Note
        ----
        The following are always true:

        - ``self.pareto[0, 0] == 0`` since ``-inf`` scores are
          prohibited;
        - ``self.pareto[-1, 1] == self.P``.

        """
        return self._pareto

    @property
    def thresholds(self) -> NDArray[np.floating]:
        """The threshold intervals associated with Pareto points.

        Returns
        -------
        thresholds: ``NDArray[np.floating]``
            Convention: lower bound is included, higher bound is
            excluded (unless ``inf``). Shape ``(n_points_pareto, 2)``.

        """
        return self._thresholds

    @property
    def pareto_ch(self) -> NDArray[np.integer]:
        """Same as :attr:`pareto`, reduced to convex hull front."""
        return self._pareto_ch

    @property
    def thresholds_ch(self) -> NDArray[np.floating]:
        """Same as :attr:`thresholds`, reduced to convex hull front."""
        return self._thresholds_ch

    @property
    def tot(self) -> int:
        """Total population size."""
        return self.P + self.N

    @property
    def FP(self) -> NDArray[np.integer]:
        """False Positive points.

        Returns
        -------
        FP: ``NDArray[np.integer]``
            Shape ``(n_points_pareto,)``.

        """
        return self.pareto[:, 0]

    @property
    def FPch(self) -> NDArray[np.integer]:
        """Same as :attr:`FP`, reduced to convex hull front."""
        return self.pareto_ch[:, 0]

    @property
    def TP(self) -> NDArray[np.integer]:
        """True Positive points.

        Returns
        -------
        TP: ``NDArray[np.integer]``
            Shape ``(n_points_pareto,)``.

        """
        return self.pareto[:, 1]

    @property
    def TPch(self) -> NDArray[np.integer]:
        """Same as :attr:`TP`, reduced to convex hull front."""
        return self.pareto_ch[:, 1]

    @property
    def FN(self) -> NDArray[np.integer]:
        """False Negative points.

        Returns
        -------
        FN: ``NDArray[np.integer]``
            Shape ``(n_points_pareto,)``.

        """
        return self.P - self.TP

    @property
    def FNch(self) -> NDArray[np.integer]:
        """Same as :attr:`FN`, reduced to convex hull front."""
        return self.P - self.TPch

    @property
    def TN(self) -> NDArray[np.integer]:
        """True Negative points.

        Returns
        -------
        TN: ``NDArray[np.integer]``
            Shape ``(n_points_pareto,)``.

        """
        return self.N - self.FP

    @property
    def TNch(self) -> NDArray[np.integer]:
        """Same as :attr:`TN`, reduced to convex hull front."""
        return self.N - self.FPch

    @property
    def FPR(self) -> NDArray[np.floating]:
        """False Positive Rate points.

        Returns
        -------
        FPR: ``NDArray[np.floating]``
            Shape ``(n_points_pareto,)``.

        """
        return self.FP / self.N

    @property
    def FPRch(self) -> NDArray[np.floating]:
        """Same as :attr:`FPR`, reduced to convex hull front."""
        return self.FPch / self.N

    @property
    def TPR(self) -> NDArray[np.floating]:
        """True Positive Rate points.

        Returns
        -------
        TPR: ``NDArray[np.floating]``
            Shape ``(n_points_pareto,)``.

        """
        return self.TP / self.P

    @property
    def TPRch(self) -> NDArray[np.floating]:
        """Same as :attr:`TPR`, reduced to convex hull front."""
        return self.TPch / self.P

    @property
    def FNR(self) -> NDArray[np.floating]:
        """False Negative Rate points.

        Returns
        -------
        FNR: ``NDArray[np.floating]``
            Shape ``(n_points_pareto,)``.

        """
        return self.FN / self.P

    @property
    def FNRch(self) -> NDArray[np.floating]:
        """Same as :attr:`FNR`, reduced to convex hull front."""
        return self.FNch / self.P

    @property
    def TNR(self) -> NDArray[np.floating]:
        """True Negative Rate points.

        Returns
        -------
        TNR: ``NDArray[np.floating]``
            Shape ``(n_points_pareto,)``.

        """
        return self.TN / self.N

    @property
    def TNRch(self) -> NDArray[np.floating]:
        """Same as :attr:`TNR`, reduced to convex hull front."""
        return self.TNch / self.N

    def plot(self, *, legend: bool = False, ax: plt.Axes | None = None) -> None:
        """Plot the ROC curve.

        Arguments
        ---------
        legend: ``bool``
            Whether to display the legend. Defaults to ``False``.
        ax: ``plt.Axes``, optional
            If provided, where the curve is plotted.

        """
        pareto_ratio = self.pareto / [self.N, self.P]
        pareto_ch_ratio = self.pareto_ch / [self.N, self.P]

        if ax is None:
            ax = plt.gca()

        # Layout
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("ROC curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        # Dummy
        ax.plot([0, 1], [0, 1], "--", color="black", lw=0.5)

        # Pareto
        ax.scatter(
            *pareto_ratio.T,
            sizes=(7,),
            color="tab:blue",
            label="$(FPR, TPR)$ Pareto front",
        )
        ax.step(
            *np.c_[pareto_ratio.T, [1, 1]],
            where="post",
            lw=1.2,
            label="ROC (pessimistic)",
        )

        # Convex Hull
        ax.fill_between(
            *np.c_[pareto_ch_ratio.T, [1, 1]],
            alpha=0.15,
            color="tab:blue",
            label="ROC (convex hull)",
        )
        ax.plot(*np.c_[pareto_ch_ratio.T, [1, 1]], color="tab:blue", ls=":")

        if legend:
            ax.legend()
