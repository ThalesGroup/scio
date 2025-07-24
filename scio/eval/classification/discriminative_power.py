"""Implements a few discriminative power measures.

Given binary labels and scalar scores, a discriminative power quantifies
our ability to discriminate between positive and negative samples
through thresholding strategies.

To be considered a proper discriminative power measure, a metric must:

- Depend only on the ``(FP, TP)`` Pareto front;
- Satisfy basic monotonicity constraints (e.g. if an OoD input is
  assigned a greater confidence score, the metric should not increase).

The first condition implies that every discriminative power measure can
be computed from ``ROC(labels, scores)``.
"""


# ruff: noqa: N806 (uppercase variables)

__all__ = [
    "AUC",
    "MCC",
    "TNR",
    "TPR",
    "BaseDiscriminativePower",
]

from abc import abstractmethod
from types import SimpleNamespace

import numpy as np
from numpy.typing import ArrayLike
from paramclasses import ParamClass, protected

from scio.utils import InterpolationKind, InterpolationKindLike, check

from .roc import ROC


class BaseDiscriminativePower(ParamClass):
    r"""Base class for discriminative power metrics.

    Metrics used when thresholding score :math:`S` with
    :math:`S\leqslant\tau` for threshold :math:`\tau`. Positive examples
    should verify this inequality and thus have low score.

    A discriminative power is defined through:

    - a computation procedure, from a :class:`~scio.eval.ROC` instance;
    - optionally, parameters checks.

    These should respectively been carried out in :meth:`from_roc` and
    :meth:`_check_params` (see source code for examples).

    Discriminative power classes are `paramclasses
    <https://github.com/eliegoudout/paramclasses>`_ and hyperparameters
    are defined as their *parameters*.
    """

    # ====================== TO IMPLEMENT IN DISCRIMINATIVE POWER ======================
    @abstractmethod
    def from_roc(self, roc: ROC) -> float:
        """Compute discriminative power from :class:`~scio.eval.ROC`.

        Returns
        -------
        discriminative_power: ``float``

        """

    @staticmethod
    def _check_params(params: SimpleNamespace) -> None:
        """Optionally check params."""

    # ==================================================================================

    @protected
    def __post_init__(self) -> None:
        """Run final params check.

        Redundant only if params were passed at instanciation.
        """
        self._check_params(SimpleNamespace(**self.params))  # type: ignore[operator]  # github.com/eliegoudout/paramclasses/issues/34

    @protected
    def _on_param_will_be_set(self, attr: str, future_val: object) -> None:  # type: ignore[override]  # github.com/eliegoudout/paramclasses/issues/34
        """Call ``self._check_params()`` on future params."""
        future_params = SimpleNamespace(**(self.params | {attr: future_val}))  # type: ignore[operator]  # github.com/eliegoudout/paramclasses/issues/34
        self._check_params(future_params)

    @protected
    def __call__(self, labels: ArrayLike, scores: ArrayLike) -> float:
        """Compute discriminative power for given labels and scores.

        It is a simple shorthand::

            self(labels, scores)                # Use this...
            self.from_roc(ROC(labels, scores))  # ...instead of this

        """  # fmt: skip
        roc = ROC(labels, scores)
        return self.from_roc(roc)


class AUC(BaseDiscriminativePower):
    """AUC for ROC, potentially partial — in which case normalized.

    Arguments
    ---------
    max_fpr: ``float``
        Maximum False Positive Rate for *partial* AUC. Defaults to
        ``1.0``.
    min_fpr: ``float``
        Minimum False Positive Rate for *partial* AUC. Defaults to
        ``0.0``.
    kind: ``InterpolationKindLike``
        See :class:`~scio.utils.InterpolationKind`. Defaults to
        ``"pessimistic"``.

    """

    max_fpr: float = 1.0
    min_fpr: float = 0.0
    kind: InterpolationKindLike = "pessimistic"

    @staticmethod
    def _check_params(params: SimpleNamespace) -> None:
        check(0 <= params.min_fpr < params.max_fpr <= 1)
        InterpolationKind(params.kind)

    def from_roc(self, roc: ROC) -> float:
        """Compute discriminative power from ROC data."""
        kind = InterpolationKind(self.kind)
        ch = kind == InterpolationKind.CONVEX_HULL
        FPR, TPR = (roc.FPRch, roc.TPRch) if ch else (roc.FPR, roc.TPR)

        # Restrict to desired partial, with extra edges
        interval = [self.min_fpr, self.max_fpr]
        i0, i1 = np.searchsorted(FPR, interval, "right")
        pTPR = np.append(TPR, 1)[i0 - 1 : i1 + 1].copy()
        pFPR = np.append(FPR, 1)[i0 - 1 : i1 + 1].copy()

        # Fix edge values and compute AUC
        if kind == InterpolationKind.PESSIMISTIC:
            pFPR[[0, -1]] = interval
            pAUC = pTPR[:-1] @ np.diff(pFPR)
        elif kind == InterpolationKind.CONVEX_HULL:
            pTPR[[0, -1]] = np.interp(interval, pFPR, pTPR)
            pFPR[[0, -1]] = interval
            pAUC = np.trapezoid(pTPR, pFPR)
        else:
            msg = f"Unsupported kind of {type(self).__name__}: {self.kind!r}"
            raise NotImplementedError(msg)

        # Return rescaled
        return pAUC / np.ptp(interval)


class MCC(BaseDiscriminativePower):
    r"""Maximum Matthews Correlation Coefficient.

    Evaluates to the maximum `Matthews Correlation Coefficient
    <https://en.wikipedia.org/wiki/Phi_coefficient#Machine_learning>`_
    over every possible threshold:

    .. math::

        MCC = \max_{\text{threshold}\in\mathbb{R}}MCC(\text{threshold}),

    where :math:`\text{threshold}` naturally defines :math:`FP, TP, FN,
    TN`, and

    .. math::

        MCC(\text{threshold}) = \frac{TP\times TN - FP\times FN}
                                     {\sqrt{(TP+FP)
                                            (TP+FN)
                                            (TN+TP)
                                            (TN+FN)}}.

    One has :math:`MCC\in[0, 1]` since, for example, the degenerate case
    :math:`\text{threshold}=+\infty` evaluates to :math:`0` for
    consistency.
    """

    def from_roc(self, roc: ROC) -> float:
        """Compute discriminative power from ROC data."""
        N, P = roc.N, roc.P
        FP, TP = roc.FP, roc.TP
        FN, TN = P - TP, N - FP
        num = TP * TN - FP * FN
        den2 = (TP + FP) * (TN + FN) * P * N
        # Suppress division by 0 warnings since np.where evaluates both options.
        with np.errstate(invalid="ignore"):
            MCCs = np.where(den2 > 0.0, num / np.sqrt(den2), 0.0)
        return MCCs.max()


class TPR(BaseDiscriminativePower):
    """True Positive Rate.

    Arguments
    ---------
    max_fpr: ``float``, optional
        Maximum *False Positive Rate*. Exactly one of ``max_fpr`` and
        ``min_tnr`` must be provided.
    min_tnr: ``float``, optional
        Minimum *True Negative Rate*. Exactly one of ``max_fpr`` and
        ``min_tnr`` must be provided.
    kind: ``InterpolationKindLike``
        See :class:`~scio.utils.InterpolationKind`. Defaults to
        ``"pessimistic"``.

    """

    max_fpr: float | None = None
    min_tnr: float | None = None
    kind: InterpolationKindLike = "pessimistic"

    @staticmethod
    def _check_params(params: SimpleNamespace) -> None:
        """Sanitize ``max_fpr`` and ``min_tnr`` at instanciation."""
        if (max_fpr := params.max_fpr) is not None:
            check(0 <= max_fpr <= 1)
        if (min_tnr := params.min_tnr) is not None:
            check(0 <= min_tnr <= 1)
        InterpolationKind(params.kind)

    def from_roc(self, roc: ROC) -> float:
        """Compute discriminative power from ROC data."""
        max_fpr = unambiguous_val(self.max_fpr, self.min_tnr, "max_fpr", "min_tnr")
        kind = InterpolationKind(self.kind)
        ch = kind == InterpolationKind.CONVEX_HULL
        FPR, TPR = (roc.FPRch, roc.TPRch) if ch else (roc.FPR, roc.TPR)

        if kind == InterpolationKind.PESSIMISTIC:
            idx = np.searchsorted(FPR, max_fpr, "right") - 1
            return TPR[idx]
        if kind == InterpolationKind.CONVEX_HULL:
            return np.interp(max_fpr, FPR, TPR)

        msg = f"Unsupported kind of {type(self).__name__}: {self.kind!r}"
        raise NotImplementedError(msg)


class TNR(BaseDiscriminativePower):
    """True Negative Rate.

    Arguments
    ---------
    min_tpr: ``float``, optional
        Minimum *True Positive Rate*. Exactly one of ``min_tpr`` and
        ``max_fnr`` must be provided.
    max_fnr: ``float``, optional
        Maximum *False Negative Rate*. Exactly one of ``min_tpr`` and
        ``max_fnr`` must be provided.
    kind: ``InterpolationKindLike``
        See :class:`~scio.utils.InterpolationKind`. Defaults to
        ``"pessimistic"``.

    """

    min_tpr: float | None = None
    max_fnr: float | None = None
    kind: InterpolationKindLike = "pessimistic"

    @staticmethod
    def _check_params(params: SimpleNamespace) -> None:
        """Sanitize ``min_tpr`` and ``max_fnr`` at instanciation."""
        if (min_tpr := params.min_tpr) is not None:
            check(0 <= min_tpr <= 1)
        if (max_fnr := params.max_fnr) is not None:
            check(0 <= max_fnr <= 1)
        InterpolationKind(params.kind)

    def from_roc(self, roc: ROC) -> float:
        """Compute discriminative power from ROC data."""
        min_tpr = unambiguous_val(self.min_tpr, self.max_fnr, "min_tpr", "max_fnr")
        kind = InterpolationKind(self.kind)
        ch = kind == InterpolationKind.CONVEX_HULL
        FPR, TPR = (roc.FPRch, roc.TPRch) if ch else (roc.FPR, roc.TPR)

        if kind == InterpolationKind.PESSIMISTIC:
            idx = np.searchsorted(TPR, min_tpr)
            return 1 - FPR[idx]
        if kind == InterpolationKind.CONVEX_HULL:
            return 1 - np.interp(min_tpr, TPR, FPR)

        msg = f"Unsupported kind of {type(self).__name__}: {self.kind!r}"
        raise NotImplementedError(msg)


def unambiguous_val(
    val: float | None,
    anti_val: float | None,
    val_name: str,
    anti_val_name: str,
) -> float:
    """Extract ``val`` from ``(val, anti_val)`` and raise if necessary.

    Exactly one of ``val`` and ``anti_val`` should not be ``None``,
    unless they sum to ``1``.

    Returns
    -------
    float
        ``val`` if provided, else ``1 - anti_val``.

    Raises
    ------
    :exc:`ValueError`
        If not exactly one of ``val`` or ``anti_val`` is not ``None``.

    """
    msg = f"Specify exactly one of {val_name!r} or {anti_val_name!r}, not {{}}"

    if val is None:
        if anti_val is None:
            raise ValueError(msg.format("none"))
        return 1 - anti_val

    if anti_val is None or val + anti_val == 1:
        return val
    raise ValueError(msg.format("both"))
