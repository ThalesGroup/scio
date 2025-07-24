"""Module implementing base class for classification scores."""

__all__ = [
    "BaseScoreClassif",
    "TemplateClassif",
]

from abc import abstractmethod

from paramclasses import protected
from torch import Tensor

from scio.scores.base import BaseScore
from scio.utils import ScoreClassifMode, ScoreClassifModeLike


class BaseScoreClassif(BaseScore[Tensor, Tensor]):
    """Base class for classification scores.

    In classification, the output of the network is expected to be in
    the logit space.

    Arguments
    ---------
    mode: ``ScoreClassifModeLike``
        See :class:`~scio.utils.ScoreClassifMode`. Defaults to
        ``"raw"``.

    """

    mode: ScoreClassifModeLike = "raw"

    def _on_param_will_be_set(self, attr: str, future_val: object) -> None:
        """Bypass unfit for specific attributes."""
        if attr != "mode":
            super()._on_param_will_be_set(attr, future_val)

    @protected
    def _process(self, inputs: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]  # github.com/eliegoudout/paramclasses/issues/34
        """Classification-specific postprocessing.

        1. If unique conformity per sample, extend to all classes.
        2. Compute preference instead of simple conformity if required.

        Note
        ----
        A preference ``mode`` is trivially useless if the score produces
        unique per-sample conformities.

        """
        out, raw_conformity = self.get_conformity(inputs)

        # Handle unique per-sample conformity
        class_slice = None if raw_conformity.ndim == 1 else slice(None)
        conformity = raw_conformity[:, class_slice].broadcast_to(out.shape)

        # Handle ``mode``
        conformity = _postprocess_conformity(conformity, self.mode)
        return out, conformity

    @abstractmethod
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference.

        Returns
        -------
        out: ``Tensor``
            The result of ``inputs`` forward pass through the network.
        conformity: ``Tensor``
            The conformity associated with ``out``, both sample-wise and
            class-wise. The shape must be one of ``(n_samples,
            n_classes)``, ``(n_samples, 1)`` or ``(n_samples,)``. In the
            last two cases, the same score will  be attributed to every
            class by reshaping and broadcasting.

        """


def _postprocess_conformity(conformity: Tensor, mode: ScoreClassifModeLike) -> Tensor:
    r"""Postprocess conformity with inter-class operation.

    Arguments
    ---------
    conformity: ``Tensor``
        The :math:`2`\ D tensor to postprocess. Shape ``(n_samples,
        n_classes)``. Should only contain finite values.
    mode: ``ScoreClassifModeLike``
        See :class:`~scio.utils.ScoreClassifMode`.

    Returns
    -------
    out: ``Tensor``
        The result of the postprocessing.

    Raises
    ------
    :exc:`ValueError`
        If ``mode`` value is unsupported.

    """
    mode = ScoreClassifMode(mode)
    if mode == ScoreClassifMode.RAW:
        return conformity

    # Mode ``"diff"`` or ``"ratio"``
    range_ = range(len(conformity))
    top1_idx, top2_idx = conformity.topk(2)[1].T
    if mode == ScoreClassifMode.DIFF:
        preference = conformity - conformity[range_, top1_idx, None]
        preference[range_, top1_idx] = -preference[range_, top2_idx]
        return preference
    if mode == ScoreClassifMode.RATIO:
        preference = conformity / conformity[range_, top1_idx, None]
        preference[range_, top1_idx] = 1 / preference[range_, top2_idx]
        return preference

    msg = f"Unsupported mode: {mode!r}"
    raise ValueError(msg)


class TemplateClassif(BaseScoreClassif):  # pragma: no cover
    """Template for classification.

    The source code of this class is a good starting point if you wish
    to implement a new score.

    Arguments
    ---------
    param_with_a_default_value: ``str``
        A parameter with a default value. Defaults to ``"default
        value"``.
    param_with_no_default_value: ``int``
        A parameter with no default value.

    References
    ----------
    .. bibliography::
       :filter: false
       :keyprefix: classif

       TemplateReference

    """

    param_with_a_default_value: str = "default value"
    param_with_no_default_value: int  # No default value

    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data.

        Example
        -------
        ::

            def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
                \"\"\"Calibrate the scoring algorithm with In-Distribution data.\"\"\"
                out = self.rnet(calib_data)  # records activations
                activations = self.activations()
                self.calibration_statistics = ...

        """
        raise NotImplementedError

    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference.

        Example
        -------
        ::

            def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
                \"\"\"Compute output and associated conformity at inference.\"\"\"
                out = self.rnet(inputs)  # records activations
                conformity = ...
                return out, conformity

        """
        raise NotImplementedError
