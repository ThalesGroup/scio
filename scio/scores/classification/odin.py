"""ODIN for classification."""

__all__ = ["ODIN"]

import torch
from torch import Tensor

from scio.scores.utils import batched_grad, fgm_direction
from scio.utils import check

from .base import BaseScoreClassif


class ODIN(BaseScoreClassif):
    """ODIN for classification.

    Distilled softmax plus adversarial reinforcement.

    Arguments
    ---------
    temperature: ``float``
        Temperature scaling factor.
    epsilon: ``float``
        Amplitude of the adversarial reinforcement. If zero, the method
        is only applying temperature scaling to logits. Defaults to
        ``0.0``.
    fgm_norm: ``float``
        Parameter :math:`p` in the :math:`L^p` adversarial
        reinforcement. Defaults to ``inf``.

    References
    ----------
    .. bibliography::
       :filter: false

       ODIN

    """

    temperature: float
    epsilon: float = 0.0
    fgm_norm: float = torch.inf  # [ODIN]

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        check(self.temperature > 0)
        check(self.fgm_norm >= 1)

    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """No calibration needed."""

    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        T = self.temperature  # noqa: N806 (uppercase T)

        # Without adversarial reinforcement
        if not (eps := self.epsilon):
            with torch.no_grad():
                out = self.rnet(inputs, dont_record=True)

            conformity = (out / T).softmax(1)
            return out, conformity

        # With adversarial reinforcement
        inputs = inputs.requires_grad_()
        out = self.rnet(inputs, dont_record=True)
        logsoftmax = (out / T).log_softmax(1).amax(1)
        inputs_grad = batched_grad(logsoftmax, inputs)
        with torch.no_grad():
            adv_inputs = inputs + eps * fgm_direction(inputs_grad, p=self.fgm_norm)
            adv_out = self.rnet(adv_inputs)

        conformity = (adv_out / T).softmax(1)
        return out, conformity
