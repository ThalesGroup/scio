"""Logit for classification."""

__all__ = ["Logit"]

import torch
from torch import Tensor

from .base import BaseScoreClassif


class Logit(BaseScoreClassif):
    """Logit for classification.

    Conformity is the raw net output (logit).

    References
    ----------
    .. bibliography::
       :filter: false

       Logit

    """

    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """No calibration needed."""

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs, dont_record=True)
        return out, out
