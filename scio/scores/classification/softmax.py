"""Softmax for classification."""

__all__ = ["Softmax"]

import torch
from torch import Tensor

from .base import BaseScoreClassif


class Softmax(BaseScoreClassif):
    """Softmax for classification.

    Conformity is softmax output.

    References
    ----------
    .. bibliography::
       :filter: false

       Softmax

    """

    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """No calibration needed."""

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs, dont_record=True)
        conformity = out.softmax(1)
        return out, conformity
