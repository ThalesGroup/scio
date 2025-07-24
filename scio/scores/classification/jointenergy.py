"""JointEnergy for classification."""

__all__ = ["JointEnergy"]

import torch
from torch import Tensor

from .base import BaseScoreClassif


class JointEnergy(BaseScoreClassif):
    """JointEnergy for classification.

    References
    ----------
    .. bibliography::
       :filter: false

       JointEnergy

    """

    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """No calibration needed."""

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs, dont_record=True)
        free_energies = torch.stack((torch.zeros_like(out), out)).logsumexp(0)
        conformity = free_energies.sum(1)

        return out, conformity
