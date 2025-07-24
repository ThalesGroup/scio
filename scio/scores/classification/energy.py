"""Energy for classification."""

__all__ = ["Energy"]

import torch
from torch import Tensor

from scio.utils import check

from .base import BaseScoreClassif


class Energy(BaseScoreClassif):
    """Energy for classification.

    Arguments
    ---------
    temperature: ``float``
        Temperature scaling factor. Defaults to ``1.0``.

    References
    ----------
    .. bibliography::
       :filter: false

       Energy

    """

    temperature: float = 1.0  # [Energy] default

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        check(0 < self.temperature < torch.inf)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """No calibration needed."""

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs, dont_record=True)
        conformity = self.temperature * (out / self.temperature).logsumexp(1)

        return out, conformity
