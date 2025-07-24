"""IsoMax for classification."""

__all__ = ["IsoMax"]

import torch
from torch import Tensor

from scio.utils import check

from .base import BaseScoreClassif


class IsoMax(BaseScoreClassif):
    """Isomax for classification.

    The authors propose to train networks with modified logits, learning
    "prototypes". Since retraining is not allowed in our setup, we
    propose to simply optimize the so-called "prototypes" a posteriori
    for the given pretrained network, minimizing the associated proposed
    function. We then use the same scoring function.

    Arguments
    ---------
    entropic_scale: ``float``
        Scaling factor for the entropy computation. Defaults to ``10``.
    dist_norm: ``float``
        Order of the vector norm used in the prototypes space. Defaults
        to ``2.0``.
    lr: ``float``
        Learning rate for prototypes optimization, passed to
        :class:`torch.optim.Adam`. Defaults to ``0.1``.
    n_steps: ``int``
        Number of learning steps for prototypes optimization for
        :class:`torch.optim.Adam`. Defaults to ``1000``.

    References
    ----------
    .. bibliography::
       :filter: false

       IsoMax

    """

    entropic_scale: float = 10.0  # [IsoMax] default
    dist_norm: float = 2.0  # [IsoMax] default
    lr: float = 0.1
    n_steps: int = 1000

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        check(self.lr > 0)
        check(self.n_steps >= 0)

    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data.

        Learn what authors call prototypes.
        """
        with torch.no_grad():
            logits = self.rnet(calib_data, dont_record=True)

        n_classes = logits.size(1)
        # Init at 0 like [IsoMax]
        self.prototypes = torch.zeros(
            n_classes,
            n_classes,
            dtype=logits.dtype,
            device=logits.device,
            requires_grad=True,
        )

        # Optimize prototypes to minimize loss
        eps = 1e-4 if logits.dtype is torch.half else 1e-8  # Adjust numerical stability
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr, eps=eps)
        for _ in range(self.n_steps):
            optimizer.zero_grad()
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn(self.modified_logits(logits), calib_labels).backward()
            optimizer.step()

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs, dont_record=True)
        conformity = self.modified_logits(out).softmax(1)
        return out, conformity

    def modified_logits(self, logits: Tensor) -> Tensor:
        """Negative :math:`L^p` distance to prototype, with scale."""
        distances = torch.linalg.vector_norm(
            logits[:, None] - self.prototypes,
            ord=self.dist_norm,
            dim=2,
        )  # (n_samples, n_classes)
        return -self.entropic_scale * distances
