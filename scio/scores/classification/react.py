"""ReAct for classification."""

__all__ = ["ReAct"]

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from scio.scores.utils import torch_quantile
from scio.utils import check

from .base import BaseScoreClassif

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


class ReAct(BaseScoreClassif):
    """ReAct for classification.

    Arguments
    ---------
    percentile: ``float``
        The percentile used to compute the upper threshold. Defaults to
        ``90.0``.
    symmetric: ``bool``
        Whether to apply similar lower clipping. Defaults to ``False``.
    per_layer: ``bool``
        If ``True``, every recorded layer gets clipped with its own
        threshold(s) computed independently. If ``False``, the
        threshold(s) is(are) computed over concatenated layers
        activations. Irrelevant when only one layer is recorded, as in
        :cite:`ReAct`. Defaults to ``False``.

    References
    ----------
    .. bibliography::
       :filter: false

       ReAct

    """

    percentile: float = 90.0  # [ReAct] default
    symmetric: bool = False  # [ReAct]
    per_layer: bool = False  # [ReAct]

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        check(0.5 * self.symmetric < self.percentile / 100 <= 1)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, _calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data."""
        self.rnet(calib_data)  # Records activations
        activations = self.activations(concatenate=not self.per_layer)

        q = self.percentile / 100

        # Max size for quantile: 2**24 (pytorch/issues/67592)
        # Should find alternative is necessary
        highs = [torch_quantile(act, q, interpolation="higher") for act in activations]
        lows = [
            (
                torch_quantile(activation, 1 - q, interpolation="lower")
                if self.symmetric
                else None
            )
            for activation in activations
        ]

        self.clippers: list[Callable] | Callable
        self.clippers = [
            partial(torch.clamp, min=low, max=high)
            for low, high in zip(lows, highs, strict=False)
        ]

        if not self.per_layer:
            self.clippers = self.clippers.pop()

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs)
        new_out = self.rnet(inputs, activation_postproc=self.clippers)
        conformity = new_out.softmax(1)

        return out, conformity
