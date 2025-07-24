"""GradNorm for classification."""

__all__ = ["GradNorm"]

import torch
from torch import Tensor
from torch.func import functional_call, grad, vmap

from scio.utils import check

from .base import BaseScoreClassif


class GradNorm(BaseScoreClassif):
    """GradNorm for classification.

    Personal interpretation: The computed gradient norm mimicks brain
    elasticity estimation. The network should be more elastic in
    In-Distribution regions.

    Arguments
    ---------
    temperature: ``float``
        Temperature scaling factor. Defaults to ``1.0``.
    grad_norm: ``float``
        Order of the vector norm used for gradnorm computation. Defaults
        to ``1.0``.
    discard_functional_forward: ``bool``
        Whether to compute output from vanilla forward, if the
        implemented functional call is not satisfactory. Setting this to
        ``True`` requires an additional forward pass through the
        network. Defaults to ``False``.

    References
    ----------
    .. bibliography::
       :filter: false

       GradNorm

    """

    temperature: float = 1.0  # [GradNorm] default
    grad_norm: float = 1.0  # [GradNorm] default
    discard_functional_forward: bool = False

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        check(self.temperature > 0)

    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """No calibration needed."""

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        # Vectorized per-sample computation of weights gradients: follow great tutorial
        # https://pytorch.org/tutorials/intermediate/per_sample_grads.html
        params = {
            k: v.detach()
            for k, v in self.rnet.recorded_params.items()
            if v.requires_grad
        }
        if not params:
            msg = (
                "Recorded layers must have at least one learnable parameters for score"
                f" {type(self).__qualname__}"
            )
            raise RuntimeError(msg)

        grads, out = self.params_gradients_with_logits(params, inputs)
        conformity = torch.linalg.vector_norm(grads, ord=self.grad_norm, dim=1)

        if self.discard_functional_forward:
            out = self.rnet(inputs, dont_record=True)

        return out, conformity

    def to_derive(
        self,
        params: dict[str, Tensor],
        sample: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Quantity to derive as a function of selected parameters."""
        sample_logits = functional_call(
            self.rnet,
            params,
            args=sample[None],
            kwargs={"dont_record": True},
        )[0]
        kl_div = -(sample_logits / self.temperature).log_softmax(0).mean()  # + constant
        return kl_div, sample_logits

    def params_gradients_with_logits(
        self,
        params: dict[str, Tensor],
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """In batch: computes gradients, flattens and concatenates."""
        per_sample_grad_fn = vmap(grad(self.to_derive, has_aux=True), in_dims=(None, 0))
        grad_dict, logits = per_sample_grad_fn(params, inputs)
        grad_2d = torch.cat([t.flatten(1) for t in grad_dict.values()], dim=1)
        return grad_2d, logits
