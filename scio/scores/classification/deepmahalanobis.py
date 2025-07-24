"""DeepMahalanobis for classification."""

__all__ = ["DeepMahalanobis"]

from collections.abc import Sequence
from warnings import warn

import torch
from torch import Tensor

from scio.scores.utils import batched_grad, fgm_direction
from scio.utils import check

from .base import BaseScoreClassif


class DeepMahalanobis(BaseScoreClassif):
    """DeepMahalanobis for classification.

    Arguments
    ---------
    epsilon: ``float``
        Amplitude of the adversarial reinforcement. Defaults to ``0``.
    fgm_norm: ``float``
        Parameter :math:`p` for :math:`L^p` adversarial reinforcement.
        Defaults to ``inf``.
    weights: ``Sequence[float]``, optional
        Weights for layer aggregation. If not provided, treated as
        ``[1] * n_layers`` at inference.

    Notes
    -----
    In :cite:`DeepMahalanobis`, the authors benchmark in a supervised
    setup, which allows the learning of ``weights``. The unsupervised
    equivalent uses predetermined weights or only one layer (when
    ``weights`` is not provided).

    When ``epsilon > 0``, this score applies an "adversarial
    perturbation" to move input sample closer to its best candidate
    class. The paper does not mention conformity for other classes. As
    such, when computing conformity for other classes, we chose to use
    still use the perturbation for the best candidate class (for
    computational reasons).

    Functionality for Relative Mahalanobis Distance (RMD) is implemented
    but unused. See :class:`~.relativemahalanobis.RelativeMahalanobis`
    for more info.

    Note
    ----
    Due to a design choice in favor of inference efficiency, this class
    is not numerically very stable for ``torch.half``. See
    :meth:`compute_precision`.

    References
    ----------
    .. bibliography::
       :filter: false

       DeepMahalanobis

    """

    _relative = False  # See note for RMD
    epsilon: float = 0.0
    fgm_norm: float = torch.inf  # [DeepMahalanobis]
    weights: Sequence[float] | None = None

    def _check_relative(self) -> None:
        """Check ``_relative`` flag (RMD-related)."""
        check(not self._relative)

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        self._check_relative()
        check(self.fgm_norm >= 1)

    def _on_param_will_be_set(self, attr: str, future_val: object) -> None:
        """Bypass unfit for specific attributes."""
        if attr not in {"epsilon", "fgm_norm", "weights"}:
            super()._on_param_will_be_set(attr, future_val)
        if attr == "fgm_norm":
            check(self.fgm_norm >= 1)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data."""
        n_classes = self.rnet(calib_data).shape[1]  # Records activations
        all_activations = self.activations()
        self.mu = []
        self.precision = []

        for activations in all_activations:
            acts_flat = activations.flatten(1)

            # Centroids computation
            mu = torch.empty(
                n_classes,
                acts_flat.shape[1],
                dtype=acts_flat.dtype,
                device=acts_flat.device,
            )
            for c in range(n_classes):
                mu[c] = acts_flat[calib_labels == c].mean(0)

            # Covariance/precision computation
            precision = self.compute_precision(acts_flat - mu[calib_labels])

            self.mu.append(mu)
            self.precision.append(precision)

        if not self._relative:  # See Note on RMD
            return

        self.mu0 = [a.mean(0).flatten() for a in all_activations]
        self.precision0 = [
            self.compute_precision(a.flatten(1) - mu0)
            for a, mu0 in zip(all_activations, self.mu0, strict=True)
        ]

    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        weights = torch.tensor(
            [1] * len(self.rnet.recording) if self.weights is None else self.weights,
        ).to(inputs)

        # With adversarial reinforcement
        if self.epsilon:
            inputs = inputs.requires_grad_()
            out = self.rnet(inputs)  # Records activations
            all_D2 = torch.empty(  # noqa: N806 (uppercase all_D2)
                *out.shape,
                *weights.shape,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            all_activations = self.activations()
            for layer, (activations, mu, precision) in enumerate(
                zip(all_activations, self.mu, self.precision, strict=True),
            ):
                if self._relative:  # See Note on RMD
                    mu0, precision0 = self.mu0[layer], self.precision0[layer]

                # Find candidate class
                acts_flat = activations.flatten(1)
                with torch.no_grad():
                    candidates = (
                        self.compute_mahalanobis(acts_flat, mu, precision)
                        .nan_to_num(nan=torch.inf)
                        .argmin(1)
                    )

                # Compute minimum Mahalanobis with grads
                vanilla_D2 = self.compute_mahalanobis(  # noqa: N806 (uppercase vanilla_D2)
                    acts_flat,
                    mu[candidates],
                    precision,
                    product=False,
                )
                if self._relative:  # See Note on RMD
                    vanilla_D2 = vanilla_D2 - self.compute_mahalanobis(  # noqa: N806 (uppercase vanilla_D2)
                        acts_flat,
                        mu0,
                        precision0,
                        product=False,
                    )

                # Add adversarial perturbation to increase corresponding conformity
                inputs_grad = batched_grad(-vanilla_D2, inputs, retain_graph=True)
                try:
                    adv_inputs = inputs + self.epsilon * fgm_direction(
                        inputs_grad,
                        p=self.fgm_norm,
                    )
                except ValueError:
                    msg = (
                        "Error during the computation of adversarial perturbation at "
                        f"layer {self.rnet.recording[layer]}. This may be due to "
                        "numerical instability. Continuing without adversarial "
                        "reinforcement"
                    )
                    warn(msg, stacklevel=2)
                    adv_inputs = inputs

                # Compute new Mahalanobis distance
                with torch.no_grad():
                    self.rnet(adv_inputs)  # Recompute activations and fetch at layer
                    adv_activations = self.activations()[layer]
                    adv_acts_flat = adv_activations.flatten(1)
                    adv_D2 = self.compute_mahalanobis(adv_acts_flat, mu, precision)  # noqa: N806 (uppercase adv_D2)
                    if self._relative:  # See Note on RMD
                        adv_D2 -= self.compute_mahalanobis(  # noqa: N806 (uppercase adv_D2)
                            adv_acts_flat,
                            mu0[None],
                            precision0,
                        )
                    all_D2[..., layer] = adv_D2

        # Without adversarial reinforcement
        else:
            with torch.no_grad():
                out = self.rnet(inputs)  # Records activations
                all_D2 = torch.empty(  # noqa: N806 (uppercase all_D2)
                    *out.shape,
                    *weights.shape,
                    dtype=inputs.dtype,
                    device=inputs.device,
                )
                all_activations = self.activations()
                for layer, (activations, mu, precision) in enumerate(
                    zip(all_activations, self.mu, self.precision, strict=True),
                ):
                    acts_flat = activations.flatten(1)
                    vanilla_D2 = self.compute_mahalanobis(acts_flat, mu, precision)  # noqa: N806 (uppercase vanilla_D2)
                    if self._relative:  # See Note on RMD
                        mu0, precision0 = self.mu0[layer], self.precision0[layer]
                        vanilla_D2 = vanilla_D2 - self.compute_mahalanobis(  # noqa: N806 (uppercase vanilla_D2)
                            acts_flat,
                            mu0[None],
                            precision0,
                        )

                    all_D2[..., layer] = vanilla_D2

        conformity = -(all_D2 @ weights).nan_to_num(nan=torch.inf)
        return out.detach(), conformity

    @staticmethod
    def compute_mahalanobis(
        samples: Tensor,
        mu: Tensor,
        precision: Tensor,
        *,
        product: bool = True,
    ) -> Tensor:
        """Compute squared Mahalanobis distance.

        Arguments
        ---------
        samples: ``Tensor``
            Explicit. Shape ``(*s_shape, data_dim)``.
        mu: ``Tensor``
            Centroids. Shape ``(*m_shape, data_dim)``.
        precision: ``Tensor``
            Inverse of covariance matrix.
        product: ``bool``
            Whether to compute for the cartesian product of combinations
            of samples and centroids. If ``False``, requires ``samples``
            and ``mu`` to be broadcastable. Defaults to ``True``.

        Returns
        -------
        D2: ``Tensor``
            Squared Mahalanobis distance for combinations (defined by
            ``product``) of samples and centroids. If ``product``, the
            output shape is ``(*s_shape, *m_shape)``. Else, it is the
            result of broadcasting ``s_shape`` and ``m_shape`` together.

        """
        samples_ = samples[..., *(None,) * (mu.ndim - 1), :] if product else samples
        residues = samples_ - mu
        return (residues @ precision * residues).sum(-1)

    @staticmethod
    def compute_precision(residues: Tensor) -> Tensor:
        """Compute the inverse of the covariance of the residues.

        To avoid a potential ``GPU > CPU > GPU`` bottleneck, we use
        torch's experimental :func:`inv_ex`, which does not support
        ``torch.half``. This may induce a temporary type conversion
        internally.

        Arguments
        ---------
        residues: ``Tensor``
            Assumed centered feature-wise. Shape ``(n_samples,
            data_dim)``.

        Returns
        -------
        precision: ``Tensor``
            Inverse covariance matrix. Shape ``(data_dim, data_dim)``.
            Full of ``nan`` if the covariance matrix is singular (*i.e.*
            for ``residues`` residing on a strict submanifold) or
            generates numerical overflow during inversion.

        Note
        ----
        We compute and store the ``precision`` since it may be used
        multiple times. Note however that for a single Mahalanobis
        distance computation, it is faster and numerically more stable
        to use :func:`torch.lstsq` with the covariance directly.

        """
        dtype = torch.float if residues.dtype is torch.half else residues.dtype
        cov = residues.T.to(dtype=dtype).cov()

        try:
            precision = torch.linalg.inv_ex(cov, check_errors=True).inverse.to(residues)
            if not precision.isfinite().all():
                raise torch.linalg.LinAlgError  # pragma: no cover  # noqa: TRY301
        except torch.linalg.LinAlgError:
            data_dim = residues.size(1)
            return torch.full((data_dim, data_dim), torch.nan).to(residues)

        return precision
