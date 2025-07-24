"""Odds for classification."""

__all__ = ["Odds"]

import warnings

import torch
from torch import Tensor

from scio.utils import ZAggr, ZAggrLike, check

from .base import BaseScoreClassif


class Odds(BaseScoreClassif):
    r"""Odds for classification.

    Arguments
    ---------
    epsilon: ``float``
        :math:`L^{\infty}` scale of uniformly sampled random perturbations.
    noise_samples: ``int``
        Number of random samples to compute log-preferences. Defaults to
        ``2**8``.
    z_aggregation: ``ZAggrLike``
        See :class:`~scio.utils.ZAggr`. Defaults to ``"sum"``.
    batch_size: ``int``
        **Currently Not Implemented!**

        Necessary for computational reasons. Data and env-dependent. Use
        ``0`` for unlimited. Defaults to ``2**13``.
    rng_seed: ``int``, optional
        If provided, manual seed for ``torch.Generator``, used during
        calibration (in which case the random state is reset at every
        :meth:`~scio.scores.BaseScore.fit` call). Defaults to ``0``.

    Warning
    -------
    Current implementation is unbatched. This means it does not scale
    well for large number of random samples.

    References
    ----------
    .. bibliography::
       :filter: false

       Odds

    """

    epsilon: float
    noise_samples: int = 2**8  # [Odds] default
    z_aggregation: ZAggrLike = "sum"
    # batch_size: int = 2**13  # noqa: ERA001 (commented code)
    rng_seed: int | None = 0

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        check(self.noise_samples > 0)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data."""
        self.rng: torch.Generator | None
        if self.rng_seed is None:
            self.rng = None
        else:
            self.rng = torch.Generator(calib_data.device).manual_seed(self.rng_seed)

        logits = self.rnet(calib_data, dont_record=True)
        n_samples, n_classes = logits.shape
        self.g_stats_avg = torch.full(
            (n_classes,) * 2,
            torch.nan,
            dtype=logits.dtype,
            device=logits.device,
        )
        self.g_stats_var = torch.full(
            (n_classes,) * 2,
            torch.nan,
            dtype=logits.dtype,
            device=logits.device,
        )
        for c in range(n_classes):
            c_mask = calib_labels == c
            c_inputs = calib_data[c_mask]
            c_logits = logits[c_mask]
            c_logprefs = self.logits_pref(c_logits, c)
            random_perturbations = self.epsilon * self.uniform(
                len(c_inputs) * self.noise_samples,
                *c_inputs.shape[1:],
                to=c_inputs,
            )
            random_inputs = (
                c_inputs.repeat_interleave(self.noise_samples, dim=0)
                + random_perturbations
            )
            random_logits = self.rnet(random_inputs, dont_record=True)
            random_logprefs = self.logits_pref(random_logits, c)
            c_gstats = random_logprefs - c_logprefs.repeat_interleave(
                self.noise_samples,
                dim=0,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.g_stats_avg[c] = c_gstats.mean(0)
                self.g_stats_var[c] = c_gstats.var(0)

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = logits = self.rnet(inputs, dont_record=True)
        n_samples, n_classes = out.shape

        random_perturbations = self.epsilon * self.uniform(
            len(inputs) * self.noise_samples,
            *inputs.shape[1:],
            to=inputs,
        )
        random_inputs = (
            inputs.repeat_interleave(self.noise_samples, dim=0) + random_perturbations
        )
        random_logits = self.rnet(random_inputs, dont_record=True)
        avg_logits = random_logits.reshape(
            n_samples,
            self.noise_samples,
            n_classes,
        ).mean(1)
        diff_logits = avg_logits - logits
        g_stats = diff_logits[:, None] - diff_logits[:, :, None]
        z_scores = (g_stats - self.g_stats_avg) / self.g_stats_var

        # Conformity in unsupervised setup (see class doc)
        z_aggregation = ZAggr(self.z_aggregation)
        range_ = range(n_classes)
        if z_aggregation == ZAggr.SUM:
            z_scores[:, range_, range_] = 0
            nonconformity = z_scores.sum(2)
        elif z_aggregation == ZAggr.DETECT:
            z_scores[:, range_, range_] = -torch.inf
            nonconformity = z_scores.amax(2)
        elif z_aggregation == ZAggr.RECTIFY:
            z_scores[:, range_, range_] = torch.inf
            nonconformity = z_scores.amin(2)
        else:
            msg = f"Unsupported z-aggregation scheme: {self.z_aggregation!r}"
            raise ValueError(msg)

        conformity = -nonconformity.nan_to_num(nan=torch.inf)
        return out, conformity

    @staticmethod
    def logits_pref(samples: Tensor, reference_idx: int) -> Tensor:
        """Compute logits difference to reference index.

        In :cite:`Odds` notations, corresponds to :math:`f_{y, z}` for
        all :math:`z` given ``samples`` :math:`f` and ``reference_idx``
        :math:`y`.

        Arguments
        ---------
        samples: ``Tensor``
            Logits. Shape ``(n_samples, n_classes)``.
        reference_idx: ``int``
            Index of the reference to subtract.

        """
        return samples - samples[:, [reference_idx]]

    def uniform(self, *size: int, to: Tensor) -> Tensor:
        """Sample uniformly random ``Tensor`` in :math:`[-1, 1]^{size}`.

        The output device and dtype are those of ``to``. The same
        applies to its shape if ``size`` if empty.

        Arguments
        ---------
        *size: ``int``
            If provided, unpacked target size.
        to: ``Tensor``
            Tensor defining the target dtype and device. Also defines
            the target size if `size` is empty.

        """
        size = size or to.shape
        rand01 = torch.rand(*size, generator=self.rng, dtype=to.dtype, device=to.device)
        return 1 - 2 * rand01
