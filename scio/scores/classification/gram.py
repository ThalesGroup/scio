"""Gram for classification."""

__all__ = ["Gram"]

import torch
from torch import Tensor

from scio.utils import LabelKind, LabelKindLike, check

from .base import BaseScoreClassif


class Gram(BaseScoreClassif):
    r"""Gram for classification.

    Arguments
    ---------
    max_gram_order: ``int``
        Max order of Gram matrices to use. In :cite:`Gram`, authors use
        ``10``. We set default to ``8`` and do not recommend going
        higher. The improvement is marginal if not none, while
        increasing the required computational ressources and potential
        for overflows.
    cut_off: ``float``
        Following OODEEL's approach, instead of using ``(min, max)`` for
        deviation computation, we use quantiles ``(cut_off,
        1-cut_off)``. We also evaluate the expected deviation on the
        same calibration data. Thus ``cut_off`` should not be ``0``.
        Defaults to ``0.1`` and does not seem critical from a small
        ablation study.
    separate_diagonal: ``bool``
        Whether to separate diagonal correlations from the off-diagonal
        sum. See :cite:`Gram{appendix C}`. Defaults to ``False``.
    calib_labels: ``LabelKindLike``
        See :class:`~scio.utils.LabelKind`. Defaults to ``"pred"``.

    Notes
    -----
    In :cite:`Gram{eq. (2)}`, the authors define :math:`Gp :=
    (F^p\times{}^{\mathrm t}F^p)^{1/p}`, which is ill-defined for
    negative correlations. We fix this here using absolute values, but
    this is never discussed in the article.

    The above definition may generate many ``inf`` or ``nan``. The sane
    way to treat them is not at all straightforward. Indeed, one could
    either use :func:`nanmax` to ignore them or consider the max to be
    ``nan`` if one sample is ``nan``. The second case deactivates many
    correlations while the former feels inconsistent (``nan``
    correlations would be treated as :math:`0` deviation). Treating them
    as infinite deviation is degenerate as it would render total
    deviation infinite for too many samples. We decide to ignore ``nan``
    correlations here. Sadly, it means that if a test sample only has
    ``nan`` correlations because of overflow, it will have a total
    deviation of :math:`0`, making it a "normal" sample... I found no
    sound way to fix this overflow issue except using lower gram matrix
    orders (not :math:`10` as in article) and also taking ``(q, 1-q)``
    quantiles instead of ``(min, max)`` at calibration, like in OODEEL.
    The quantile approach may allow to avoid tails of the distribution
    containing ``nan``, but if feels very optimistic.

    The rescaling :cite:`Gram{eq. (5)}` is done using the expected value
    on some data. If there are a few extreme samples in the training
    data used to compute mins and maxs, no validation sample will have a
    nonzero deviation, making the expected deviation :math:`0` and
    raising issue. I think OODEEl's approach to use a quantile instead
    of the min and max solves this problem in a sound way. As such, We
    propose to use ``(10, 90)`` percentiles of the calibration
    correlations to compute the deviation and expected deviation.

    As mentioned in :cite:`Gram`, the method scales very poorly with the
    number of channels. The authors propose to use a row-wise max
    instead of keeping all pairwise correlations to make it linear in
    the number of channels. It feels a bit arbitrary and monkeypatchy
    but we use this as it quickly becomes necessary, like in OODEEL.

    References
    ----------
    .. bibliography::
       :filter: false

       Gram

    """

    max_gram_order: int = 8  # See module's doc
    cut_off: float = 0.1  # See module's doc
    separate_diagonal: bool = False  # [Gram, appendix C]
    calib_labels: LabelKindLike = "pred"  # [Gram]

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        check(0 < self.cut_off < 0.5)  # noqa: PLR2004 (magic value 0.5)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, calib_labels_true: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data."""
        out = self.rnet(calib_data)  # Records activations
        n_samples, n_classes = out.shape

        calib_labels_kind = LabelKind(self.calib_labels)
        if calib_labels_kind == LabelKind.PRED:
            calib_labels = out.argmax(1)
        elif calib_labels_kind == LabelKind.TRUE:
            calib_labels = calib_labels_true
        else:
            msg = f"Unsupported calib_labels_kind: {self.calib_labels!r}"
            raise ValueError(msg)

        self.layers_low_high = []
        self.layers_expected_deviation = []
        for activations in self.activations():
            stats = self.layer_stats(activations)
            if stats.dtype is torch.half:
                # Necessary upcast... github.com/pytorch/pytorch/issues/103054
                stats = stats.float()

            q = torch.tensor([self.cut_off, 1 - self.cut_off]).to(stats)
            # Shape of low_high is (n_classes, 2, stat_dim).
            # Second axis encodes [low, high].
            init = torch.tensor([torch.inf, -torch.inf]).to(stats)
            low_high = init.reshape(1, 2, 1).repeat(n_classes, 1, stats.shape[1])
            # [Gram]: Computed "irrespective of the class assigned"
            calib_deviation = torch.tensor(0.0).to(stats)
            for c in range(n_classes):
                c_stats = stats[calib_labels == c]
                if len(c_stats):
                    low_high[c] = c_stats.nanquantile(q, dim=0)

                calib_deviation += self.deviation(low_high[c], c_stats).sum()

            self.layers_low_high.append(low_high)
            self.layers_expected_deviation.append(calib_deviation / n_samples)

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs)  # Records activations
        n_samples, n_classes = out.shape

        total_deviations = torch.zeros_like(out)
        for activations, low_high, expected_deviation in zip(
            self.activations(),
            self.layers_low_high,
            self.layers_expected_deviation,
            strict=True,
        ):
            stats = self.layer_stats(activations)
            for c in range(n_classes):
                total_deviations[:, c] += (
                    self.deviation(low_high[c], stats) / expected_deviation
                )

        conformity = -total_deviations.nan_to_num(nan=torch.inf)
        return out, conformity

    def layer_stats(self, activations: Tensor) -> Tensor:
        r"""Concatenated upper Gram coefficients of all orders.

        Operates on the given layer's batched activations.

        Arguments
        ---------
        activations: ``Tensor``
            Shape ``(n_samples, *sample_activations_shape)``.
            Per-sample activations must be :math:`1`\ D, :math:`2`\ D or
            :math:`3`\ D tensors, the first two cases being treated as
            implicitly single-channel.

        Returns
        -------
        out: ``Tensor``
            Shape ``(n_samples, max_gram_order * n_channels ** 2 / 2)``.

        Raises
        ------
        :exc:`ValueError`
            If not ``2 <= activations.ndim <= 4``.

        """
        if not 2 <= (ndim := activations.ndim) <= 4:  # noqa: PLR2004 (magic values 2, 4)
            msg = (
                f"{type(self).__qualname__} score only supports per-sample activations "
                f"as 1D, 2D or 3D tensors (got {ndim - 1}D)"
            )
            raise ValueError(msg)

        n_channels = activations.size(1) if ndim == 4 else 1  # noqa: PLR2004 (magic value 4)
        per_channel = activations.shape[ndim // 2 :].numel()
        activations_3d = activations.reshape(len(activations), n_channels, per_channel)

        nan = torch.nan
        all_correlations = []
        for p in range(1, self.max_gram_order + 1):
            Fp = activations_3d**p  # noqa: N806 (uppercase Fp)
            Gp = Fp @ Fp.mT  # noqa: N806 (uppercase Gp)
            G = Gp.abs().nan_to_num(nan=nan, posinf=nan) ** (1 / p)  # noqa: N806 (uppercase G)
            # Sum row-wise for computational reasons mentioned in [Gram]
            # Diagonal correlations may be taken separately
            if self.separate_diagonal:
                all_correlations += [G.diagonal(dim1=1, dim2=2), G.triu(1).sum(2)]
            else:
                all_correlations.append(G.triu().sum(2))

        return torch.cat(all_correlations, dim=1)

    @staticmethod
    def deviation(low_high: Tensor, stats: Tensor) -> Tensor:
        r"""Deviation from :cite:`Gram`, patched as in OODEEL.

        Infinite or ``nan`` elements are treated as ``0``. This monkey
        patch is mentioned in module's doc.

        Arguments
        ---------
        low_high: ``Tensor``
            Shape ``(2, stat_dim)``, first axis encodes ``(low, high)``.
        stats: ``Tensor``
            A batch of stats to test of shape ``(n_samples, stat_dim)``.

        Returns
        -------
        out: ``Tensor``
            Deviations. Shape ``(n_samples,)``.

        """
        low, high = low_high
        low_dev = (low - stats) / low.abs()
        high_dev = (stats - high) / high.abs()

        dev_elts = low_dev.where(stats < low, high_dev.where(stats > high, 0))
        return dev_elts.nan_to_num(nan=0, posinf=0, neginf=0).sum(1)
