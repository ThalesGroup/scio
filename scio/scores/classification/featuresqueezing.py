"""Feature squeezing for classification."""

__all__ = ["FeatureSqueezing"]

from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import torch
from scipy.ndimage import (  # type: ignore[import-untyped]  # github.com/scipy/scipy/issues/21971
    vectorized_filter,
)
from torch import Tensor

from scio.scores.utils import get_aggregator
from scio.utils import AggrNameLike, check

from .base import BaseScoreClassif

type Squeezer = Callable[[Tensor], Tensor]
type SqueezerLike = str | Squeezer


class FeatureSqueezing(BaseScoreClassif):
    r"""Feature squeezing for classification.

    Arguments
    ---------
    squeezers: ``SqueezerLike | Iterable[SqueezerLike, ...]``
        One or multiple "squeezers". We define
        ::

            type Squeezer = Callable[[Tensor], Tensor]
            type SqueezerLike = str | Squeezer

        where callable squeezers should conserve the shape of their
        input. A squeezer of type ``str`` denotes one of the built-in
        squeezers:

        - ``"bits:n"`` where ``n`` is replaced with an ``int``. This
          maps data onto a ``n``-bits discretized representation.
        - ``"median:s"`` were ``s`` is replaced with an ``int``.
          Performs median pooling with size ``s`` (or ``(s, s)``). Works
          only for :math:`1`\ D, :math:`2`\ D or :math:`3`\ D samples.
          For :math:`3`\ D samples, operates per channel, with
          channel-first convention.

        Use ``"median:2"`` to match :cite:`FeatCW`.
    dist_norm: ``float``
        Order of the vector norm used to compute the distance between
        natural and squeezed softmax outputs. Defaults to ``1.0``.
    aggregation: ``AggrNameLike | float``
        When several squeezers are used, how to aggregate their
        corresponding nonconformity scores. See
        :class:`~scio.utils.AggrName`. Defaults to ``"max"``.

    References
    ----------
    .. bibliography::
       :filter: false

       Feat
       FeatCW

    """

    squeezers: SqueezerLike | Iterable[SqueezerLike]
    dist_norm: float = 1.0  # [Feat]
    aggregation: AggrNameLike | float = "max"

    def _check_params(self, _n_calib: int) -> None:
        super()._check_params(_n_calib)
        self.squeezers_fn = self.get_squeezers()
        check(len(self.squeezers_fn) > 0)

    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """No calibration needed."""

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs, dont_record=True)
        distances = torch.empty(
            len(out),
            len(self.squeezers_fn),
            dtype=out.dtype,
            device=out.device,
        )
        for i, squeezer in enumerate(self.squeezers_fn):
            out_squeezed = self.rnet(squeezer(inputs), dont_record=True)
            diff = out.softmax(1) - out_squeezed.softmax(1)
            distances[:, i] = torch.linalg.vector_norm(diff, ord=self.dist_norm)

        # Aggregation
        conformity = -get_aggregator(self.aggregation)(distances, dim=1)
        return out, conformity

    def get_squeezers(self) -> tuple[Squeezer, ...]:
        """Preprocessing to provide tuple of callable squeezers.

        Returns
        -------
        out: ``tuple[Squeezer, ...]``
            The callable squeezers to use, extracted from
            :attr:`self.squeezers`.

        Raises
        ------
        :exc:`TypeError`
            If a custom squeezer is not callable.
        :exc:`ValueError`
            If a ``str`` squeezer is unsupported.

        """
        if isinstance(self.squeezers, Iterable) and not isinstance(self.squeezers, str):
            squeezers_iterable = self.squeezers
        else:
            squeezers_iterable = (self.squeezers,)

        out = []
        for squeezer in squeezers_iterable:
            # Custom squeezer
            if not isinstance(squeezer, str):
                if not callable(squeezer):
                    msg = f"Custom squeezers must be callable (got {squeezer!r})"
                    raise TypeError(msg)

                out.append(squeezer)
                continue

            # Built-in squeezers
            if squeezer.startswith("bits:"):
                n_bits = int(squeezer[5:].strip())
                out.append(partial(self.bits_squeezing, n_bits=n_bits))
            elif squeezer.startswith("median:"):
                size = int(squeezer[7:].strip())
                out.append(partial(self.median_squeezing, size=size))
            else:
                msg = f"Squeezer {squeezer!r} is not supported"
                raise ValueError(msg)

        return tuple(out)

    @staticmethod
    def bits_squeezing(inputs: Tensor, *, n_bits: int) -> Tensor:
        """Squeeze ``inputs`` into ``n_bits`` representation.

        Arguments
        ---------
        inputs: ``Tensor``
            The inputs to be squeezed into ``n_bits`` representation.
        n_bits: ``int``
            The number of bits used to encode :math:`[0, 1)` data. Must
            satisfy ``0 < n_bits < 32``.

        Returns
        -------
        squeezed: ``Tensor``
            ``inputs`` rounded to the closest bin center, for bins of
            width ``1 / 2**n_bits``.

        Raises
        ------
        :exc:`ValueError`
            If not ``0 < n_bits < 32``.

        Note
        ----
        Note that ``1`` is rounded *up* to ``1 + 1 / 2**(n_bits + 1)``.

        """
        if not 0 < n_bits < 32:  # noqa: PLR2004 (magic value 32)
            msg = f"Bits squeezing requires 0<n_bits<32 (got {n_bits!r})"
            raise ValueError(msg)

        scale = 2**n_bits
        return ((inputs * scale).round() + 0.5) / scale

    @staticmethod
    def median_squeezing(
        inputs: Tensor,
        *,
        size: int,
        channel_first: bool = True,
    ) -> Tensor:
        r"""Median pooling for batched samples (of max :math:`3`\ D).

        Raises
        ------
        :exc:`ValueError`
            If ``size < 1``.
        :exc:`ValueError`
            If not ``2 <= inputs.ndim <= 4``.

        Note
        ----
        Channel first convention is used by default for :math:`3`\ D
        samples.

        """
        if size < 1:
            msg = f"Median squeezing requires a positive size (got {size!r})"
            raise ValueError(msg)

        axes: tuple[int, ...]
        ndim = inputs.ndim - 1
        if ndim == 1:
            axes = (1,)
        elif ndim == 2:  # noqa: PLR2004 (magic value 2)
            axes = (1, 2)
        elif ndim == 3:  # noqa: PLR2004 (magic value 3)
            axes = (2, 3) if channel_first else (1, 2)
        else:
            msg = f"Individual samples must be 1D, 2D or 3D (got {ndim}D)"
            raise ValueError(msg)

        inputs_np = inputs.numpy(force=True)
        median = vectorized_filter(inputs_np, np.median, size=size, axes=axes)
        return torch.from_numpy(median).to(inputs)
