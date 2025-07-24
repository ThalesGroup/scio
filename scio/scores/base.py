"""Module implementing base class for every score."""

__all__ = ["BaseScore"]

from abc import abstractmethod
from collections.abc import Iterable
from functools import partial
from math import isnan
from typing import Self

import torch
from paramclasses import MISSING, ParamClass
from paramclasses import protected as _protected
from torch import Tensor

from scio.recorder import Recorder
from scio.utils import ScoreTimer, ScoreTimerOperation, check

from .utils import normalize_samples


def protected[T](obj: T) -> T:
    """Monkeypatch ``protected`` type problems locally."""
    return _protected(obj)  # type: ignore[return-value]  # github.com/eliegoudout/paramclasses/issues/34


class BaseScore[Labels: Iterable, Conformities: Iterable](ParamClass):
    """Base class for every scoring algorithms.

    A score is defined through:

    1. a calibration procedure, during which In-Distribution data and
       labels may be observed;
    2. an inference procedure.

    These should respectively be carried out in :meth:`calibrate` and
    :meth:`get_conformity`.

    Scores are `paramclasses
    <https://github.com/eliegoudout/paramclasses>`_ and hyperparameters
    are defined as their *parameters*. Optionally, scores may override
    :meth:`_on_param_will_be_set` and :meth:`_check_params` (if so, it
    should call :meth:`super()._check_params`). Refer to source code for
    examples.

    For development, the score should not assume anything about which
    layers are recorded. All that is known is that the
    :meth:`activations` method returns activations following the layers
    recording order (see ``concatenate`` keyword for scores using only
    one layer).

    It is also possible to use on-the-fly activations postprocessing
    during inference (see :meth:`Recorder.forward()
    <scio.recorder.Recorder.forward>`) and access the network's named
    parameters for recorded layers with :attr:`Recorder.recorded_params
    <scio.recorder.Recorder.recorded_params>` (see *e.g.*
    :class:`~scio.scores.classification.gradnorm.GradNorm`).


    For usage, instances should be :meth:`fit` before being called::

        score = Score(param="param")
        score.fit(rnet, calib_data, calib_labels)
        out, conformity = score(test_data)

    Arguments
    ---------
    act_norm: ``float``, optional
        If provided, the recorded activations are normalized for norm
        ``act_norm``. Should not be ``0`` or ``nan``.

    """

    # ============================= TO IMPLEMENT IN SCORES =============================
    @abstractmethod
    def calibrate(self, calib_data: Tensor, calib_labels: Labels) -> None:
        """Calibrate the score with In-Distribution data.

        Arguments
        ---------
        calib_data: ``Tensor``
            Batched calibration samples. Shape ``(n_calib,
            *sample_shape)``.
        calib_labels:  ``Labels``
            Batched labels.

        """

    @abstractmethod
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Conformities]:
        """Compute output and associated conformity at inference.

        Arguments
        ---------
        inputs: ``Tensor``
            Batched data to be processed by the bound neural network.
            Shape ``(n_sample, *sample_shape)``.

        Returns
        -------
        out: ``Tensor``
            The neural network output. Shape ``(n_samples,
            *output_sample_shape)``.
        conformity: ``Conformities``
            Associated conformities.

        """

    def _on_param_will_be_set(self, attr: str, future_val: object) -> None:
        """Optionally define callback on param change (*e.g.* unfit)."""
        if future_val != getattr(self, attr, MISSING):
            self.unfit()

    def _check_params(self, _n_calib: int) -> None:
        """Optionally check params before fit.

        As a rule of thumb, for consistency and conciseness, the type of
        the params should be assumed correct, without any need for
        sanitization.

        This method is called during :meth:`fit` calls. As such, the
        number of calibration samples ``n_calib`` is known and can be
        used. If not, name it ``_n_calib`` per convention.

        Except for base class, it should almost always contain a
        ``super()._check_params(n_calib)`` call.

        Arguments
        ---------
        n_calib: ``int``
            Number of calibration samples.

        Note
        ----
        This method is **not** called before inference. As such, if a
        subclass overrides :meth:`_on_param_will_be_set` in order to
        modify unfitting behaviour, it must carry out consistent
        sanitization.

        """
        if missing := self.missing_params:  # type: ignore[operator]  # github.com/eliegoudout/paramclasses/issues/34
            msg = f"Missing parameters for {type(self).__qualname__} score: {missing}"
            raise RuntimeError(msg)

        if (act_norm := self.act_norm) is None:
            return

        check(act_norm != 0)
        check(not isnan(act_norm))

    # ==================================================================================

    __is_fit = False
    __timer = None
    act_norm: float | None = None

    @protected
    def __call__(self, inputs: Tensor) -> tuple[Tensor, Conformities]:
        """Return bound net output with conformity estimate.

        Arguments
        ---------
        inputs: ``Tensor``
            Input samples to process.

        Returns
        -------
        out: ``Tensor``
            The natural output of the bound neural network.
        conformities: ``Conformities``
            The associated conformity.

        Raises
        ------
        :exc:`RuntimeError`
            If the instance is not fit.

        """
        if not self.__is_fit:
            msg = "The score needs to be fit before inference"
            raise RuntimeError(msg)

        with self.timer(ScoreTimerOperation.INFERENCE, n_samples=len(inputs)):
            return self._process(inputs)

    def _process(self, inputs: Tensor) -> tuple[Tensor, Conformities]:
        """Process inputs. Subclasses may introduce postprocessing."""
        return self.get_conformity(inputs)

    @protected
    def __repr__(self) -> str:
        """Add fit/unfit information to *paramclass* repr.

        Example
        -------
        ::

            >>> KNN()
            KNN[unfit](act_norm=2.0, mode='raw', k=?, index_metric='l2')

        """
        paramclass_repr = ParamClass.__repr__(self)
        classname, sep, tail = paramclass_repr.partition("(")
        status = f"[{'' if self.__is_fit else 'un'}fit]"
        return classname + status + sep + tail

    @protected
    def fit(self, rnet: Recorder, calib_data: Tensor, calib_labels: Labels) -> Self:
        """Fit the score for given ``rnet`` and In-Distribution data.

        Arguments
        ---------
        rnet: ``Recorder``
            Target :class:`Recorder <scio.recorder.Recorder>` net under
            analysis. It becomes bound to the score as ``self.rnet``
            until the next :meth:`fit` call.
        calib_data: ``Tensor``
            Batched calibration samples. Shape ``(n_calib,
            *sample_shape)``.
        calib_labels:  ``Labels``
            Batched labels.

        Returns
        -------
        self: ``Self``
            The fit instance.

        """
        n_samples = len(calib_data)
        self._check_params(n_samples)
        self.__is_fit = False
        self._rnet = rnet
        with self.timer(ScoreTimerOperation.CALIBRATION, n_samples=n_samples):
            self.calibrate(calib_data, calib_labels)
        self.__is_fit = True
        return self

    @protected
    def activations(self, *, concatenate: bool = False) -> tuple[Tensor, ...]:
        r"""Return last recorded activations of bound net.

        During a standard forward pass, :class:`~scio.recorder.Recorder`
        nets record the requested activations. This method retrieves
        those, with additional per-sample normalization if specified by
        :attr:`act_norm`. The normalization occurs layer-wise, unless
        ``concatenate is True``.

        Arguments
        ---------
        concatenate: ``bool``
            If ``True``, flattens every sample at every layer and
            concatenates across layers, in which case ``out`` is a
            :math:`1`\ -tuple of one ``Tensor`` with shape ``(n_samples,
            tot_data_dim)``. This option is handy for scoring techniques
            that only work with one latent space: concatenating across
            layers mimicks having recorded only one layer. Keep in mind
            that it might be preferable to only record one layer in the
            first place, if the technique is designed this way. Defaults
            to ``False``.

        Returns
        -------
        out: ``tuple[Tensor, ...]``
            The last recorded activations. It is a tuple of length the
            number of recorded layers if ``not contatenate``, otherwise
            one. If ``not concatenate``, the shape of samples is
            untouched.

        """
        act_dict = self.rnet.activations

        act_iter: Iterable[Tensor]
        if concatenate:
            act_iter = [torch.cat([t.flatten(1) for t in act_dict.values()], dim=1)]
        else:
            act_iter = act_dict.values()

        normalizer = partial(normalize_samples, ord=self.act_norm)
        return tuple(map(normalizer, act_iter))

    @protected
    def unfit(self) -> None:
        """Unfit the score instance."""
        self.__is_fit = False

    @protected  # type: ignore[prop-decorator]
    @property
    def timer(self) -> ScoreTimer:
        """The :class:`~scio.utils.ScoreTimer` instance for ``self``."""
        if self.__timer is None:
            self.__timer = ScoreTimer(self)

        return self.__timer

    @protected
    def reset_timer(self) -> None:
        """Reset :attr:`timer`."""
        self.__timer = None

    @protected  # type: ignore[prop-decorator]
    @property
    def rnet(self) -> Recorder:
        """:class:`~scio.recorder.Recorder` net bound by :meth:`fit`.

        If :meth:`fit` was never called, raises :exc:`AttributeError`.
        """
        return self._rnet
