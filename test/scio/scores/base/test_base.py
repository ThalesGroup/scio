"""Test base score."""

import re
from math import prod
from typing import Any

import pytest
import torch
from torch import Tensor

from scio.scores import BaseScore

from .conftest import N_SAMPLES


def test_needs_fit(score, tensor):
    """Check that calls requires to be fit."""
    msg = "The score needs to be fit before inference"
    with pytest.raises(RuntimeError, match=f"^{re.escape(msg)}$"):
        score(tensor)

    # Call works after fit
    score_fit = score.fit(..., tensor, ...)
    score_fit(tensor)


def test_unfit_on_param_change(score_fit, tensor):
    """Check base :meth:`_check_params` unfits on new param value."""
    score_fit(tensor)

    # Param value change -> unfit
    score_fit.obj = 0
    msg = "The score needs to be fit before inference"
    with pytest.raises(RuntimeError, match=f"^{re.escape(msg)}$"):
        score_fit(tensor)


def test_same_param_does_not_unfit(score_fit, tensor, null):
    """Check base :meth:`_check_params` handles noop assignment."""
    score_fit(tensor)

    # Still fit since param did not change value
    score_fit.obj = null
    score_fit(tensor)


def test_checks_params_on_fit(score, tensor):
    """Check that parameters are checked on :meth:`fit`."""
    score.obj_check_params = 0
    msg = "_check_params"
    with pytest.raises(AssertionError, match=f"^{re.escape(msg)}\n"):
        score.fit(..., tensor, ...)


def test_requires_all_params(score, tensor):
    """Check raise on missing params."""
    del score.obj

    msg = f"Missing parameters for {type(score).__qualname__} score: ('obj',)"
    with pytest.raises(RuntimeError, match=f"^{re.escape(msg)}$"):
        score.fit(..., tensor, ...)


@pytest.mark.parametrize("act_norm", [float("nan"), 0])
def test_sanitizes_act_norm(score, act_norm, tensor):
    """Check ``act_norm`` sanitization (supposer float)."""
    score.act_norm = act_norm

    msg = ""
    with pytest.raises(AssertionError, match=f"^{re.escape(msg)}$"):
        score.fit(..., tensor, ...)


def test_requires_calibrate_and_get_conformity():
    """Check that abstract class requires implementations."""
    msg = (
        "Can't instantiate abstract class BaseScore without an implementation for "
        "abstract methods 'calibrate', 'get_conformity'"
    )
    with pytest.raises(TypeError, match=f"^{re.escape(msg)}$"):
        BaseScore()


def test_process_may_post_process(score_fit, tensor, null):
    """Check that :meth:`_process` is called."""
    out, _ = score_fit(tensor)

    class ScoreWithProcess(type(score_fit)):
        def _process(self, inputs) -> tuple[Tensor, Any]:
            assert self.obj is null, "_process"
            return super()._process(inputs)

    score_process = ScoreWithProcess(obj=null)

    # Inference is the same
    out_process, _ = score_process.fit(..., tensor, ...)(tensor)
    assert out_process is out

    # :meth:`_process` is indeed called
    score_process.obj = 0
    msg = "_process"
    with pytest.raises(AssertionError, match=f"^{re.escape(msg)}\n"):
        score_process.fit(..., tensor, ...)(tensor)


def test_repr(score, null, tensor):
    """Check the ``repr``."""
    expected = (
        f"ScoreFactory[unfit](act_norm=None, obj_check_params={null!r}, "
        f"obj_calibrate={null!r}, obj_get_conformity={null!r}, obj={null!r})"
    )
    assert repr(score) == expected

    score_fit = score.fit(..., tensor, ...)
    expected_fit = expected.replace("unfit", "fit")
    assert repr(score_fit) == expected_fit


def test_fit_uses_calibrate(score, tensor):
    """Check that :meth:`calibrate` is used."""
    score.obj_calibrate = 0

    msg = "calibrate"
    with pytest.raises(AssertionError, match=f"^{re.escape(msg)}\n"):
        score.fit(..., tensor, ...)


def test_call_uses_get_conformity(score, tensor):
    """Check that :meth:`get_conformity` is used."""
    score.obj_get_conformity = 0
    score_fit = score.fit(..., tensor, ...)

    msg = "get_conformity"
    with pytest.raises(AssertionError, match=f"^{re.escape(msg)}\n"):
        score_fit(tensor)


@pytest.mark.parametrize("act_norm", [None, -1, 2, -torch.inf])
def test_activations_and_act_norm(score, act_norm, rnet, tensor, appender):
    """Check the impact of ``concatenate`` and ``act_norm``."""
    score.act_norm = act_norm
    score_fit = score.fit(rnet, tensor, ...)
    N = 5
    rnet.record(*((1, i) for i in range(1, 1 + N)))
    rnet(tensor)  # Records activations
    activations = score_fit.activations()
    activations_cat = score_fit.activations(concatenate=True)

    shapes = tuple(act.shape for act in activations)
    tot_data_dim = sum(prod(shape[1:]) for shape in shapes)

    assert len(activations) == N
    assert len(activations_cat) == 1
    assert activations_cat[0].shape == (N_SAMPLES, tot_data_dim)

    if act_norm is None:
        return

    for i, act in zip(range(N + 1), (*activations, *activations_cat), strict=True):
        norms = torch.linalg.vector_norm(act.flatten(1), ord=act_norm, dim=1)
        ones = torch.ones_like(norms)
        torch.testing.assert_close(norms, ones, msg=appender(f"Loop index: {i}"))


def test_unfit(score_fit, tensor):
    """Test :meth:`unfit`."""
    score_fit(tensor)

    score_fit.unfit()
    msg = "The score needs to be fit before inference"
    with pytest.raises(RuntimeError, match=f"^{re.escape(msg)}$"):
        score_fit(tensor)


def test_timer_reset(score, tensor):
    """Check :attr:`timer` reset with :math:`reset_timer`.

    The functionalities of :attr:`timer` with regards to
    :class:`BaseScore` are tested with :class:`ScoreTimer` tests.
    """
    assert not score.timer.stats

    score.fit(..., tensor, ...)
    assert score.timer.stats

    score.reset_timer()
    assert not score.timer.stats
