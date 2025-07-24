"""Test scio miscellaneous utils."""

import gc
import re
from contextlib import ExitStack
from difflib import SequenceMatcher
from functools import partial

import pytest
import torch

import scio
from scio.utils import ScoreTimer
from scio.utils.misc import check, format_time

from .conftest import FORMAT_TIME_EXPLICIT, SleeperScore


def test_check(null):
    """Test :func:``check`` error message."""
    msg = repr(null)
    with pytest.raises(AssertionError, match=f"^{re.escape(msg)}$"):
        check(condition=False, message=repr(null))


def test_timer_target(make_weakable):
    """Check :attr:`target` works even with dead object."""
    null = make_weakable()
    timer = ScoreTimer(null)

    assert timer.target is null

    del null
    gc.collect()
    assert timer.target is None


def test_timer_context_requires_object_alive(make_weakable):
    """Check raises on context use with dead object."""
    timer = ScoreTimer(make_weakable())
    gc.collect()
    timer_context = timer("inference", n_samples=0)

    msg = "Target 'Weakable' object is dead"
    with (
        pytest.raises(RuntimeError, match=f"^{re.escape(msg)}$"),
        ExitStack() as stack,
    ):
        stack.enter_context(timer_context)


def test_timer_context_requires_params_attribute(make_weakable):
    """Check raises on context use without :attr:`params` attribute."""
    target = make_weakable()
    timer = ScoreTimer(target)
    gc.collect()
    timer_context = timer("inference", n_samples=0)

    msg = (
        f"Target {target} is missing a required 'params' attribute. Hint: "
        f"{type(timer).__qualname__} instances should only target Score instances"
    )
    with (
        pytest.raises(AttributeError, match=f"^{re.escape(msg)}$"),
        ExitStack() as stack,
    ):
        stack.enter_context(timer_context)


@pytest.mark.parametrize("attr", ["stats", "calibration", "inference"])
def test_timer_stats(request, attr, score_timer, assert_close_stats):
    """Compare captured satts with expectations."""
    observed_attr = getattr(score_timer, attr)
    expected_attr = request.getfixturevalue(f"expected_{attr}")
    for observed, expected in zip(observed_attr, expected_attr, strict=True):
        assert_close_stats(observed, expected)


def test_score_timer_report(monkeypatch, score_timer, match_outerr):  # noqa: ARG001 (unused argument)
    """Check the report console output."""
    # Show only 3 digits for reproducibility
    format_time_3 = partial(format_time, num_digits=3)
    monkeypatch.setattr(scio.utils.misc, "format_time", format_time_3)

    score_timer.report


def test_score_timer_report_contains_id_when_alive(capsys):
    """Check report contains id when target is alive."""
    score = SleeperScore(calib_sleep=0.1, infer_sleep=0)
    timer = score.fit(..., torch.tensor([0]), ...).timer
    id_line = f"at {hex(id(score))}"

    # Collect striped lines from report with, score both alive and dead
    timer.report
    report_alive = capsys.readouterr().out
    del score
    gc.collect()
    timer.report
    report_dead = capsys.readouterr().out

    # Use ``difflib`` to match up to ``id_line`` diff
    sm = SequenceMatcher(None, report_alive, report_dead)
    (op0, *_), (op1, a, b, c, d), (op2, *_) = sm.get_opcodes()
    assert op0 == op2 == "equal"
    assert op1 == "delete"
    assert a == c == d
    assert report_alive[a:b].strip() == id_line


@pytest.mark.parametrize(("duration", "expected"), FORMAT_TIME_EXPLICIT)
def test_format_time_valid(duration, expected):
    """Test :func:`format_time` for explicit values."""
    assert format_time(duration) == expected


def test_format_time_out_of_bounds():
    """Check raises on ``duration`` out of bounds."""
    assert format_time(99999.49999) == "99999 s"

    msg = "Duration out of bounds"
    with pytest.raises(AssertionError, match=f"^{re.escape(msg)}$"):
        format_time(99999.5)


def test_format_time_num_digits():
    """Check raises if ``num_digits`` too low."""
    msg = "Display at least 3 digits"
    with pytest.raises(AssertionError, match=f"^{re.escape(msg)}$"):
        format_time(0, num_digits=2)
