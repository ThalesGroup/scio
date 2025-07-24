"""Fixtures for scio scio miscellaneous utils."""

import gc
from collections.abc import Callable
from itertools import product
from time import perf_counter
from types import MappingProxyType
from typing import Any

import pytest
import torch
from torch import Tensor

from scio.scores import BaseScore
from scio.utils import ScoreTimer, ScoreTimerOperation
from scio.utils.misc import ScoreTimerStat, format_time


class SleeperScore(BaseScore):
    """Score with explicit execution timings."""

    __tensor = torch.tensor(0)
    calib_sleep: float
    infer_sleep: float

    def calibrate(self, _calib_data: Tensor, _calib_labels: Any) -> None:  # noqa: ANN401 (Any)
        """Calibrate the score with In-Distribution data."""
        start = perf_counter()
        while perf_counter() < start + self.calib_sleep:
            pass

    def get_conformity(self, _inputs: Tensor) -> tuple[Tensor, Any]:
        """Compute output and associated conformity at inference."""
        start = perf_counter()
        while perf_counter() < start + self.infer_sleep:
            pass
        return self.__tensor, ()


@pytest.fixture(
    scope="module",
    params=[
        ((0.1, 0.1), (0.1234, 0.1010)),
        ((0.1, 0.1),),
        (),
    ],
)
def all_sleeps(request) -> tuple[tuple[float, float], ...]:
    """Parametrize ``(calib_sleep, infer_sleep)`` sequences."""
    return request.param


@pytest.fixture(scope="module")
def ops() -> tuple[tuple[ScoreTimerOperation, int], ...]:
    """Define ``(op, n_samples)`` sequence."""
    return (
        (ScoreTimerOperation.CALIBRATION, 1),
        (ScoreTimerOperation.INFERENCE, 2),
        (ScoreTimerOperation.INFERENCE, 3),
        (ScoreTimerOperation.CALIBRATION, 4),
        (ScoreTimerOperation.INFERENCE, 5),
    )


@pytest.fixture(scope="module")
def expected_stats(all_sleeps, ops) -> tuple[ScoreTimerStat, ...]:
    """Generate expected :attr:`stats` from ``sleeps`` and ``ops``."""
    return tuple(
        ScoreTimerStat(
            op=op,
            n_samples=n_samples,
            duration=infer_sleep
            if op == ScoreTimerOperation.INFERENCE
            else calib_sleep,
            params=MappingProxyType(
                SleeperScore(calib_sleep=calib_sleep, infer_sleep=infer_sleep).params,
            ),
        )
        for (calib_sleep, infer_sleep), (op, n_samples) in product(all_sleeps, ops)
    )


@pytest.fixture(scope="module")
def expected_calibration(expected_stats) -> tuple[ScoreTimerStat, ...]:
    """Generate expected :attr:`calibration` from ``sleeps`` and ``ops``."""
    return tuple(
        stat for stat in expected_stats if stat.op == ScoreTimerOperation.CALIBRATION
    )


@pytest.fixture(scope="module")
def expected_inference(expected_stats) -> tuple[ScoreTimerStat, ...]:
    """Generate expected :attr:`inference` from ``sleeps`` and ``ops``."""
    return tuple(
        stat for stat in expected_stats if stat.op == ScoreTimerOperation.INFERENCE
    )


@pytest.fixture(scope="module")
def score_timer(ops, all_sleeps) -> ScoreTimer:
    """Make :class:`ScoreTimer` instance with controlled timings.

    Its target is dead and collected.
    """
    score = SleeperScore()
    for (calib_sleep, infer_sleep), (op, n_samples) in product(all_sleeps, ops):
        score.calib_sleep = calib_sleep
        score.infer_sleep = infer_sleep

        if op == ScoreTimerOperation.CALIBRATION:
            score.fit(..., torch.rand(n_samples), ...)
        else:
            assert op == ScoreTimerOperation.INFERENCE
            score(torch.rand(n_samples))

    timer = score.timer
    del score
    gc.collect()

    return timer


@pytest.fixture
def assert_close_stats() -> Callable[[ScoreTimerStat, ScoreTimerStat], None]:
    """Provide "assertclose" function for stat instances."""

    def inner(observed: ScoreTimerStat, expected: ScoreTimerStat) -> None:
        """Assert that stats are equal up to same formatted duration."""
        assert type(observed) is type(expected) is ScoreTimerStat
        assert observed.op == expected.op
        assert observed.n_samples == expected.n_samples
        assert observed.params == expected.params
        assert format_time(observed.duration, num_digits=3) == format_time(
            expected.duration,
            num_digits=3,
        )

    return inner


FORMAT_TIME_EXPLICIT = (
    (1.5, "1.500 s"),
    (0.56789, "567.9 ms"),
    (0.99995, "1.000 s"),
    (0.12345, "123.4 ms"),
    (1234, "1234. s"),
    (12345, "12345 s"),
    (0, "0.000 as"),
    (5.67e-20, "0.057 as"),
)
