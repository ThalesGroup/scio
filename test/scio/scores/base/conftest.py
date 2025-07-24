"""Fixtures for base score tests."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor

from scio.recorder import Recorder
from scio.scores import BaseScore
from test.scio.conftest import Net3DFactory

N_SAMPLES = 10


@pytest.fixture
def rnet(dtype, device) -> Recorder:
    """Sample recorder net."""
    net = Net3DFactory(dtype=dtype, device=device)
    input_size = (1, *net.sample_shape)
    return Recorder(net, input_size=input_size, dtypes=[dtype], device=device)


@pytest.fixture
def shape(rnet) -> tuple[int, ...]:
    """Shape of samples used for forward passes."""
    return (N_SAMPLES, *rnet.net.sample_shape)


@pytest.fixture
def tensor(shape, dtype_device) -> Tensor:
    """Sample tensor."""
    return torch.rand(shape, **dtype_device)


@pytest.fixture
def make_score() -> Callable[[object, object, object], BaseScore]:
    """Make a score-maker."""

    def inner(
        obj_check_params_: object,
        obj_calibrate_: object,
        obj_get_conformity_: object,
    ) -> BaseScore:
        """Make a score with configured params to be checked."""

        class ScoreFactory(BaseScore):
            """Test score that checks params in associated methods."""

            obj_check_params: object = obj_check_params_
            obj_calibrate: object = obj_calibrate_
            obj_get_conformity: object = obj_get_conformity_
            obj: object

            def _check_params(self, _n_calib: int) -> None:
                super()._check_params(_n_calib)
                assert self.obj_check_params is obj_check_params_, "_check_params"

            def calibrate(self, _calib_data: Tensor, _calib_labels: Any) -> None:  # noqa: ANN401 (Any)
                """Calibrate the score with In-Distribution data."""
                assert self.obj_calibrate is obj_calibrate_, "calibrate"

            def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Any]:
                """Compute output and associated conformity at inference."""
                assert self.obj_get_conformity is obj_get_conformity_, "get_conformity"
                return inputs, ...

        return ScoreFactory()

    return inner


@pytest.fixture
def score(make_score, null) -> BaseScore:
    """Make an unfit, unbound score."""
    score = make_score(null, null, null)
    score.obj = null
    return score


@pytest.fixture
def score_fit(score, rnet, tensor) -> BaseScore:
    """Make a score fit, bound to ``rnet``."""
    return score.fit(rnet, tensor, ...)
