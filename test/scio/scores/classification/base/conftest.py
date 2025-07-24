"""Fixtures for classification base score."""

from types import MappingProxyType
from typing import Literal

import pytest
import torch
from torch import Tensor

from scio.scores import BaseScoreClassif


class ScoreClassifFactory(BaseScoreClassif):
    """Score that porentially aggregates conformity overs classes."""

    class_aggregation: Literal["yes", "no", "keepdim"]

    def calibrate(self, _calib_data: Tensor, _calib_labels: Tensor) -> None:
        """Calibrate the score with In-Distribution data."""

    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs)
        if self.class_aggregation == "yes":
            conformity = out.mean(1)
        elif self.class_aggregation == "keepdim":
            conformity = out.mean(1, keepdim=True)
        else:
            conformity = out
        return out, conformity


@pytest.fixture(params=["yes", "no", "keepdim"])
def class_aggregation(request) -> str:
    """Define whether score aggregates over classes."""
    return request.param


@pytest.fixture
def score(class_aggregation, rnet, calib_data, calib_labels) -> BaseScoreClassif:
    """Make a score-maker."""
    score = ScoreClassifFactory(class_aggregation=class_aggregation)
    score.fit(rnet, calib_data, calib_labels)

    return score


POSTPROCESS_CONFORMITY_EXPLICIT = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 2.0, 1.0],
    [1.0, 2.0, 1.0],
])
# Keys are ``mode``. Compute values MANUALLY!
POSTPROCESS_CONFORMITY_OUT = MappingProxyType({
    "raw": torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 1.0],
        [1.0, 2.0, 1.0],
    ]),
    "diff": torch.tensor([
        [-2.0, -1.0, 1.0],
        [0.0, 0.0, -1.0],
        [-1.0, 1.0, -1.0],
    ]),
    "ratio": torch.tensor([
        [1 / 3, 2 / 3, 1.5],
        [1.0, 1.0, 0.5],
        [0.5, 2.0, 0.5],
    ]),
})


@pytest.fixture
def conformity(dtype_device_gen) -> Tensor:
    """Prepare conformity sample for :func:`postprocess_conformity`."""
    return torch.rand(10, 20, **dtype_device_gen)
