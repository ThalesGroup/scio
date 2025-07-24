"""Fixtures for bench_classif tests."""

from types import MappingProxyType

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor

from scio.eval import (
    ROC,
    BaseDiscriminativePower,
    compute_confidence,
    compute_metrics,
    fit_scores,
)
from scio.eval.classification.benchmark import ScoreClassifAndLayers
from scio.scores import BaseScoreClassif
from test.scio.conftest import Net3DFactory

# All of these constants are essentially arbitrary
OPS = (torch.var, torch.amax, torch.mean)
ALL_LAYERS = (((1, 1),), (), ((1, 2), (1, 3), (1, 5)))
N_CLASSES = 3
N_CALIB = 10
N_TEST_IND = 10
N_TEST_OODS = (8, 12)
OODS_TITLE = ("Test OoD 1", "Test OoD 2")
BASELINE = 0
HIST_KW = MappingProxyType({"common_bins": False})


class ScoreFactory(BaseScoreClassif):
    """ScoreFactory for bench_classif.py tests.

    Score with predictible explicit behaviour regarding inputs
    processing, for test purposes.

    Arguments
    ---------
    post_init_arg: ``list[object]`` (positional-only)
        Must be ``[op]`` with ``op`` is a torch operator with signature
        ``Tensor -> scalar``, that must accept ``dim`` argument to
        operate over ``dim``..

    """

    __test__ = False

    def __post_init__(self, op, /) -> None:
        """Store :attr:`op` as nonparameter to avoid ``id`` in repr."""
        self.op = op

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, _calib_labels: Tensor) -> None:
        """Observe :attr:`op` on calibration data plus net output."""
        out = self.rnet(calib_data)
        self.observed = self.op(calib_data) + self.op(out)

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Confidence is net output."""
        out = self.rnet(inputs)
        return out, self.op(out, dim=1)


@pytest.fixture
def scores() -> tuple[BaseScoreClassif, ...]:
    """Test scores (see :class:`ScoreFactory`)."""
    return tuple(ScoreFactory([op]) for op in OPS)


@pytest.fixture
def scores_and_layers(scores) -> tuple[ScoreClassifAndLayers, ...]:
    """Test scores + layers."""
    return tuple(
        (score, layers) for score, layers in zip(scores, ALL_LAYERS, strict=True)
    )


@pytest.fixture
def net(dtype_device_gen) -> Net3DFactory:
    """Create toy network for 3D samples."""
    return Net3DFactory(n_classes=N_CLASSES, **dtype_device_gen)


@pytest.fixture
def calib_data(net, dtype_device_gen) -> Tensor:
    """Generate random calibration data."""
    return torch.rand(N_CALIB, *net.sample_shape, **dtype_device_gen)


@pytest.fixture
def calib_labels(calib_data, generator) -> Tensor:
    """Generate random calibration labels.

    All classes except one are represented, on purpose.
    """
    all_but_one = torch.arange(N_CLASSES - 1, device=calib_data.device)
    remaining = torch.randint(
        0,
        N_CLASSES - 1,
        (N_CALIB - N_CLASSES + 1,),
        device=calib_data.device,
        generator=generator,
    )
    return torch.cat([all_but_one, remaining])


@pytest.fixture(params=[True, False])
def show_progress(request) -> bool:
    """Parametrize ``show_progress``."""
    return request.param


@pytest.fixture
def scores_fit(
    scores_and_layers,
    net,
    calib_data,
    calib_labels,
) -> list[BaseScoreClassif]:
    """Test output from ``fit_scores``.

    Note
    ----
    In principle, should be used after having tested ``fit_scores`` for
    better bug reports, but not critical.

    """
    return fit_scores(
        scores_and_layers,
        net,
        calib_data,
        calib_labels,
        show_progress=False,
    )


@pytest.fixture
def ind(net, dtype_device_gen) -> Tensor:
    """Generate random ind data."""
    return torch.rand(N_TEST_IND, *net.sample_shape, **dtype_device_gen)


@pytest.fixture
def oods(net, dtype_device_gen) -> tuple[Tensor, ...]:
    """Generate random ood data."""
    return tuple(
        torch.rand(n_test_ood, *net.sample_shape, **dtype_device_gen)
        for n_test_ood in N_TEST_OODS
    )


@pytest.fixture
def confs(
    scores_fit,
    ind,
    oods,
) -> tuple[NDArray[np.floating], tuple[NDArray[np.floating], ...]]:
    """Test output from ``compute_confidence``.

    Note
    ----
    In principle, should be used after having tested
    ``compute_confidence`` for better bug reports, but not critical.

    """
    return compute_confidence(scores_fit, ind=ind, oods=oods, show_progress=False)


@pytest.fixture
def confs_ind(confs) -> NDArray[np.floating]:
    """First output of ``compute_confidence`` on fixture data."""
    return confs[0]


@pytest.fixture
def confs_oods(confs) -> tuple[NDArray[np.floating], ...]:
    """Second output of ``compute_confidence`` on fixture data."""
    return confs[1]


class MetricMaxPosFactory(BaseDiscriminativePower):
    """Test metric to check inputs processing."""

    def from_roc(self, roc: ROC) -> float:
        """Trick, should return highest positive sample's score."""
        return roc.thresholds[-1][0]


class MetricMinNegFactory(BaseDiscriminativePower):
    """Test metric to check inputs processing."""

    def from_roc(self, roc: ROC) -> float:
        """Trick, should return lowest negative sample's score."""
        return roc.thresholds[0][1]


@pytest.fixture
def metrics() -> tuple[BaseDiscriminativePower, ...]:
    """Test metrics for ``compute_metrics`` and others."""
    return (MetricMaxPosFactory(), MetricMinNegFactory())


@pytest.fixture
def evals(confs_ind, confs_oods, metrics) -> NDArray[np.floating]:
    """Test output from ``compute_metrics``.

    Note
    ----
    In principle, should be used after having tested ``compute_metrics``
    for better bug reports, but not critical.

    """
    return compute_metrics(confs_ind, confs_oods, metrics)
