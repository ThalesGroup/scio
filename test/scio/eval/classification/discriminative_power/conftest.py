"""Fixtures for discriminative_power tests."""

from collections.abc import Callable
from math import sqrt
from types import SimpleNamespace

import pytest

from scio.eval import AUC, MCC, ROC, TNR, TPR, BaseDiscriminativePower


@pytest.fixture
def make_dp_check_params_tester() -> Callable[[object], BaseDiscriminativePower]:
    """Make Discriminative Power tester factory."""

    def inner(null: object, val: float = 0.0) -> BaseDiscriminativePower:
        """Make a :meth:`_check_params` tester."""

        class DiscriminativePowerCheckParamsFactory(BaseDiscriminativePower):
            """Excpect ``obj`` parameter to be set to ``null``."""

            obj: object

            @staticmethod
            def _check_params(params: SimpleNamespace) -> None:
                if params.obj is not null:
                    msg = f"Expected: {null!r}. Got: {params.obj!r}"
                    raise ValueError(msg)

            def from_roc(self, _roc: ROC) -> float:
                return val

        return DiscriminativePowerCheckParamsFactory

    return inner


T, F = True, False
LABELS_AND_SCORES = (
    ((F, T, T, T, F), (0, 1, 2, 3, 3)),
    ((T, F, F, T, F), (0, 1, 2, 3, 4)),
    ((T, F), (0, 1)),
    ((F, T), (0, 1)),
)
METRICS = (
    AUC(kind="pessimistic"),
    AUC(kind="convex_hull"),
    AUC(kind="convex_hull", min_fpr=1 / 4, max_fpr=3 / 4),
    MCC(),
    TPR(kind="pessimistic", min_tnr=1 / 4),
    TPR(kind="convex_hull", max_fpr=1 / 4),
    TNR(kind="pessimistic", min_tpr=1 / 4),
    TNR(kind="convex_hull", max_fnr=1 / 4),
)
# Scenarios in columns, metrics in line. Compute MANUALLY!
EXPECTED = (
    # AUC
    (1 / 3, 2 / 3, 1.0, 0.0),
    (7 / 12, 5 / 6, 1.0, 0.5),
    (5 / 8, 167 / 192, 1.0, 0.5),
    # MCC
    (1 / 6, sqrt(3 / 8), 1.0, 0.0),
    # TPR
    (2 / 3, 1.0, 1.0, 0.0),
    (1 / 3, 11 / 16, 1.0, 0.25),
    # TNR
    (1 / 2, 1.0, 1.0, 0.0),
    (3 / 8, 2 / 3, 1.0, 0.25),
)
