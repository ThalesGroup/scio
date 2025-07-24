"""Test many hyper-parameters combination."""

from itertools import product
from pathlib import Path

import pytest
import torch
from ruamel.yaml import YAML
from torch import Tensor

import scio.scores
from scio.scores import BaseScoreClassif

from .conftest import N_CLASSES


class CustomObjectsLoader:
    """YAML loader for custom python objects."""

    yaml_tag = "!custom"

    @classmethod
    def from_yaml(cls, _constructor, node) -> object:
        """Define loading behaviour."""
        return getattr(cls, node.value)

    @staticmethod
    def callable_squeezer(inputs: Tensor) -> Tensor:
        """Callable squeezer for FeatureSqueezing."""
        return inputs.roll(1)


def load_combinations() -> list[tuple[BaseScoreClassif, dict]]:
    """Load combinations of `Score` and `params` from yaml config."""
    yaml = YAML()
    yaml.register_class(CustomObjectsLoader)
    yaml_dict = dict(yaml.load(Path(__file__).with_suffix(".yaml")))
    yaml_dict.pop(None)  # Remove yaml anchors
    combinations: list = []
    for Score_str, params_lists in yaml_dict.items():
        Score = getattr(scio.scores, Score_str.partition("/")[0])
        all_params = [
            dict(zip(params_lists, values, strict=False))
            for values in product(*params_lists.values())
        ]
        combinations.extend(product([Score], all_params))
    return combinations


def idfn(Score_or_params: BaseScoreClassif | dict) -> str:
    """Get better ID for params."""

    def to_str(obj: object) -> str:
        """Name or repr."""
        return getattr(obj, "__name__", repr(obj))

    if isinstance(Score_or_params, dict):
        params = Score_or_params
        return f"{' | '.join(f'{k}={to_str(v)}' for k, v in params.items())}"

    return to_str(Score_or_params)


@pytest.mark.parametrize(("Score", "params"), load_combinations(), ids=idfn)
def test_score_fit_and_infer(
    Score,
    params,
    rnet,
    calib_data,
    calib_labels,
    test_data,
    test_out_expected,
    atol_rtol,
):
    """Test ``Score(**params).fit(...)`` and inference work."""
    score = Score(**params).fit(rnet, calib_data, calib_labels)
    test_out, test_conformity = score(test_data)

    assert test_out.shape == test_conformity.shape == (len(test_data), N_CLASSES)
    assert test_out.device == test_conformity.device == test_data.device
    torch.testing.assert_close(test_out, test_out_expected, **atol_rtol)
    assert not test_conformity.isnan().any()
