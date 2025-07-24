"""Complementary tests for DeepMahalanobis score."""

import pytest
import torch

from scio.scores import DeepMahalanobis


def test_compute_precision_handles_singular_covariance(dtype_device):
    """Check that singular covariance is handled."""
    residues = torch.tensor([[1, 2], [2, 4], [4, 8]], **dtype_device)
    DeepMahalanobis.compute_precision(residues)


@pytest.mark.parametrize(
    ("attr", "values"),
    [
        ("epsilon", (0, 1)),
        ("fgm_norm", (1, 2)),
        ("weights", (None, [1, 2])),
    ],
)
def test_specific_attrs_do_not_unfit(
    rnet,
    calib_data,
    calib_labels,
    test_data,
    attr,
    values,
):
    """Check that changing these attrs does not unfit."""
    rnet.record((1, 2), (1, 5))
    score = DeepMahalanobis().fit(rnet, calib_data, calib_labels)
    for value in values:
        setattr(score, attr, value)
        score(test_data)
