"""Complementary tests for Odds score."""

import re

import pytest

from scio.scores import Odds
from scio.utils import ZAggr


def test_raises_unsupported_zaggr(
    monkeypatch_enum_new_member,
    rnet,
    calib_data,
    calib_labels,
    test_data,
):
    """Check raise on unexpected :class:`ZAggr` value."""
    value = monkeypatch_enum_new_member(ZAggr)
    score = Odds(epsilon=0, z_aggregation=value).fit(rnet, calib_data, calib_labels)

    msg = f"Unsupported z-aggregation scheme: {value!r}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score(test_data)
