"""Complementary tests for JTLA score."""

import re

import pytest

from scio.utils import ClassAggr


@pytest.mark.parametrize(
    ("attr", "values"),
    [
        ("class_aggregation", ("nat", "adv")),
        ("layer_aggregation_consecutive", (True, False)),
    ],
)
def test_specific_attrs_do_not_unfit(score, attr, values, test_data):
    """Check that changing these attrs does not unfit."""
    for value in values:
        setattr(score, attr, value)
        score(test_data)


def test_raises_unsupported_zaggr(
    monkeypatch_enum_new_member,
    score,
    test_data,
):
    """Check raise on unexpected :class:`ZAggr` value."""
    value = monkeypatch_enum_new_member(ClassAggr)
    score.class_aggregation = value

    msg = f"Unsupported class aggregation: {value!r}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        score(test_data)
