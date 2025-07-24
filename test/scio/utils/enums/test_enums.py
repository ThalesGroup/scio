"""Test scio enums."""

import re
from inspect import signature

import pytest


def test_enum_with_explicit_support_signature(enum):
    """Test our enums signature."""
    assert repr(signature(enum)) == "<Signature (value)>"


def test_enum_with_explicit_support_missing_msg(enum):
    """Test error message on invalid call."""
    value = "invalid"
    msg = f"{value!r} is not a valid {enum.__qualname__}. Supported: 'test'"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        enum(value)
