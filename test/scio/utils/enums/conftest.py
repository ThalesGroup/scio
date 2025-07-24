"""Fixtures for scio enums."""

from enum import EnumType

import pytest

from scio.utils.enums import EnumWithExplicitSupport


@pytest.fixture
def enum() -> EnumType:
    """Make a sample enum with explicit support for tests."""

    class EnumFactory(EnumWithExplicitSupport):
        """Test enum with only ``TEST = "test"`` member."""

        __slots__ = ()
        TEST = "test"

    return EnumFactory
