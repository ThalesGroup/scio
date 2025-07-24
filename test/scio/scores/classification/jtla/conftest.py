"""Fixtures for JTLA tests."""

import pytest

from scio.scores import JTLA


@pytest.fixture
def score(rnet, calib_data, calib_labels) -> JTLA:
    """Prepare a fit JTLA instance."""
    test = {"type": "multinomial", "mode": "mle", "k": 1}
    return JTLA(test=test).fit(rnet, calib_data, calib_labels)
