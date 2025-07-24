"""Test recorder conftest."""

import re

import pytest


@pytest.mark.parametrize("layer", [0, (1, 6)])
def test_layer_func_raises_on_invalid(layer_func, layer):
    """Test that ``layer_func`` fixture raises on invalid layer."""
    msg = f"Invalid layer: {layer}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        layer_func(layer)
