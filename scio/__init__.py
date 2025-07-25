"""scio package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scio-pypi")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

import lazy_loader as lazy

# Lazily load from adjacent `.pyi` file
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
