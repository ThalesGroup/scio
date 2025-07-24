"""Package for confidence scores in classification."""

import lazy_loader as lazy

# Lazily load from adjacent `.pyi` file
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
