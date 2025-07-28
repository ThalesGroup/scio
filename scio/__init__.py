"""scio package."""

import os
import platform
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scio-pypi")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

import lazy_loader as lazy

# Lazily load from adjacent `.pyi` file
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# Monkeypatch the use of ``faiss`` on non-Linux platforms
if (
    platform.system() != "Linux"
    and os.environ.get("KMP_DUPLICATE_LIB_OK", None) != "TRUE"
):  # pragma: no cover
    import importlib
    import sys
    import types
    from warnings import warn

    null = object()

    class DummyFaiss(types.ModuleType):
        """Dummy faiss to intercept faiss loading."""

        __faiss = null

        def __getattribute__(self, attr: str) -> object:
            """Intercept loading, warn & enable KMP_DUPLICATE_LIB_OK."""
            __faiss = super().__getattribute__(f"_{type(self).__name__}__faiss")

            # If necessary, warn and load faiss
            if __faiss is null:
                msg = (
                    "On non-Linux platforms, the use of any `faiss`-dependant feature "
                    "is partially supported by `scio`: you might experience slowdowns "
                    "or even incorrect results! This is due to a non-Python dependency "
                    "conflict regarding OpenMP (see https://pypackaging-native.github"
                    ".io/key-issues/native-dependencies for interesting ressources). "
                    "As a monkeypatch, we enable the `KMP_DUPLICATE_LIB_OK`."
                )
                warn(msg, stacklevel=2)
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

                if sys.modules.pop("faiss") is not self:
                    msg = "Dummy faiss module should only be defined during scio init"
                    raise RuntimeError(msg)

                self.__faiss = __faiss = importlib.import_module("faiss")
                sys.modules["faiss"] = __faiss

            return __faiss.__getattribute__(attr)

    sys.modules["faiss"] = DummyFaiss("faiss")
