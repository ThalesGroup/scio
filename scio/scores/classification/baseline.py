"""Baseline for classification."""

__all__ = ["BaselineClassif"]

from .softmax import Softmax


class BaselineClassif(Softmax):
    """Alias for :class:`~.softmax.Softmax`."""
