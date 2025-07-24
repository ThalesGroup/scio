"""RelativeMahalanobis for classification."""

__all__ = ["RelativeMahalanobis"]

from paramclasses import protected

from scio.utils import check

from .deepmahalanobis import DeepMahalanobis


class RelativeMahalanobis(DeepMahalanobis):
    """RelativeMahalanobis for classification.

    Functionality is implemented in
    :class:`~.deepmahalanobis.DeepMahalanobis`.

    References
    ----------
    .. bibliography::
       :filter: false

       RelativeMahalanobis

    """

    _relative = protected(val=True)  # type: ignore[assignment]  # github.com/eliegoudout/paramclasses/issues/34

    def _check_relative(self) -> None:
        check(self._relative)
