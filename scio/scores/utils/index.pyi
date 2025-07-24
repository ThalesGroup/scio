from typing import Literal, overload

import faiss  # type: ignore[import-untyped]
from paramclasses import ParamClass
from torch import Tensor

from scio.utils import IndexMetricLike

def IndexType(metric_like: IndexMetricLike) -> type[faiss.Index]: ...  # noqa: N802 (uppercase IndexType)

class Index(ParamClass):
    dim: int
    metric: IndexMetricLike
    def add(self, samples: Tensor) -> None: ...
    def remove_ids(self, ids: Tensor) -> int: ...
    def search(
        self,
        samples: Tensor,
        k: int,
        *,
        self_query: bool = ...,
    ) -> tuple[Tensor, Tensor]: ...
    @property
    def ntotal(self) -> int: ...
    @property
    def D_is_similarity(self) -> bool: ...  # noqa: N802 (uppercase D)

# github.com/python/mypy/issues/18653
# ``squeeze`` is ``False``
@overload
def make_indexes(
    all_samples: tuple[Tensor, ...] | Tensor,
    all_groups: tuple[Tensor, ...] | Tensor | None = ...,
    *,
    n_groups: int | None = ...,
    metric: str,
    squeeze: Literal[False],
) -> tuple[tuple[tuple[Index, ...], ...], ...]: ...

# ``all_groups`` is ``None``
@overload
def make_indexes(
    all_samples: Tensor,
    all_groups: None = ...,
    *,
    n_groups: int | None = ...,
    metric: str,
    squeeze: Literal[True] = ...,
) -> Index: ...
@overload
def make_indexes(
    all_samples: tuple[Tensor, ...],
    all_groups: None = ...,
    *,
    n_groups: int | None = ...,
    metric: str,
    squeeze: Literal[True] = ...,
) -> tuple[Index, ...]: ...

# ``all_groups`` is ``Tensor``
@overload
def make_indexes(
    all_samples: Tensor,
    all_groups: Tensor,
    *,
    n_groups: int,
    metric: str,
    squeeze: Literal[True] = ...,
) -> tuple[Index, ...]: ...
@overload
def make_indexes(
    all_samples: tuple[Tensor, ...],
    all_groups: Tensor,
    *,
    n_groups: int,
    metric: str,
    squeeze: Literal[True] = ...,
) -> tuple[tuple[Index, ...], ...]: ...

# ``all_groups`` is ``tuple[Tensor, ...]``
@overload
def make_indexes(
    all_samples: Tensor,
    all_groups: tuple[Tensor, ...],
    *,
    n_groups: int,
    metric: str,
    squeeze: Literal[True] = ...,
) -> tuple[tuple[Index, ...], ...]: ...
@overload
def make_indexes(
    all_samples: tuple[Tensor, ...],
    all_groups: tuple[Tensor, ...],
    *,
    n_groups: int,
    metric: str,
    squeeze: Literal[True] = ...,
) -> tuple[tuple[tuple[Index, ...], ...], ...]: ...
