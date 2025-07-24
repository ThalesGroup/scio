"""scio fixtures.

Highlights
----------
dtype, dtype_device, device_gen, dtype_device_gen
atol_rtol
match_outerr_torch, match_plots_torch
monkeypatch_enum_new_member
Net3DFactory

"""

from collections.abc import Callable
from enum import Enum, EnumType
from operator import itemgetter
from types import MappingProxyType
from typing import TypedDict

import pytest
import torch
from torch import Tensor, nn


@pytest.fixture(scope="session", params=["half", "float", "double"])
def dtype(request) -> torch.dtype:
    """Define the dtype for current test."""
    return getattr(torch, request.param)


class DtypeDevice(TypedDict):
    """Options for ``dtype`` and ``device``."""

    dtype: torch.dtype
    device: torch.device


@pytest.fixture(scope="session")
def dtype_device(dtype, device) -> DtypeDevice:
    """Unpack to set ``dtype`` and ``device``."""
    return MappingProxyType({  # type: ignore[return-value]  # github.com/python/typing/issues/1952
        "dtype": dtype,
        "device": device,
    })


class DeviceGen(TypedDict):
    """Options for ``device`` and ``generator``."""

    device: torch.device
    generator: torch.Generator


@pytest.fixture
def device_gen(device, generator) -> DeviceGen:
    """Unpack to set ``device`` and ``generator``."""
    return MappingProxyType({  # type: ignore[return-value]  # github.com/python/typing/issues/1952
        "device": device,
        "generator": generator,
    })


class DtypeDeviceGen(DtypeDevice):
    """Options for ``dtype``, ``device`` and ``generator``."""

    generator: torch.Generator


@pytest.fixture
def dtype_device_gen(dtype_device, generator) -> DtypeDeviceGen:
    """Unpack to set ``dtype``, ``device`` and ``generator``."""
    return MappingProxyType({"generator": generator} | dtype_device)  # type: ignore[return-value]  # github.com/python/typing/issues/1952


class ATolRTol(TypedDict):
    """Tolerances for testing functions."""

    atol: float | None
    rtol: float | None


@pytest.fixture(scope="session")
def atol_rtol(dtype) -> ATolRTol:
    """Unpack to set ``atol`` and ``rtol``. Adjusts ``torch.half``."""
    return MappingProxyType({  # type: ignore[return-value]  # github.com/python/typing/issues/1952
        "atol": 1e-3 if dtype is torch.half else None,
        "rtol": 3e-3 if dtype is torch.half else None,
    })


@pytest.fixture
def match_outerr_torch(dtype, device, match_outerr) -> None:
    """Disable ``match_outerr`` unless float on cpu."""
    if dtype != torch.float or device != torch.device("cpu"):
        match_outerr.ignore()


@pytest.fixture
def match_plots_torch(dtype, device, match_plots) -> None:
    """Disable ``match_plots`` unless float on cpu."""
    if dtype != torch.float or device != torch.device("cpu"):
        match_plots.ignore()


@pytest.fixture
def monkeypatch_enum_new_member(monkeypatch) -> Callable[[EnumType], Enum]:
    """Add target Enum a :attr:`NEW` member for tests."""

    def inner(enum: EnumType) -> Enum:
        """Add new member :attr:`NEW` with ``unexpected`` value.

        Only adds the :meth:`__call__` interface, not attribute access.
        """
        name, value = "NEW", "unexpected"
        new_member = enum._new_member_(enum)  # type: ignore[attr-defined]
        new_member._name_ = name
        new_member._value_ = value
        assert value not in enum._value2member_map_
        monkeypatch.setitem(enum._value2member_map_, value, new_member)

        # Quick check of minimalist monkeypatch
        assert enum(value) is enum(new_member) is new_member

        return new_member

    return inner


class Net3DFactory(nn.Module):
    """Test classif NN for 3D samples CHW.

    ::

        Net3DFactory (Net3DFactory)
        ├─Conv2d (conv) 1-1
        ├─GELU (gelu) 1-2
        ├─MaxPool2d (pool) 1-3
        ├─Flatten (flatten) 1-4
        └─Linear (fc) 1-5

    """

    __test__ = False

    def __init__(  # noqa: PLR0913 (too many arguments)
        self,
        *,
        n_channels: int = 1,
        height: int = 4,
        width: int = 4,
        n_classes: int = 2,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Construct a Net3DFactory.

        Arguments
        ---------
        n_channels: ``int``
            Defaults to ``1`` (minimal).
        height: ``int``
            Defaults to ``4`` (minimal).
        width: ``int``
            Defaults to ``4`` (minimal).
        n_classes: ``int``
            Defaults to ``2`` (minimal).
        dtype: ``torch.dtype``, optional
            If provided, the common dtype to use for every layer.
        device: ``torch.device``, optional
            If provided, on which device to transfer the model it.
        generator: ``torch.Generator``, optional
            Used to initialize the weights of the network.

        """
        # Minimal values
        assert n_channels >= 1
        assert height >= 4
        assert width >= 4
        assert n_classes >= 2
        # Utils
        self.sample_shape = (n_channels, height, width)
        self.n_classes = n_classes
        self.conv_weight_shape = (n_channels + 1, n_channels, 3, 3)
        self.fc_weight_shape = (
            n_classes,
            (n_channels + 1) * (height // 2 - 1) * (width // 2 - 1),
        )
        # Modules
        super().__init__()
        self.conv = nn.Conv2d(*itemgetter(1, 0, 2)(self.conv_weight_shape))
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(*self.fc_weight_shape[::-1])
        self.softplus = nn.Softplus(beta=0.4)  # Avoid saturating, zero and zero-grad

        # Handle dtype, device and randomness
        self.to(dtype=dtype, device=device)
        for parameter in self.parameters():
            nn.init.normal_(parameter, generator=generator)

    def forward(self, x) -> Tensor:
        """Define forward as conv > softplus > pool > flat > fc."""
        return self.fc(self.flatten(self.pool(self.softplus(self.conv(x)))))
