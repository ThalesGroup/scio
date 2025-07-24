"""Fixtures for recorder tests."""

from collections.abc import Callable
from functools import partial
from itertools import chain, combinations
from types import MappingProxyType

import pytest
import torch
from torch import Tensor, nn
from torch.nn.functional import conv2d, linear, max_pool2d, softplus

from scio.recorder import DepthIdx, Recorder
from test.scio.conftest import Net3DFactory

LAYERS_ALL = tuple((1, i) for i in range(1, 6))
LAYERS_RECORD = (
    (),
    ((1, 4),),
    ((1, 3), (1, 1)),
    ((1, 5), (1, 2), (0, 1), (1, 5)),
)
LAYERS_FORWARD = tuple(
    chain.from_iterable(combinations(LAYERS_ALL, n) for n in [0, 1, 2, 5]),
)


@pytest.fixture
def _net(dtype_device) -> Net3DFactory:
    """Create toy network for 3D samples."""
    return Net3DFactory(**dtype_device)


@pytest.fixture
def conv_weight(_net, dtype_device_gen) -> Tensor:
    """Generate controlled weigths for ``_net`` convolution."""
    return torch.randn(_net.conv_weight_shape, **dtype_device_gen)


@pytest.fixture
def conv_bias(_net, dtype_device_gen) -> Tensor:
    """Generate controlled biases for ``_net`` convolution."""
    return torch.randn(_net.conv_weight_shape[0], **dtype_device_gen)


@pytest.fixture
def fc_weight(_net, dtype_device_gen) -> Tensor:
    """Generate controlled weigths for ``_net`` fully connected."""
    return torch.randn(_net.fc_weight_shape, **dtype_device_gen)


@pytest.fixture
def fc_bias(_net, dtype_device_gen) -> Tensor:
    """Generate controlled biases for ``_net`` fully connectr."""
    return torch.randn(_net.fc_weight_shape[0], **dtype_device_gen)


@pytest.fixture
def net(_net, conv_weight, conv_bias, fc_weight, fc_bias) -> Net3DFactory:
    """:class:`Net3DFactory` instance with manually set parameters."""
    _net.load_state_dict(
        {
            "conv.weight": conv_weight,
            "conv.bias": conv_bias,
            "fc.weight": fc_weight,
            "fc.bias": fc_bias,
        },
        strict=True,
    )
    return _net


@pytest.fixture
def layer_func(
    conv_weight,
    conv_bias,
    fc_weight,
    fc_bias,
) -> Callable[[DepthIdx], Callable[[Tensor], Tensor]]:
    """Fixture to access manually created layer functions."""

    def inner(layer: DepthIdx) -> Callable[[Tensor], Tensor]:
        """Transform like ``layer`` would."""
        if layer == (1, 1):
            return partial(conv2d, weight=conv_weight, bias=conv_bias)
        if layer == (1, 2):
            return partial(softplus, beta=0.4)
        if layer == (1, 3):
            return partial(max_pool2d, kernel_size=2)
        if layer == (1, 4):
            return partial(torch.flatten, start_dim=1)
        if layer == (1, 5):
            return partial(linear, weight=fc_weight, bias=fc_bias)

        msg = f"Invalid layer: {layer}"
        raise ValueError(msg)

    return inner


N_SAMPLES = 5


@pytest.fixture
def data(net, dtype_device_gen) -> Tensor:
    """Generate random calibration data."""
    return torch.rand(N_SAMPLES, *net.sample_shape, **dtype_device_gen)


@pytest.fixture
def rnet(net, data) -> Recorder:
    """Create :class:`Recorder` net for recorder tests."""
    return Recorder(net, input_data=data)


@pytest.fixture
def layers_to_modules(_net) -> MappingProxyType[DepthIdx, nn.Module]:
    """Define expected map from layer to module."""
    return MappingProxyType({
        (0, 1): _net,
        (1, 1): _net.conv,
        (1, 2): _net.softplus,
        (1, 3): _net.pool,
        (1, 4): _net.flatten,
        (1, 5): _net.fc,
    })


@pytest.fixture
def make_recorder_forward_kwarg_tester() -> Callable[[object], Recorder]:
    """Make Recorder tester factory."""

    def inner(null: object) -> Recorder:
        """Make a :meth:`forward` kwarg tester."""

        class NetRaiseOnGivenKwargFactory(nn.Module):
            """Raises on particular :meth:`forward` kwarg."""

            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, *, obj: object = 0) -> Tensor:
                """Raise only if ``obj is null`."""
                if obj is null:
                    msg = f"'obj' should not be {null!r}"
                    raise ValueError(msg)
                return x

        return Recorder(NetRaiseOnGivenKwargFactory(), input_size=(1,))

    return inner


class NetDynamicFlowFactory(nn.Module):
    """Network with dynamic control flow."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> Tensor:
        """If statement for dynamic flow."""
        return x if x.sum() > 0 else -x


@pytest.fixture
def make_postproc_func(rnet) -> Callable[[object], Callable[[Tensor], None]]:
    """Fixture for activation postproc function maker."""

    def inner(desc: object) -> Callable[[Tensor], None]:
        """Make activation postproc function with given description."""

        def postproc_func(tensor: Tensor) -> None:
            """Print description and tensor maker(s)."""
            layers = {layer for layer, act in rnet.activations.items() if act is tensor}
            print(f"Executed postproc func {desc!r} on tensor output by {layers}")  # noqa: T201 (print statement)

        return postproc_func

    return inner


@pytest.fixture
def postproc_act_remover(rnet) -> Callable[[Tensor], None]:
    """Sneaky deleter fixture."""

    def inner(_tensor: Tensor) -> None:
        """Remove activations."""
        rnet._activations = {}  # noqa: SLF001 (private attribute)

    return inner
