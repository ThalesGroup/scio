"""Test recorder class."""

import gc
import random
import re
from functools import partial
from weakref import ref

import pytest
import torch

from scio.recorder import Recorder, get_layer_info_depth_idx, lazy_func
from test.conftest import parametrize_bool

from .conftest import LAYERS_ALL, LAYERS_FORWARD, LAYERS_RECORD, NetDynamicFlowFactory


def test_recorder_passes_torchinfo_kwargs(net, data, match_outerr_torch):  # noqa: ARG001 (unused argument)
    """Check that kwargs are passed to :func:`torchinfo.summary`."""
    rnet = Recorder(net, input_data=data, col_names=["output_size"])
    print(f"# With col_names=['output_size']\n{rnet!r}\n#")


@pytest.mark.parametrize("layers", LAYERS_RECORD)
def test_recorder_repr(rnet, layers, match_outerr_torch):  # noqa: ARG001 (unused argument)
    """Match ``repr`` for instances."""
    rnet.record(*layers)
    print(f"# With layers={layers}\n{rnet!r}\n#")


def test_invalid_record(rnet):
    """Test invalid :meth:`record` layers."""
    msg = "Following layers not found: {(1, 0)}. Operation cancelled"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        rnet.record((1, 0))


def test_invalid_record_with_hint(rnet):
    """Test ``rnet.record(1, 2)`` misuse hint."""
    msg = (
        "Following layers not found: {1, 3}. Did you mean `rnet.record((1, 3))`? "
        "Operation cancelled"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        rnet.record(1, 3)


@pytest.mark.parametrize("layers", [*LAYERS_RECORD, *LAYERS_FORWARD])
def test_recording(rnet, layers):
    """Check :attr:`recording` value."""
    rnet.record(*layers)
    assert rnet.recording == tuple(sorted(set(layers)))


@pytest.mark.parametrize("layers", LAYERS_FORWARD)
def test_correct_activations(rnet, layers, data, layer_func, appender):
    """Test recorded activations on forward pass without option."""
    rnet.record(*layers)
    out = rnet(data)
    activations = rnet.activations

    # Check keys
    assert tuple(sorted(activations)) == rnet.recording

    # Check values
    expected = data
    for layer in LAYERS_ALL:
        expected = layer_func(layer)(expected)
        if layer in layers:
            torch.testing.assert_close(
                activations[layer],
                expected,
                msg=appender(f"Layer: {layer}"),
            )

    # Check that manual computation yields same result as network computation
    torch.testing.assert_close(out, expected)


def test_layer_01(rnet, data):
    """Check special ``(0, 1)`` layer."""
    rnet.record((0, 1))
    out = rnet(data)
    torch.testing.assert_close(rnet.activations[(0, 1)], out)


def test_dont_record(rnet, data):
    """Test ``dont_record`` option."""
    rnet.record((0, 1))
    _ = rnet(data, dont_record=True)
    assert not rnet.activations


@parametrize_bool("dont_record")
@pytest.mark.parametrize("layers", LAYERS_FORWARD)
def test_correct_activations_with_postproc(
    rnet,
    layers,
    data,
    dont_record,
    layer_func,
    appender,
):
    """Test recorded activations on forward pass with actual postproc.

    Tested with ``torch.roll`` postprocessing. Also check that when
    ``dont_record``, preprocessing still applies and indeed disables
    recording.

    """
    rnet.record(*layers)
    postproc = partial(torch.roll, shifts=1)  # Arbitrary, must impact net out
    out = rnet(data, activation_postproc=postproc, dont_record=dont_record)
    activations = rnet.activations
    if dont_record:
        assert not activations

    # Check values
    expected = data
    for layer in LAYERS_ALL:
        expected = layer_func(layer)(expected)
        if layer in layers:
            if not dont_record:
                torch.testing.assert_close(
                    activations[layer],
                    expected,
                    msg=appender(f"Layer: {layer}"),
                )
            expected = postproc(expected)

    # Check that manual computation yields same result as network computation
    torch.testing.assert_close(out, expected)

    # Postprocessing indeed modified output
    if layers:
        assert not torch.allclose(rnet(data), out)


def test_no_interference_multiple_recorder(net, data):
    """Check that two ``rnet`` on same ``net`` do not interfere."""
    layer = random.choice(LAYERS_ALL)  # noqa: S311 (not cryptographically safe)

    rnet1 = Recorder(net, input_data=data)
    rnet1.record(layer)
    rnet2 = Recorder(net, input_data=data)
    rnet2.record(layer)

    # Empty activations
    assert layer not in rnet1.activations
    assert layer not in rnet2.activations

    # After one call, only one activation recorded
    rnet1(data)
    assert layer in rnet1.activations
    assert layer not in rnet2.activations

    # After second call, both recorded the same result
    rnet2(data)
    torch.testing.assert_close(rnet1.activations[layer], rnet2.activations[layer])


@pytest.mark.parametrize("layers", LAYERS_RECORD)
def test_recorded_modules(rnet, layers, layers_to_modules):
    """Test attribute :attr:`recorded_modules`."""
    rnet.record(*layers)
    expected = {layer: layers_to_modules[layer] for layer in layers}
    assert rnet.recorded_modules == expected


@pytest.mark.parametrize("layers", LAYERS_RECORD)
def test_recorded_params(
    rnet,
    conv_weight,
    conv_bias,
    fc_weight,
    fc_bias,
    layers,
    appender,
):
    """Test attribute :attr:`_recorded_params`."""
    rnet.record(*layers)

    expected = {}
    if (0, 1) in layers or (1, 1) in layers:
        expected |= {
            "_net.conv.weight": conv_weight,
            "_net.conv.bias": conv_bias,
        }
    if (0, 1) in layers or (1, 5) in layers:
        expected |= {
            "_net.fc.weight": fc_weight,
            "_net.fc.bias": fc_bias,
        }

    observed = rnet.recorded_params
    assert observed.keys() == expected.keys()
    for name in observed:
        torch.testing.assert_close(
            observed[name],
            expected[name],
            msg=appender(f"Name: {name}"),
        )


def test_respects_grad(data, rnet, layer_func):
    """Test that gradients flow through ``rnet``."""
    data.requires_grad_()
    data.grad = None
    out = data
    for layer in LAYERS_ALL:
        out = layer_func(layer)(out)
    out.sum().backward()
    expected = data.grad
    assert expected.shape == data.shape

    data.grad = None
    rnet(data).sum().backward()
    observed = data.grad

    torch.testing.assert_close(observed, expected)


def test_recorder_forward_with_kwarg(make_recorder_forward_kwarg_tester, null, data):
    """Test optional kwargs can be passed to :meth:`forward`."""
    rnet = make_recorder_forward_kwarg_tester(null)

    # No kwarg: OK
    rnet(data)

    # ``obj=null``: FAILS
    msg = f"'obj' should not be {null!r}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        rnet(data, obj=null)


def test_dynamic_flow_requires_enabling(data):
    """Test dynamic control flow works when enabled."""
    # Dynamic flow fails
    msg = (
        "The control flow of the input network is not static (see traceback). Use "
        "`Recorder` at your own risk with `force_static_flow=False`. See "
        "`help(Recorder)` for a detailed warning"
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(msg)}$"):
        Recorder(NetDynamicFlowFactory(), input_data=data)

    # Works by disabling forced static flow
    Recorder(NetDynamicFlowFactory(), input_data=data, force_static_flow=False)


def test_recorder_manages_refs(net, data):
    """Check that :class:`Recorder` properly manages target's refs.

    Also check that hooks are removed on garbage collection.
    """
    gc.collect()
    n_refs_before = len(gc.get_referrers(net))
    assert len(net._forward_hooks) == 0  # noqa: SLF001 (private attribute)

    rnet = Recorder(net, input_data=data)
    gc.collect()
    n_refs_during = len(gc.get_referrers(net))
    assert len(net._forward_hooks) == 1  # noqa: SLF001 (private attribute)
    assert n_refs_during > n_refs_before

    del rnet
    gc.collect()
    n_refs_after = len(gc.get_referrers(net))
    assert len(net._forward_hooks) == 0  # noqa: SLF001 (private attribute)
    assert n_refs_after == n_refs_before


def test_recorder_requires_shape_info(net):
    """Check that missing information raises."""
    msg = "You must provide `input_data` or `input_size`"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        Recorder(net)


@pytest.mark.parametrize("layers", LAYERS_RECORD)
def test_recorder_activation_postproc_as_list(
    rnet,
    data,
    make_postproc_func,
    layers,
    match_outerr_torch,  # noqa: ARG001 (unused argument)
):
    """Test ``activation_postproc`` as list goes through the list."""
    print(f"Layers: {layers}")
    rnet.record(*layers)
    n_funcs = max(map(len, LAYERS_RECORD)) + 1  # Always at least one more
    activation_postproc = [make_postproc_func(desc=i) for i in range(n_funcs)]
    rnet(data, activation_postproc=activation_postproc)


def test_recorder_activation_postproc_missing(rnet, data):
    """Test raise on missing postproc funcion when list."""
    rnet.record(*LAYERS_ALL)

    msg = (
        "There were not enough postprocessing functions for every recorded layers "
        f"({len(LAYERS_ALL)}). If you wish to use the same function for every "
        "recorded layer, use the signature `rnet(*args, activation_postproc: "
        "Postprocessor, **kwargs). See `help(rnet.forward)` for more information"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        rnet(data, activation_postproc=[])


@pytest.mark.parametrize("layers", filter(bool, LAYERS_RECORD))
def test_recorder_missing_activation(rnet, data, layers, postproc_act_remover):
    """Test raise on missing activations."""
    rnet.record(*layers)

    msg = (
        f"Missing {list(rnet.recording)} activations. This may be due to dynamic"
        " control flow, for which functionality is NOT GUARANTEED!"
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(msg)}$"):
        rnet(data, activation_postproc=postproc_act_remover)


def test_lazy_func_works():
    """Test :func:`lazy_func` works when target is garbage collected."""

    def func() -> int:
        """Return ``0``."""
        return 0

    lazy = lazy_func(ref(func))
    assert lazy() == 0

    del func
    gc.collect()
    assert lazy() is None


def test_layer_info_util(rnet):
    """Test type sanitizating patch."""
    layer_info = rnet.summary.summary_list[0]
    assert get_layer_info_depth_idx(layer_info) == (0, 1)

    layer_info.depth_index = None
    msg = f"Unexpected `None` depth_index for {layer_info}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        get_layer_info_depth_idx(layer_info)
