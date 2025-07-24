"""Defining Recorder class."""

__all__ = ["DepthIdx", "Recorder"]

from collections import defaultdict
from collections.abc import Callable
from types import MappingProxyType
from weakref import ReferenceType, WeakMethod

import torch
import torchinfo
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torchinfo.layer_info import LayerInfo

type DepthIdx = tuple[int, int]
type Postprocessor = Callable[[Tensor], Tensor | None]


class Recorder(nn.Module):
    """Wrapper class to operate inside a torch neural network.

    A "recorder net" (referred to as ``rnet``) is a wrapped instance of
    :class:`torch.nn.Module`, augmented with inspection utilities.

    Arguments
    ---------
    net: ``nn.Module``
        Net to be recorded.
    force_static_flow: ``bool``
        If ``True``, the control flow is supposed static, meaning that
        it should be the same for any processed input. Two counter
        examples would be:

        - if the network operates a data-shape-specific resizing flows;
        - if the network uses ``if`` statements or loops based on the
          layer output values, leading to specific handling.

        More information on `dynamic control flow
        <https://pytorch.org/docs/stable/fx.html#dynamic-control-flow>`_.
        Defaults to ``True``.
    **summary_kwargs:
        Passed to :func:`torchinfo.summary`. Requires ``input_data`` or
        ``input_size``. Keyword ``force_static_flow`` is restricted by
        design.

    Example
    -------
    Consider the following toy ``rnet``, built with :class:`Recorder`
    and layers from :class:`torch.nn.Module`::

        toynet = Sequential(Linear(5, 4), ReLU(), Linear(4, 3), ReLU())
        rnet = Recorder(toynet, input_size=(1, 5))

    First of all, we easily observe the architecture of the network in a
    human-redeable form with labelled layers::

        >>> rnet
        Recorder instance for the following network
        ============================================================================================================================================
        Layer (type (var_name):depth-idx)        Input Shape               Output Shape              Param #                   Param %
        ============================================================================================================================================
        Sequential (Sequential)                  [1, 5]                    [1, 3]                    --                             --
        ├─Linear (0): 1-1                        [1, 5]                    [1, 4]                    24                         61.54%
        ├─ReLU (1): 1-2                          [1, 4]                    [1, 4]                    --                             --
        ├─Linear (2): 1-3                        [1, 4]                    [1, 3]                    15                         38.46%
        ├─ReLU (3): 1-4                          [1, 3]                    [1, 3]                    --                             --
        ============================================================================================================================================
        Total params: 39
        Trainable params: 39
        Non-trainable params: 0
        Total mult-adds (Units.MEGABYTES): 0.00
        ============================================================================================================================================
        Input size (MB): 0.00
        Forward/backward pass size (MB): 0.00
        Params size (MB): 0.00
        Estimated Total Size (MB): 0.00
        ============================================================================================================================================
        Currently recording: None
        ============================================================================================================================================

    Note how layers are labelled (*e.g.* ``1-3``). The (hidden) label
    ``0-1`` refers to the entire network. In the above example, the last
    line reports that no layer is being recorded. One can change that
    with :meth:`record`::

        >>> rnet.record((0, 1), (1, 3))
        >>> rnet
        [...]
        Currently recording: 0-1, 1-3
        [...]

    Upon computing a forward pass, the associated activations (*i.e.*
    the output of the targeted modules) are accessible through the
    :attr:`activations` attribute::

        >>> rnet(torch.rand((1, 5)))  # Forward pass for 1 random sample
        tensor([[0.0000, 0.1592, 0.0471]], grad_fn=<ReluBackward0>)
        >>> rnet.activations
        mappingproxy({(1, 3): tensor([[-0.1398,  0.7325, -0.2175]], grad_fn=<AddmmBackward0>), (0, 1): tensor([[0.0000, 0.7325, 0.0000]], grad_fn=<ReluBackward0>)})

    Other utilities are documented below.

    Warning
    -------
    In case of `dynamic control flow
    <https://pytorch.org/docs/stable/fx.html#dynamic-control-flow>`_,
    the result is **not guaranteed**! It might work, fail, or **fail
    silently** (*i.e.* the recorded activations could in fact correspond
    to the wrong layers).

    """  # noqa: E501 (line too long)

    def __init__(
        self,
        net: nn.Module,
        /,
        *,
        force_static_flow: bool = True,
        **summary_kwargs: object,
    ) -> None:
        """Construct ``Recorder`` instance for ``net``."""
        super().__init__()
        self._net = net
        self._activations: dict[DepthIdx, Tensor] = {}
        if force_static_flow:
            self._check_static_flow()

        default_summary_kwargs = {
            "col_names": ("input_size", "output_size", "num_params", "params_percent"),
            "depth": torch.inf,
            "row_settings": ("var_names", "depth"),
            "verbose": torchinfo.Verbosity.QUIET,
        }

        final_summary_kwargs = default_summary_kwargs | summary_kwargs

        self._compute_summary(**final_summary_kwargs)
        self.record()
        self._apply_hooks()

    def _check_static_flow(self) -> None:
        """Check whether the control flow of ``self.net`` is static."""
        try:
            torch.fx.symbolic_trace(self.net)
        except torch.fx.proxy.TraceError as e:
            msg = (
                "The control flow of the input network is not static (see traceback). "
                "Use `Recorder` at your own risk with `force_static_flow=False`. See "
                "`help(Recorder)` for a detailed warning"
            )
            raise RuntimeError(msg) from e

    def _compute_summary(self, **summary_kwargs: object) -> None:
        """Run ``torchinfo.summary`` on ``self.net``."""
        if not {"input_size", "input_data"} & summary_kwargs.keys():
            msg = "You must provide `input_data` or `input_size`"
            raise ValueError(msg)

        self.summary = torchinfo.summary(self.net, **summary_kwargs)  # type: ignore[arg-type]

        # https://github.com/TylerYep/torchinfo/issues/369
        # Now we clear ``tochinfo`` cached forward pass since it stores
        # useless reference to ``net``. It can also ignore changes in
        # the network flow when dynamic!
        torchinfo.torchinfo.clear_cached_forward_pass()

    def _recording_and_postproc_hook(
        self,
        module: nn.Module,
        _input: tuple,  # Here for signature compliance
        output: Tensor,
    ) -> Tensor | None:
        """Store the activations and apply postprocessing if any."""
        # A module's call doesn't necessarily originate from this
        # instance's forward call. This is checked by the presence or
        # absence of self._record_seqs.
        if not hasattr(self, "_record_seqs"):
            return None

        # Record
        depth_idx, should_record = self._record_seqs[module].pop(0)
        if not should_record:
            return None
        if not self._dont_record:
            self._activations[depth_idx] = output

        # Output postproc if any
        if (funcs := self._activation_postproc) is None:
            return None
        if funcs == []:
            msg = (
                "There were not enough postprocessing functions for every recorded "
                f"layers ({len(self.recording)}). If you wish to use the same function"
                " for every recorded layer, use the signature `rnet(*args,"
                " activation_postproc: Postprocessor, **kwargs). See "
                "`help(rnet.forward)` for more information"
            )
            raise ValueError(msg)

        func = funcs.pop(0) if isinstance(funcs, list) else funcs
        return func(output)

    def _apply_hooks(self) -> None:
        """Apply hook to every module.

        Uses :meth:`weakref.WeakMethod` so ``del rnet`` collects
        ``rnet`` and cleans the wrapped :attr:`net._forward_hooks`.
        """
        hook_ref = WeakMethod(self._recording_and_postproc_hook)
        hook_weak = lazy_func(hook_ref)

        self._handles = [
            module.register_forward_hook(hook_weak) for module in self.record_seqs
        ]

    def forward(
        self,
        *args: object,
        activation_postproc: list[Postprocessor] | Postprocessor | None = None,
        dont_record: bool = False,
        **kwargs: object,
    ) -> object:
        """Forward pass with optional recording & on-the-fly processing.

        During instanciation, the structure of the network is computed
        with ``torchinfo``. By using :meth:`record`, the user defines
        layers of interest. During a forward pass, their activations are
        recorded and later accessible through the :attr:`activations`
        attribute. If required, these are also postprocessed in-place
        before they are propagated further in the network (see
        ``activation_postproc``).

        Arguments
        ---------
        *args:
            Passed to :attr:`self.net <net>`.
        activation_postproc: ``list[Postprocessor] | Postprocessor``, optional
            If provided, the postprocessing to apply to activations
            before propagating them further in the network. We define::

                type Postprocessor = Callable[[Tensor], Tensor | None]

            where the signature can be interpreted as ``output ->
            modified output or None``. Postprocessings are applied
            **after** recording and **not** on a copy. As such, any
            in-place operation will impact records. If a ``list``,
            elements correspond to per-layer postprocessing, in the same
            order recorded layers are called. The list must be long
            enough for every recorded layer to get a postprocessor, and
            remaining elements are ignored. If a ``Postprocessor``, it
            will be applied to all recorded layers.
        dont_record: ``bool``
            If ``True``, activations are not recorded. Postprocessing,
            if any, will still occur. Defaults to ``False``.
        **kwargs:
            Passed to :attr:`self.net <net>`. The keywords
            ``activation_postproc`` and ``dont_record`` are restricted
            by design.

        Note
        ----
        As with any :class:`torch.nn.Module` instance, one should
        directly use call statements instead of using this method.

        Raises
        ------
        :exc:`RuntimeError`
            If not all the requested activations were found during the
            forward pass.

        """
        self._dont_record = dont_record
        self._record_seqs = {
            module: records.copy() for module, records in self.record_seqs.items()
        }
        self._activation_postproc = (
            activation_postproc.copy()
            if isinstance(activation_postproc, list)
            else activation_postproc
        )

        self._activations = {}
        try:
            out = self.net(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            self._activations = {}
            self.__clean()
            raise

        self.__clean()
        if not dont_record and set(self.recording) != set(self._activations):
            self._activations = {}
            diff = sorted(set(self.recording) - set(self._activations))
            msg = (
                f"Missing {diff} activations. This may be due to dynamic control flow,"
                " for which functionality is NOT GUARANTEED!"
            )
            raise RuntimeError(msg)

        return out

    def record(self, *depth_idxs: DepthIdx) -> None:
        """Set the recorded layers.

        Arguments
        ---------
        *depth_idxs: ``DepthIdx``
            The 2-tuples ``(depth, idx)`` of layers to record.
            Duplicates and order are ignored.

        Example
        -------
        ::

            rnet.record((0, 1), (1, 3))

        Raises
        ------
        :exc:`ValueError`
            If the requested layers were not found.

        """
        recording = []
        # Modules may be called several time for a single forward pass.
        # As such, we store which calls to record in a dict of boolean
        # lists, one for every module.
        record_seqs = defaultdict(list)
        for layer_info in self.summary.summary_list:
            depth_idx = get_layer_info_depth_idx(layer_info)
            is_requested = depth_idx in depth_idxs

            module = layer_info.module
            record_seqs[module].append((depth_idx, is_requested))

            if is_requested:
                recording.append(depth_idx)

        # Error if not all requested layers were found
        if diff := (set(depth_idxs) - set(recording)):
            # Add hint in case of ``rnet.record(1, 2)`` mistake
            hint = ""
            if len(depth_idxs) == 2 and all(isinstance(elt, int) for elt in depth_idxs):  # noqa: PLR2004 (magic value)
                hint = f"Did you mean `rnet.record({depth_idxs})`? "

            msg = f"Following layers not found: {diff}. {hint}Operation cancelled"
            raise ValueError(msg)

        self.record_seqs: dict[nn.Module, list[tuple[DepthIdx, bool]]] = dict(
            record_seqs,
        )
        self._recording = sorted(recording)

    @property
    def net(self) -> nn.Module:
        """The wrapped :class:`torch.nn.Module` instance."""
        return self._net

    # https://github.com/sphinx-doc/sphinx/issues/13488
    @property
    def activations(self) -> MappingProxyType[DepthIdx, Tensor]:
        """Recorded activations from the latest forward pass.

        Example
        -------
        ::

            >>> rnet.activations
            mappingproxy({(1, 3): tensor([[0.0729, 0.0405, 0.2566]], grad_fn=<AddmmBackward0>), (0, 1): tensor([[0.0729, 0.0405, 0.2566]], grad_fn=<ReluBackward0>)})
            >>> rnet.activations[(1, 3)]
            tensor([[0.0729, 0.0405, 0.2566]], grad_fn=<AddmmBackward0>)

        """  # noqa: E501 (line too long)
        return MappingProxyType(self._activations)

    @property
    def recording(self) -> tuple[DepthIdx, ...]:
        """Tuple of layers which are being recorded.

        Example
        -------
        ::

            >>> rnet.recording
            ((0, 1), (1, 3))

        """
        return tuple(self._recording)

    @property
    def recorded_modules(self) -> dict[DepthIdx, nn.Module]:
        """Map to recorded modules.

        Example
        -------
        ::

            >>> rnet.recorded_modules
            {(0, 1): Sequential(
              (0): Linear(in_features=5, out_features=4, bias=True)
              (1): ReLU()
              (2): Linear(in_features=4, out_features=3, bias=True)
              (3): ReLU()
            ), (1, 3): Linear(in_features=4, out_features=3, bias=True)}

        """
        map_ = {}
        for layer_info in self.summary.summary_list:
            depth_idx = get_layer_info_depth_idx(layer_info)
            if depth_idx in self.recording:
                map_[depth_idx] = layer_info.module

        return map_

    @property
    def recorded_params(self) -> dict[str, Parameter]:
        """Map to named parameters from recorded modules only.

        Example
        -------
        ::

            >>> rnet.recorded_params
            {'_net.0.weight': Parameter containing:
            tensor([[-0.2762, -0.1564,  0.1478, -0.1285, -0.1707],
                    [ 0.1708, -0.3490, -0.3634,  0.4290, -0.0088],
                    [ 0.2128,  0.2551,  0.4042,  0.4373,  0.2639],
                    [ 0.1333,  0.0394,  0.2923,  0.2545, -0.4048]], requires_grad=True), '_net.0.bias': Parameter containing:
            tensor([ 0.2976,  0.3207, -0.0690, -0.1086], requires_grad=True), '_net.2.weight': Parameter containing:
            tensor([[ 0.3134,  0.0609, -0.2515, -0.1318],
                    [ 0.0082,  0.3514, -0.3874,  0.0739],
                    [ 0.4850, -0.3249,  0.4155, -0.1247]], requires_grad=True), '_net.2.bias': Parameter containing:
            tensor([-0.0571, -0.2613,  0.0395], requires_grad=True)}

        """  # noqa: E501 (line too long)
        params_set = {
            param
            for module in self.recorded_modules.values()
            for param in module.parameters()
        }

        return {k: v for k, v in self.named_parameters() if v in params_set}

    def __repr__(self) -> str:
        """Add recorded layers info to torchinfo repr."""
        recording_list_str = ", ".join([
            "-".join(map(str, depth_idx)) for depth_idx in self.recording
        ])
        summ_str = repr(self.summary)
        return (
            "Recorder instance for the following network\n"
            + summ_str
            + f"\nCurrently recording: {recording_list_str or None}\n"
            + summ_str[0] * summ_str.find("\n")
        )

    def __clean(self) -> None:
        """Clean after or during forward call."""
        to_del = [
            "_dont_record",
            "_record_seqs",
            "_activation_postproc",
        ]

        for k in to_del:
            if hasattr(self, k):
                delattr(self, k)

    def __del__(self) -> None:
        """Remove recording hooks."""
        for handle in getattr(self, "_handles", ()):
            handle.remove()


# github.com/TylerYep/torchinfo/issues/347
def get_layer_info_depth_idx(layer_info: LayerInfo) -> tuple[int, int]:
    """Check that the index is not ``None``."""
    depth, index = layer_info.depth, layer_info.depth_index

    if index is None:
        msg = f"Unexpected `None` depth_index for {layer_info}"
        raise ValueError(msg)

    return depth, index


def lazy_func[**P, T](
    func_weak: ReferenceType[Callable[P, T]],
) -> Callable[P, T | None]:
    """Make lazy call from weak ref."""

    def inner(*args: P.args, **kwargs: P.kwargs) -> T | None:
        func = func_weak()
        if func is None:
            return None

        return func(*args, **kwargs)

    return inner
