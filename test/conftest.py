"""Global fixtures.

Highlights
----------
match_outerr, match_plots, match_array (with CLI options update/ignore)
reproducibility (with generator and rng, matplotlib), device
parametrize_bool
null, make_weakable

"""

import functools
import re
from collections.abc import Callable, Iterator
from functools import partial
from itertools import product, zip_longest
from math import ceil, log10
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import matplotlib.pyplot as plt
import matplotlib.testing
import numpy as np
import pytest
import torch
from matplotlib.testing.compare import compare_images
from numpy.typing import NDArray
from torch import Tensor

TEST_EXPECTED_NAME = "expected"
CAPTURE = "capture"
PLOT = "plot"
ARRAYS = "array"
TEST_EXPECTED_DIR = Path(__file__).with_name(TEST_EXPECTED_NAME)
assert TEST_EXPECTED_DIR.is_dir(), f"Missing directory: {TEST_EXPECTED_DIR}"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Retrieve cmd opts through ``sys.argv``."""
    add_flag = partial(parser.addoption, action="store_true", default=False)
    add_flag("--update-outerr", help=f"Update '{TEST_EXPECTED_DIR}/' outerr files")
    add_flag("--ignore-outerr", help="Skip the outerr matching tests")
    add_flag("--update-plots", help=f"Update '{TEST_EXPECTED_DIR}/' plots")
    add_flag("--ignore-plots", help="Skip the plots matching tests")
    add_flag("--update-arrays", help=f"Update '{TEST_EXPECTED_DIR}/' arrays")
    add_flag("--ignore-arrays", help="Use to skip the arrays matching tests")


@pytest.fixture
def update_outerr(request) -> bool:
    """Retrieve command line option."""
    return request.config.getoption("--update-outerr")


@pytest.fixture
def ignore_outerr(request) -> bool:
    """Retrieve command line option."""
    return request.config.getoption("--ignore-outerr")


@pytest.fixture
def update_plots(request) -> bool:
    """Retrieve command line option."""
    return request.config.getoption("--update-plots")


@pytest.fixture
def ignore_plots(request) -> bool:
    """Retrieve command line option."""
    return request.config.getoption("--ignore-plots")


@pytest.fixture
def update_arrays(request) -> bool:
    """Retrieve command line option."""
    return request.config.getoption("--update-arrays")


@pytest.fixture
def ignore_arrays(request) -> bool:
    """Retrieve command line option."""
    return request.config.getoption("--ignore-arrays")


@pytest.fixture
def test_stem(request) -> Path:  # pragma: no cover
    """Stem path for current test expectation files.

    May be modified locally for context (capture, plot, etc...).

    Example
    -------
    ``~/workspace/scio/test/expected/test_func/float-mode=0``

    Warning
    -------
    Stem may contain a dot ``"."``! As such, **prefer using**
    ``test_stem.name`` to ``test_stem.stem``.

    """
    test_name = request.node.name
    orig_name = request.node.originalname

    regex = rf"^{orig_name}\[(.+)\]$"  # ``test_func[foo=0]`` matches ``foo=0``
    if (match := re.match(regex, test_name)) is not None:
        name = match.group(1)
    else:
        name = test_name.removeprefix("test_")

    return TEST_EXPECTED_DIR / orig_name / name


def assert_same_outerr(stem_path, captured_outerr) -> None:  # pragma: no cover
    """Check ``expected/`` files for ``stem_path`` match captures.

    Arguments
    ---------
    stem_path: ``Path``
        E.g. ``PosixPath('~/workspace/scio/test/expected/test_func/capture_mode=1')``.
    captured_outerr: ``str``
        Captured content from ``stdout`` or ``stderr``.

    Raises
    ------
    :exc:`AssertionError`
        If one of following is true for exactly one of stdout or stderr.

        - There was nothing to capture but an associated expectation
          file exists.
        - There was captured content but the associated excpectation
          file does not exist.
        - Captured content and expectation file content do not match.

    :exc:`ExceptionGroup`
        Two :exc:`AssertionError` as above when both captured stdout and
        stderr mismatch with expectations.

    """
    excs = {}
    for captured, ext in zip(captured_outerr, ["out", "err"], strict=True):
        path = stem_path.with_name(f"{stem_path.name}.{ext}")
        if captured:
            if not path.exists():
                hash_ = hash(captured)
                excs[ext] = AssertionError(
                    f"Missing file '{path}' for the following std{ext} capture (hash="
                    f"{hash_}):\n{captured}\n<end of {hash_}>",
                )
                continue
            try:
                assert captured == path.read_text()
            except AssertionError as e:
                excs[ext] = e
        elif path.exists():
            excs[ext] = AssertionError(
                f"Unexpected file '{path}' for empty std{ext} capture",
            )

    if len(excs) == 1:
        raise excs.popitem()[1]
    if len(excs) == 2:
        msg = "Unexpected capture for both stdout and stderr"
        raise ExceptionGroup(msg, tuple(excs.values()))


class MatchIgnore:
    """Activatable ignore flag."""

    __ignore = False

    def ignore(self) -> None:
        """Activate ignore mode."""
        self.__ignore = True

    def __bool__(self) -> bool:
        """Flag value."""
        return self.__ignore


@pytest.fixture
def match_outerr(
    ignore_outerr,
    capsys,
    test_stem,
    update_outerr,
) -> Iterator[MatchIgnore]:  # pragma: no cover
    """Check captured stdout and stderr, update if required.

    In CLI, use ``--update-outerr`` to update files when not matching.
    In CLI, use ``--ignore-outerr`` to always disable.
    Use :meth:`match_outerr.ignore()` to dynamically disable.

    """
    ignore = MatchIgnore()
    yield ignore
    if ignore:
        return
    elif ignore_outerr:
        pytest.skip("Skip matching outerr as requested by '--ignore-outerr'")

    captured_outerr = capsys.readouterr()
    capture_stem = test_stem.with_name(f"{CAPTURE}_{test_stem.name}")
    try:
        assert_same_outerr(capture_stem, captured_outerr)
    except (AssertionError, ExceptionGroup):
        if not update_outerr:
            raise
    else:
        return

    # On mismatch, update expectation files if ``update_outerr``
    capture_stem.parent.mkdir(parents=True, exist_ok=True)
    for captured, ext in zip(captured_outerr, ["out", "err"], strict=True):
        if not captured:
            continue
        path = capture_stem.with_name(f"{capture_stem.name}.{ext}")
        path.write_text(captured)

    pytest.skip(reason=f"Updated related capture files in {capture_stem.parent}/")


def assert_same_plots(paths) -> None:  # pragma: no cover
    """Check ``expected/`` plots from ``paths`` match.

    Arguments
    ---------
    paths: ``dict[Path, Path | None]``
        Pairs ``expected_path: observed_path`` of paths of images that
        should match. ``observed_path`` may be ``None`` if there
        are extra expected images.

    Raises
    ------
    Every plot mismatch generates an error. When there are more than
    one, they are raised as an :exc:`ExceptionGroup`. Heavily relies on
    :func:`matplotlib.testing.compare.compare_images`.

    """
    excs: list[Exception] = []
    for expected_path, observed_path in paths.items():
        if observed_path is None:
            excs.append(AssertionError(f"Found extra plot at {expected_path}"))
            continue

        try:
            diff = compare_images(expected_path, observed_path, tol=0.01)
            if diff is not None:
                excs.append(AssertionError(diff))
        except Exception as e:  # noqa: BLE001 (raw Exception from mpl)
            excs.append(e)

    if len(excs) == 1:
        raise excs[0]
    if len(excs) >= 1:
        msg = "Multiple plots mismatch"
        raise ExceptionGroup(msg, excs)


@pytest.fixture
def match_plots(  # noqa: C901 (too complex)
    ignore_plots,
    test_stem,
    update_plots,
    monkeypatch,
) -> Iterator[MatchIgnore]:  # pragma: no cover
    """Check plots from :meth:`plt.show`, update if required.

    In CLI, use ``--update-plots`` to update files when not matching.
    In CLI, use ``--ignore-plots`` to always disable.
    Use :meth:`match_plots.ignore()` to dynamically disable.

    Note
    ----
    Does not check figures saved directly with :meth:`plt.savefig`.

    """
    ignore = MatchIgnore()

    plot_stem = test_stem.with_name(f"{PLOT}_{test_stem.name}")
    ext = "png"
    fig = "fig"

    max_plots_per_test = 10
    N = ceil(log10(max_plots_per_test - 1))
    n = 0
    pattern = re.compile(rf"^{re.escape(plot_stem.name)}\.{fig}\d+(\..*)?\.{ext}$")
    paths: dict[Path, Path | None] = dict.fromkeys(  # Find expectations paths
        path
        for path in plot_stem.parent.glob(f"{plot_stem.name}.{fig}*.{ext}")
        if pattern.match(path.name)
    )

    plt_show_orig = plt.show

    @functools.wraps(plt_show_orig)
    def plt_show(*, block: bool | None = None) -> None:  # noqa: ARG001 (unused argument)
        """Mock :meth:`plt.show`, call :meth:`plt.savefig` instead."""
        if ignore:
            return

        nonlocal n
        assert n < max_plots_per_test, f"Too many plots (max: {max_plots_per_test})"

        # Save figure
        expected_path = plot_stem.with_name(f"{plot_stem.name}.{fig}{n:0{N}d}.{ext}")
        observed_path = expected_path.with_stem(f"{expected_path.stem}.observed")
        plot_stem.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(observed_path, dpi=100, format=ext)
        plt.close("all")

        # Update collected paths
        paths[expected_path] = observed_path
        paths.pop(observed_path, None)
        n += 1

    monkeypatch.setattr(plt, "show", plt_show)
    plt.close("all")

    yield ignore
    if ignore:
        return
    elif ignore_plots:
        pytest.skip("Skip matching plots as requested by '--ignore-plots'")

    try:
        assert_same_plots(paths)
    except Exception:
        if not update_plots:
            raise
    else:
        for observed_path in paths.values():
            observed_path.unlink()  # type: ignore[union-attr]  # Should not be ``None`` when :func:`assert_same_plots` does not raise
        return

    # On mismatch, update expectation files if ``update_plots``
    for expected_path, observed_path in paths.items():
        if observed_path is None:
            expected_path.unlink()
        else:
            observed_path.replace(expected_path)

    pytest.skip(reason=f"Updated related plots in {plot_stem.parent}/")


def save_numpy(ndarray: NDArray, path: Path) -> None:  # pragma: no cover
    """Save numpy array."""
    assert type(ndarray) is np.ndarray
    np.save(path, ndarray, allow_pickle=False)


def load_numpy(path: Path) -> NDArray:  # pragma: no cover
    """Load array from ``path`` with numpy."""
    try:
        array = np.load(path, allow_pickle=False)
        assert type(array) is np.ndarray
    except Exception as e:
        msg = f"Failed to load following file with numpy binding: '{path}'"
        raise RuntimeError(msg) from e
    return array


def match_numpy(observed: NDArray, expected: NDArray, **match_opts: Any) -> None:  # noqa: ANN401 (Any)  # pragma: no cover
    """Match ``observed`` and ``expected`` for numpy arrays."""
    if {type(observed), type(expected)} != {np.ndarray}:
        msg = f"Invalid 'observed, expected' types: {type(observed)}, {type(expected)}"
        raise TypeError(msg)

    kwargs = {"equal_nan": False, "strict": True} | match_opts
    np.testing.assert_allclose(observed, expected, **kwargs)


def save_torch(tensor: Tensor, path: Path) -> None:  # pragma: no cover
    """Save torch tensor."""
    assert type(tensor) is torch.Tensor
    torch.save(tensor.cpu(), path)


def load_torch(path: Path) -> Tensor:  # pragma: no cover
    """Load tensor from ``path`` with torch."""
    try:
        array = torch.load(path, weights_only=True)
        assert type(array) is torch.Tensor
    except Exception as e:
        msg = f"Failed to load following file with torch binding: '{path}'"
        raise RuntimeError(msg) from e
    return array


def match_torch(
    observed: Tensor,
    expected: Tensor,
    **match_opts: Any,  # noqa: ANN401 (Any)
) -> None:  # pragma: no cover
    """Match ``observed`` and ``expected`` for torch tensors."""
    if {type(observed), type(expected)} != {torch.Tensor}:
        msg = f"Invalid 'observed, expected' types: {type(observed)}, {type(expected)}"
        raise TypeError(msg)

    kwargs = {"check_device": False} | match_opts
    torch.testing.assert_close(observed, expected, **kwargs)


ARRAY_TYPES_BINDINGS = MappingProxyType({
    np.ndarray: SimpleNamespace(
        save=save_numpy,
        load=load_numpy,
        assert_close=match_numpy,
        ext="npy",
    ),
    torch.Tensor: SimpleNamespace(
        save=save_torch,
        load=load_torch,
        assert_close=match_torch,
        ext="pt",
    ),
})
EXT_TO_TYPE = MappingProxyType({"npy": np.ndarray, "pt": torch.Tensor})


def assert_same_arrays(  # pragma: no cover
    paths: list[Path],
    arrays_opts: list[tuple[NDArray | Tensor, dict[str, object]]],
) -> None:
    """Check ``expected/`` arrays from ``paths`` match.

    Arguments
    ---------
    paths: ``list[Path]``
        Sorted list of expectation paths to arrays that should match
        their corresponding element from ``arrays_opts``.
    arrays_opts: ``list[tuple[NDArray | Tensor, dict[str, object]]]``
        List of ``(array, match_opts)``.

    Raises
    ------
    Every array mismatch generates an error. When there are more than
    one, they are raised as an :exc:`ExceptionGroup`.

    """
    excs: list[Exception] = []
    null = object()
    for path, array_opts in zip_longest(paths, arrays_opts, fillvalue=null):
        if path is null:
            array = cast("Any", array_opts)[0]
            excs.append(AssertionError(f"Missing file for: {array}"))
            continue

        if array_opts is null:
            excs.append(AssertionError(f"Found extra expectation ar {path}"))
            continue

        ext = cast("Path", path).suffix[1:]
        array_binds = ARRAY_TYPES_BINDINGS[EXT_TO_TYPE[ext]]

        try:
            expected = array_binds.load(path)
            array, match_opts = cast("Any", array_opts)
            array_binds.assert_close(array, expected, **match_opts)
        except Exception as e:  # noqa: BLE001 (too broad Exception)
            excs.append(e)

    if len(excs) == 1:
        raise excs[0]
    if len(excs) >= 1:
        msg = "Multiple arrays mismatch"
        raise ExceptionGroup(msg, excs)


class ArraysToMatch(MatchIgnore, list):  # pragma: no cover
    """List that extends through call, for cleaner fixture."""

    def __call__(self, array: NDArray | Tensor, **match_opts: object) -> None:
        """Extend the list with ``(array, match_opts)``.

        Arguments
        ---------
        array: ````
            Array from the test, to match.
        **match_opts: object
            Options passed to the appropriate "assert close" function.

        Raises
        ------
        :exc:`TypeError`
            If ``array`` is of unsupported type.

        """
        msg = f"Unsupported array type: {type(array)}"
        if type(array) not in {np.ndarray, torch.Tensor}:
            raise TypeError(msg)
        self.append((array, match_opts))


@pytest.fixture
def match_array(  # noqa: C901 (too complex)
    ignore_arrays,
    test_stem,
    update_arrays,
    null,
) -> Iterator[ArraysToMatch]:  # pragma: no cover
    """Check arrays passed to it during tests and update if required.

    In CLI, use ``--update-arrays`` to update files when not matching.
    In CLI, use ``--ignore-arrays`` to always disable.
    Use :meth:`match_arrays.ignore()` to dynamically disable.

    Example
    -------
    ::

        match_array(observed, equal_nan=True)  # Will be matched with expectation

    Note
    ----
    Only supports :class:`numpy.ndarray` and :class:`torch.Tensor`.

    """
    arrays_stem = test_stem.with_name(f"{ARRAYS}_{test_stem.name}")
    exts = {array_type.ext for array_type in ARRAY_TYPES_BINDINGS.values()}
    arr = "arr"

    # Find expectations paths
    pattern = re.compile(
        rf"^{re.escape(arrays_stem.name)}\.{arr}(\d+)\.({'|'.join(exts)})$",
    )
    paths: list[Path] = []
    if arrays_stem.parent.is_dir():
        paths = sorted(  # ``sorted`` works since padded array number below
            path for path in arrays_stem.parent.iterdir() if pattern.match(path.name)
        )

    ignore = fixture = ArraysToMatch()
    yield ignore
    if ignore:
        return
    elif ignore_arrays:
        pytest.skip("Skip matching arrays as requested by '--ignore-arrays'")

    if len(fixture) == 0:
        msg = "'match_array' fixture unused. Consider removing or disabling it"
        raise RuntimeError(msg)

    try:
        assert_same_arrays(paths, fixture)
    except Exception:
        if not update_arrays:
            raise
    else:
        return

    # On mismatch, update expectation files if ``update_arrays``
    max_arrays_per_test = 100
    N = ceil(log10(max_arrays_per_test - 1))
    for n, (path, array_opts) in enumerate(zip_longest(paths, fixture, fillvalue=null)):
        assert n < max_arrays_per_test, f"Too many arrays (max: {max_arrays_per_test})"
        if path is not null:
            path.unlink()

        if array_opts is null:
            continue

        # Save new expectation
        array = array_opts[0]
        array_binds = ARRAY_TYPES_BINDINGS[type(array)]
        ext = array_binds.ext
        path_new = arrays_stem.with_name(f"{arrays_stem.name}.{arr}{n:0{N}d}.{ext}")
        path_new.parent.mkdir(parents=True, exist_ok=True)
        array_binds.save(array, path_new)

    pytest.skip(reason=f"Updated related arrays in {arrays_stem.parent}/")


# Adapted from github.com/eliegoudout/paramclasses
def parametrize_bool(argnames: str, *, lite: int = -1) -> pytest.MarkDecorator:
    """Parametrize each argument to be ``True`` or ``False``.

    Arguments
    ---------
    argnames: ``str``
        Coma-separated list of argument names.
    lite: ``int``
        If nonnegative, filter to only argvalues with sum at most
        ``lite``, and all true. Negative values are treated as infinity.

    """
    args = tuple(raw.strip() for raw in argnames.split(","))
    argvalues, ids = zip(
        *[
            (
                values,
                "-".join(
                    f"{arg}={value}" for arg, value in zip(args, values, strict=False)
                ),
            )
            for values in product([True, False], repeat=len(args))
            if lite < 0 or sum(values) <= lite or all(values)
        ],
        strict=False,
    )

    return pytest.mark.parametrize(args, argvalues, ids=ids)


@pytest.fixture(scope="session", autouse=True)
def reproducibility() -> Iterator[None]:
    """Enhance reproducibility for the test session."""
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(mode=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.utils.deterministic.fill_uninitialized_memory = True  # type: ignore[attr-defined]
        matplotlib.use("agg")
        matplotlib.testing.set_font_settings_for_testing()
        matplotlib.testing.set_reproducibility_for_testing()
        yield


SEED = 0
DEVICES = []
if torch.cuda.is_available():  # pragma: no cover
    n = torch.cuda.device_count()
    DEVICES.append(f"cuda:{n - 1}")
DEVICES.append("cpu")


@pytest.fixture(
    scope="session",
    params=DEVICES,
    ids=lambda param: "cpu" if param == "cpu" else "cuda",
)
def device(request) -> torch.device:
    """Define the device for current test."""
    return torch.device(request.param)


@pytest.fixture
def generator(device) -> torch.Generator:
    """PyTorch random generator object for device."""
    return torch.Generator(device=device).manual_seed(SEED)


@pytest.fixture
def rng() -> np.random.Generator:
    """NumPy random generator object."""
    return np.random.default_rng(seed=SEED)


@pytest.fixture
def null() -> object:
    """Provide unique object for identity comparisons in tests."""
    return object()


class Weakable:
    """Weakable object factory."""

    __slots__ = ("__weakref__",)


@pytest.fixture
def make_weakable() -> type[Weakable]:
    """Make weakable object, collectible in tests after deletion.

    Do not return object directly since ``pytest`` would hold strong
    reference.
    """
    return Weakable


@pytest.fixture
def appender() -> Callable[[object], Callable[[str], str]]:
    r"""Maker of ``f"\n{obj}"`` appender."""

    def inner(obj: object) -> Callable[[str], str]:
        r"""Create function that appends ``f"\n{obj}"`` to input."""

        def _appender(msg: str) -> str:
            r"""Append ``f"\n{obj}"`` to ``msg``."""
            return f"{msg}\n{obj}"  # pragma: no cover

        return _appender

    return inner
