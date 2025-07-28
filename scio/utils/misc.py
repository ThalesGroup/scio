"""Module with misc utils."""

__all__ = [
    "HEAVY_HEAD_ROUNDED_BOTTOM",
    "ScoreTimer",
    "ScoreTimerStat",
    "check",
    "format_time",
]

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain
from math import ceil, log10
from time import perf_counter
from types import MappingProxyType
from typing import Literal, NamedTuple
from weakref import ref

import rich
from dill import copy  # type: ignore[import-untyped]
from rich.box import Box
from rich.highlighter import ReprHighlighter
from rich.table import Table

from .enums import ScoreTimerOperation, ScoreTimerOperationLike

HEAVY_HEAD_ROUNDED_BOTTOM = Box("┏━┳┓\n┃ ┃┃\n┡━╇┩\n│ ││\n├─┼┤\n├─┼┤\n│ ││\n╰─┴╯\n")


def check(condition: object, message: str = "") -> None:
    """Monkeypatch for ruff S101, originating in bandit."""
    if not condition:
        raise AssertionError(message)


class ScoreTimerStat(NamedTuple):
    """Timing statistic element for :class:`~scio.utils.ScoreTimer`.

    Fields
    ------
    op: ``ScoreTimerOperation``
        The operation that was timed. See
        :class:`~.enums.ScoreTimerOperation`.
    n_samples: ``int``
        The number of samples that were processed.
    duration: ``float``
        The duration of the timed execution.
    params: ``MappingProxyType[str, object]``
        The score parameters at the beginning of the execution.

    """

    op: ScoreTimerOperation
    n_samples: int
    duration: float
    params: MappingProxyType[str, object]


class ScoreTimer:
    """Timer for :class:`Score <scio.scores.BaseScore>` instances.

    Provides a context manager to easily time successive executions of
    :meth:`~scio.scores.BaseScore.fit` and
    :meth:`~scio.scores.BaseScore.__call__` for
    :class:`Score <scio.scores.BaseScore>` instances. For correct usage,
    see :meth:`__call__`.

    Timing statistics are respectively stored in :attr:`calibration` and
    :attr:`inference` attributes.

    Arguments
    ---------
    target: :class:`Score <scio.scores.BaseScore>` instance
        The target instance, for which execution should be timed.

    Note
    ----
    Different statistics from the same :class:`Score
    <scio.scores.BaseScore>` instance may originate from different
    contexts (parameters, calibration data, bound :attr:`rnet`, ...). If
    timing statistics need to be compared, it is up to the user to
    ensure full consistency since only the type of operation, the number
    of processed samples and the score's parameters are stored alongside
    the execution time.

    """

    def __init__(self, target: object) -> None:
        """Construct :class:`ScoreTimer` instance."""
        self._target_type_name = type(target).__qualname__
        self._target = ref(target)
        self._stats: list[ScoreTimerStat] = []

    @property
    def target(self) -> object | None:
        """The :class:`Score <scio.scores.BaseScore>` instance being timed.

        It is stored as a weak reference and evaluates to ``None`` if
        the target object is dead.
        """
        return self._target()

    @property
    def stats(self) -> tuple[ScoreTimerStat, ...]:
        """Timing statistics for this instance, from oldest to newest.

        Returns
        -------
        stats: ``tuple[ScoreTimerStat, ...]``
            See :class:`~scio.utils.misc.ScoreTimerStat`.

        """
        return tuple(self._stats)

    @property
    def calibration(self) -> tuple[ScoreTimerStat, ...]:
        """Timing statistics for calibrations.

        Returns
        -------
        calibration: ``tuple[ScoreTimerStat, ...]``
            See :class:`~scio.utils.misc.ScoreTimerStat`.

        """
        return tuple(
            stat for stat in self._stats if stat.op == ScoreTimerOperation.CALIBRATION
        )

    @property
    def inference(self) -> tuple[ScoreTimerStat, ...]:
        """Timing statistics for inferences.

        Returns
        -------
        inference: ``tuple[ScoreTimerStat, ...]``
            See :class:`~scio.utils.misc.ScoreTimerStat`.

        """
        return tuple(
            stat for stat in self._stats if stat.op == ScoreTimerOperation.INFERENCE
        )

    @contextmanager
    def __call__(
        self,
        op: ScoreTimerOperationLike,
        /,
        *,
        n_samples: int,
    ) -> Iterator[None]:
        """Context manager to time execution and store results.

        Arguments
        ---------
        op: ``ScoreTimerOperationLike``
            See :class:`~.enums.ScoreTimerOperation`.
        n_samples: ``int``
            Number of samples to be processed during ``op``.

        Example
        -------
        ::

            with timer("calibration", n_calib):
                # Calibration procedure...

        Raises
        ------
        :exc:`RuntimeError`
            If the instance's :attr:`target` is already dead.
        """
        if (target := self.target) is None:
            msg = f"Target {self._target_type_name!r} object is dead"
            raise RuntimeError(msg)

        if not hasattr(target, "params"):
            msg = (
                f"Target {target} is missing a required 'params' attribute. Hint: "
                f"{type(self).__qualname__} instances should only target Score"
                " instances"
            )
            raise AttributeError(msg)

        params = copy(MappingProxyType(target.params))  # type: ignore[attr-defined]

        # Time execution
        start = perf_counter()
        yield None
        duration = perf_counter() - start

        # Store result
        stat = ScoreTimerStat(
            op=ScoreTimerOperation(op),
            n_samples=n_samples,
            duration=duration,
            params=params,
        )
        self._stats.append(stat)

    @property
    def report(self) -> None:
        """Show statistics report in rich table.

        Example
        -------
        ::

            >>> # For a given `gram` instance already in use
            >>> gram.timer.report
               ScoreTimer report for Gram(separate_diagonal=False,
            cut_off=0.1, act_norm=2, calib_labels='pred', mode='raw')
                                at 0x77346d30e490
            ┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
            ┃ Extra params     ┃  Operation  ┃ # samples ┃ Duration ┃
            ┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
            │ max_gram_order=2 │  inference  │        20 │ 115.4 ms │
            │                  │  inference  │      1352 │ 6.585 s  │
            │                  │ calibration │        50 │ 302.9 ms │
            ├──────────────────┼─────────────┼───────────┼──────────┤
            │ max_gram_order=4 │  inference  │        20 │ 246.5 ms │
            │                  │  inference  │      1352 │ 13.78 s  │
            │                  │ calibration │       677 │ 6.819 s  │
            └──────────────────┴─────────────┴───────────┴──────────┘
                    Entries are listed from newest to oldest

        Note
        ----
        Shows nothing if there are no stats.

        Warning
        -------
        The behaviour is only guaranteed for classical (*e.g.*
        nonrecursive) parameter values.

        """
        stats = self.stats[::-1]  # Reversed for most recent at the top

        if not stats:
            return

        # Prepare params display by moving common parameters to title
        all_params = [stat.params for stat in stats]
        all_attrs = sorted({attr for params in all_params for attr in params})
        null = object()
        common = {}
        for attr in all_attrs:
            attr_values = [params.get(attr, null) for params in all_params]
            if all(value == attr_values[0] for value in attr_values):
                common[attr] = attr_values[0]

        common_str = ", ".join(f"{k}={v!r}" for k, v in common.items())
        params_strs = [
            ", ".join(f"{k}={v!r}" for k, v in params.items() if k not in common)
            for params in all_params
        ]
        no_extra = not any(params_strs)

        # Table
        title_raw = f"ScoreTimer report for {self._target_type_name}({common_str})"
        if (target := self.target) is not None:
            title_raw += f"\nat {hex(id(target))}"
        title = ReprHighlighter()(title_raw).markup
        caption = "Entries are listed from newest to oldest"
        table = Table(
            title=title,
            caption=caption,
            highlight=True,
            show_lines=True,
            box=HEAVY_HEAD_ROUNDED_BOTTOM,
            safe_box=False,
        )

        # Headers
        headers = ("Extra params", "Operation", "# samples", "Duration")
        type Align = Literal["center", "left", "right"]
        aligns: tuple[Align, ...] = ("left", "center", "right", "center")
        for header, align in zip(headers[no_extra:], aligns[no_extra:], strict=True):
            table.add_column(header, justify=align)

        # Add rows (merge consecutive lines with similar params)
        lines: list[tuple[str, str, str]] = []
        prev_params_str = params_strs[0]
        for elt in chain(zip(stats, params_strs, strict=True), [None]):
            # Add row when exiting loop (``None``) or upon new params
            if elt is None or prev_params_str != elt[-1]:
                row_cells = prev_params_str, *map("\n".join, zip(*lines, strict=True))
                table.add_row(*row_cells[no_extra:])
                lines = []

            if elt is not None:
                (op, n_samples, duration, _), params_str = elt
                lines.append((op.value, str(n_samples), format_time(duration)))
                prev_params_str = params_str

        rich.print(table)


# The function below was copied and adapted from
# https://github.com/eliegoudout/lasvegas/blob/dev/perf/__init__.py
# Original license: MIT
#
# Copyright (c) 2023 Élie Goudout
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def format_time(duration: float, num_digits: int = 4) -> str:
    """Format a short ``float`` duration into a readable ``str``.

    Arguments
    ---------
    duration: ``float``
        Expressed in seconds, duration to format. Must satisfy ``0 <=
        duration < 10 ** (n + 1) - 0.5``.
    num_digits: ``int``
        Number of significant digits to display. Larger durations can
        have one more and shorter durations less (see examples). Must be
        at least ``3``. Defaults to ``4``.

    Returns
    -------
    out: ``str``
        Formated duration -- *e.g.* ``"567.9 ms"``.

    Raises
    ------
    :exc:`AssertionError`:
        If either ``num_digits < 3`` or
        ``not 0 <= duration < 10 ** (n + 1) - 0.5``.

    Examples
    --------
    With ``num_digits=4``::

        ╭───────────────┬────────────────┬───────────────────────────────────────╮
        │   Duration    │     Result     │                  Comment              │
        ├───────────────┼────────────────┼───────────────────────────────────────┤
        │      1.5      │    1.500 s     │ Significant 0's added                 │
        │      0.56789  │    567.9 ms    │ Last digit is rounded...              │
        │      0.99995  │    1.000 s     │ ...which can lead to precision loss   │
        │      0.12345  │    123.4 ms    │ Rounds half to even (python built-in) │
        │   1234        │    1234. s     │ Point is added for constant witdh     │
        │  12345        │    12345 s     │ One more digit for longer durations   │
        │ 123456        │ AssertionError │ Exceeded max duration                 │
        │     -1        │ AssertionError │ Negative duration                     │
        │      0        │    0.000 as    │ Smallest unit for shorter durations   │
        │      5.67e-20 │    0.057 as    │ Precision is worse near 0.            │
        ╰───────────────┴────────────────┴───────────────────────────────────────╯

    Notes
    -----
    Implementation heavily relies on following facts:

    - Consecutive units have constant ratio of ``10**3``,
    - Highest unit is the unit of ``duration``'s encoding.

    """
    units = ["s", "ms", "μs", "ns", "ps", "fs", "as"]
    max_pow = 3 * (len(units) - 1)
    n = num_digits
    check(n >= 3, "Display at least 3 digits")  # noqa: PLR2004 (magic number)
    check(0 <= duration < 10 ** (n + 1) - 0.5, "Duration out of bounds")

    # Special case 0
    if duration == 0:
        return f"{0:.{n - 1}f} " + units[-1]

    # Retrieve left shift for significant part
    left_shift = ceil(-log10(duration)) + n - 1
    significant = round(duration * 10**left_shift)
    if significant == 10**n:  # Special case `0.0099996` -> `'10.00ms'`
        significant //= 10
        left_shift -= 1

    # If `duration` is barely too big: remove floating point
    if left_shift < 0:
        return f"{round(duration)} " + units[0]

    # Nominal case
    if left_shift < max_pow + n:
        unit_index = max(0, 1 + (left_shift - n) // 3)
        y = significant * 10 ** (3 * unit_index - left_shift)
        n_left = int(log10(y) + 1)
        unit = units[unit_index]
        return f"{y:.{max(0, n - n_left)}f}{'.' if n == n_left else ''} " + unit

    # If so small that smallest unit loses precision
    return f"{duration * 10**max_pow:.{n - 1}f} " + units[-1]
