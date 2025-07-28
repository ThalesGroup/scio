"""Utility for benchmarking purposes on classification scores."""

__all__ = [
    "compute_confidence",
    "compute_metrics",
    "fit_scores",
    "histogram_oods",
    "roc_scores",
    "summary",
    "summary_plot",
    "summary_table",
]

from collections.abc import Collection, Iterable
from functools import partial
from itertools import chain, repeat, starmap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import seaborn as sns  # type: ignore[import-untyped]
from numpy.typing import NDArray
from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from torch import Tensor, nn

from scio.recorder import DepthIdx, Recorder
from scio.scores import BaseScoreClassif
from scio.utils.misc import HEAVY_HEAD_ROUNDED_BOTTOM

from .discriminative_power import BaseDiscriminativePower
from .roc import ROC

type ScoreClassifAndLayers = tuple[BaseScoreClassif, Collection[DepthIdx]]


def _pretty(text: str) -> str:
    """Markup for ``text`` highlighted by :class:`~rich.highlighter.ReprHighlighter`."""
    return ReprHighlighter()(text).markup


def fit_scores(
    scores_and_layers: Collection[ScoreClassifAndLayers],
    net: nn.Module,
    calib_data: Tensor,
    calib_labels: Tensor,
    *,
    show_progress: bool = True,
) -> list[BaseScoreClassif]:
    """Fit confidence scores given net and calibration data.

    Arguments
    ---------
    scores_and_layers: ``Collection[ScoreClassifAndLayers]``
        Instanciated classification score algorithms with their
        associated layers to record. Example::

            scores_and_layers = [
                (IsoMax(), []),
                (KNN(k=6), [(1, 1)]),
            ]

    net: ``nn.Module``
        Neural net.
    calib_data: ``Tensor``
        Calibration data (InD).
    calib_labels: ``Tensor``
        Calibration labels.
    show_progress: ``bool``
        Whether to show progress bar. Defaults to ``True``.

    Returns
    -------
    scores_fit: ``list[BaseScoreClassif]``
        Fit score functions.

    """
    scores_fit = []

    columns = (
        SpinnerColumn(),
        TimeElapsedColumn(),
        _pretty(f"Fitting Scores... ({len(calib_data)} samples)"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.description}"),
    )
    with Progress(*columns, disable=not show_progress) as progress:
        task = progress.add_task("", total=len(scores_and_layers))
        score_task = partial(progress.update, task, refresh=True)

        for score, layers in scores_and_layers:
            score_task(description=_pretty(_score_and_layers_str(score, layers)))

            rnet = Recorder(net, input_data=calib_data[[0]])
            rnet.record(*layers)
            score.fit(rnet, calib_data, calib_labels)
            scores_fit.append(score)
            score_task(advance=1)

        progress.columns = progress.columns[1:]  # Remove empty spinner
        score_task(description="Done")

    return scores_fit


def compute_confidence(
    scores_fit: Collection[BaseScoreClassif],
    *,
    ind: Tensor,
    oods: tuple[Tensor, ...],
    oods_title: Collection[str] | None = None,
    show_progress: bool = True,
) -> tuple[NDArray, tuple[NDArray, ...]]:
    """Compute the prediction confidence scores for given InD and OoDs.

    Arguments
    ---------
    scores_fit: ``Collection[BaseScoreClassif]``
        Fit scores.
    ind: ``Tensor``
        In-Distribution data.
    oods: ``tuple[Tensor, ...]``
        Out-of-Distribution data. Each element is a ``Tensor``
        representing one OoD dataset.
    oods_title: ``Collection[str]``, optional
        Title of OoD sets. Used only for progress bar details.
    show_progress: ``bool``
        Whether to show progress bar. Defaults to ``True``.

    Returns
    -------
    confs_ind: ``NDArray``
        The scores associated with the net's predicted class for
        In-Distribution data. Shape ``(n_scores, n_ind_samples)``.
    confs_oods: ``tuple[NDArray, ...]``
        Analogue for Out-of-Distribution data (one element per element
        in ``oods``).

    """
    if oods_title is None:
        oods_title = ("",) * len(oods)

    confs_ind_list: list[NDArray] = []
    confs_oods_list: list[list[NDArray]] = [[] for _ in oods]

    # Loop over scores
    columns = (
        SpinnerColumn(),
        TimeElapsedColumn(),
        _pretty("Computing confidences..."),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.description}"),
    )
    with Progress(*columns, disable=not show_progress) as progress:
        task = progress.add_task("", total=len(scores_fit))
        score_task = partial(progress.update, task, refresh=True)

        for score in scores_fit:
            layers = score.rnet.recording
            score_task(description=_pretty(_score_and_layers_str(score, layers)))

            task = progress.add_task("", total=1 + len(oods))
            data_task = partial(progress.update, task, refresh=True)
            data_task(description=_pretty(f"↳ In-Distribution ({len(ind)} samples)"))

            # Compute on InD
            out_ind, conf_ind = score(ind)
            pred_conf_ind = conf_ind[range(len(ind)), out_ind.argmax(1)]
            confs_ind_list.append(pred_conf_ind.numpy(force=True))
            data_task(advance=1)

            # Loop over OoD sets
            for ood, confs_ood_list, (i, ood_title) in zip(
                oods,
                confs_oods_list,
                enumerate(oods_title, start=1),
                strict=True,
            ):
                ood_str = f"OoD {i}{f': {ood_title}' if ood_title else ''}"
                data_task(description=_pretty(f"↳ {ood_str} ({len(ood)} samples)"))

                out_ood, conf_ood = score(ood)
                pred_conf_ood = conf_ood[range(len(ood)), out_ood.argmax(1)]
                confs_ood_list.append(pred_conf_ood.numpy(force=True))
                data_task(advance=1)

            progress.remove_task(data_task.args[0])
            score_task(advance=1)

        progress.columns = progress.columns[1:]  # Remove empty spinner
        score_task(description="Done")

    stack = partial(np.stack, dtype=float)
    confs_ind = stack(confs_ind_list)  # Shape (n_scores, n_ind_samples)
    confs_oods = tuple(map(stack, confs_oods_list))  # Shapes (n_scores, n_oodi_samples)
    return confs_ind, confs_oods


def compute_metrics(
    confs_ind: NDArray,
    confs_oods: tuple[NDArray, ...],
    metrics: tuple[BaseDiscriminativePower, ...],
) -> NDArray[np.floating]:
    """From precomputed confidence scores, evaluate scores.

    Arguments
    ---------
    confs_ind: ``NDArray``
        First output of :func:`compute_confidence`.
    confs_oods: ``tuple[NDArray, ...]``
        Second output of :func:`compute_confidence`.
    metrics: ``tuple[BaseDiscriminativePower, ...]``
        The different types of metrics to compute for every ``(score,
        ood)`` combination.

    Returns
    -------
    evals: ``NDArray[np.floating]``
        Computed metrics for every ``(score, ood_set, metric)`` possible.
        Shape corresponding to ``(n_scores, n_ood_sets, n_metrics)``.

    """
    evals = np.empty((len(confs_ind), len(confs_oods), len(metrics)))
    for conf_ind, *confs_ood, metrics_score in zip(
        confs_ind,
        *confs_oods,
        evals,
        strict=False,
    ):
        for conf_ood, metrics_ood in zip(confs_ood, metrics_score, strict=False):
            labels = [False] * len(conf_ind) + [True] * len(conf_ood)
            scores = np.concatenate([conf_ind, conf_ood])
            roc = ROC(labels, scores)
            metrics_ood[:] = [metric.from_roc(roc) for metric in metrics]

    return evals


def summary_table(
    evals: NDArray[np.floating],
    *,
    scores_and_layers: Iterable[ScoreClassifAndLayers] | None = None,
    oods_title: Iterable[str] | None = None,
    metrics: Iterable[BaseDiscriminativePower] | None = None,
    baseline: int | None = None,
) -> None:
    """Print scores evaluation results summary in rich table.

    Arguments
    ---------
    evals: ``NDArray[np.floating]``
        Result from a :func:`compute_metrics` call. Shape is
        ``(n_scores, n_ood_sets, n_metrics)``.
    scores_and_layers: ``Iterable[ScoreClassifAndLayers]``, optional
        See :func:`fit_scores`. Used only for row headers.
    oods_title: ``Iterable[str]``, optional
        See :func:`compute_confidence`. Used only for column headers.
    metrics: ``Iterable[BaseDiscriminativePower]``, optional
        Metrics used to compute ``evals`` in :func:`compute_metrics`.
        For highlight purposes, elements should take values in
        :math:`[0, 1]` and be to *maximize*. Used only for table title.
    baseline: ``int``, optional
        The index of the baseline score, for advanced highlighting.

    """
    n_scores, n_ood_sets, n_metrics = evals.shape

    # Preprocess optional arguments
    recorded = scores_and_layers is not None

    scores_str: Iterable[str]
    if scores_and_layers is None:
        scores_str = (f"Score {i + 1}" for i in range(n_scores))
    else:
        scores_str = starmap(_score_and_layers_str, scores_and_layers)

    if oods_title is None:
        oods_title = repeat("", times=n_ood_sets)

    title_str = (
        f"[i]Evaluation of {n_scores} scores against {n_ood_sets} OoD sets and "
        f"{n_metrics} metrics[/i]"
    )

    if metrics is not None:
        title_str += f"[i]:[/i]\n{' / '.join(map(str, metrics))}"

    # Create columns with headers
    title = Console().render_str(title_str)
    table = Table(
        title=title,
        highlight=True,
        show_lines=True,
        caption_justify="left",
        box=HEAVY_HEAD_ROUNDED_BOTTOM,
        safe_box=False,
    )
    table.add_column("Scores" + ("\n↳ Recorded layers" if recorded else ""))
    for i, ood_title in enumerate(oods_title):
        ood_str = f"OoD {i + 1}{f':\n{ood_title}' if ood_title else ''}"
        table.add_column(ood_str, justify="center", vertical="middle")

    # Results highlighting: masks > funcs > logic
    gold_mask = evals == evals.max(0)
    uline_mask = np.full(evals.shape, fill_value=False)
    bold_mask = True
    if baseline is not None:
        uline_mask[baseline] = True
        bold_mask = evals > evals[baseline]

    stringify = np.vectorize("{:.3f}".format)
    golden = np.vectorize(lambda s, do: f"[gold3]{s}[/gold3]" if do else s)
    bold = np.vectorize(lambda s, do: f"[b]{s}[/b]" if do else f"[not b]{s}[/not b]")
    uline = np.vectorize(lambda s, do: f"[u]{s}[/u]" if do else f"[not u]{s}[/not u]")

    elts = stringify(evals)
    elts = golden(elts, gold_mask)
    elts = uline(elts, uline_mask)
    elts = bold(elts, bold_mask)

    # Fill table
    for score_str, elts_score in zip(scores_str, elts, strict=False):
        table.add_row(score_str.strip(), *map(" / ".join, elts_score))

    # Show
    rich.print(table)


def histogram_oods(
    conf_ind: NDArray,
    conf_oods: tuple[NDArray, ...],
    *,
    oods_title: tuple[str, ...] | None = None,
    score_and_layers: ScoreClassifAndLayers | None = None,
    **hist_kw: object,
) -> plt.Axes:
    """For a given score, plot histograms over all OoD sets.

    Arguments
    ---------
    conf_ind: ``NDArray``
        Confidence scores of In-Distribution samples. Shape
        ``(n_ind_samples,)``.
    conf_oods: ``tuple[NDArray, ...]``
        Same but for iterable of Out-of-Distribution samples. Shapes
        ``(n_oodi_samples,)``.
    oods_title: ``tuple[str, ...]``, optional
        See :func:`compute_confidence`. Used only for legend purposes.
    score_and_layers: ``ScoreClassifAndLayers``, optional
        The score (and layers) used to compute the confidence scores.
        Example::

            score_and_layers = KNN(k=6), [(1, 1)]

        Used only for legend purposes.
    **hist_kw:
        Passed to :func:`sns.histplot`.

    Returns
    -------
    ax: ``plt.Axes``
        The matplotlib axes containing the plot.

    """
    # Preprocess optional arguments
    if oods_title is None:
        oods_title = tuple(f"OoD {i + 1}" for i in range(len(conf_oods)))

    if score_and_layers is None:
        title = ""
    else:
        score, layers = score_and_layers
        title = (f"{score}\n↳ " + ", ".join(map(str, layers))).strip("\n↳ ")

    # Histogram
    conf_all = (conf_ind, *conf_oods)
    confidence_score = np.concatenate(conf_all)
    dataset = np.repeat(
        ("In-Distribution", *oods_title),
        [len(conf) for conf in conf_all],
    )
    frame = pd.DataFrame({"Confidence score": confidence_score, "Dataset": dataset})
    ax = sns.histplot(frame, x="Confidence score", hue="Dataset", **hist_kw)
    ax.set_title(title)
    return ax


def roc_scores(  # noqa: PLR0913 (too many arguments)
    confs_ind: NDArray,
    confs_ood: NDArray,
    *,
    scores_and_layers: Iterable[ScoreClassifAndLayers] | None = None,
    ood_title: str | None = None,
    legend: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """For a given OoD set, plot ROCs over all scores.

    Arguments
    ---------
    confs_ind: ``NDArray``
        Confidence scores on In-Distribution data. Shape ``(n_scores,
        n_ind_samples)``.
    confs_ood: ``NDArray``
        Confidence scores on Out-of-Distribution data. Shape ``(n_scores,
        n_ood_samples)``.
    scores_and_layers: ``Iterable[ScoreClassifAndLayers]``, optional
        Scores (and layers) used to compute ``confs_*`` in
        :func:`compute_confidence`. Used only for legend purposes.
    ood_title: ``str``, optional
        Title of the OoD set related to ``confs_ood``. Used only for the
        plot title.
    legend: ``bool``
        Whether or not to show legend. Defaults to ``True``.
    ax: ``plt.Axes``, optional
        If provided, ROCs are plotted on this ``ax``.

    Returns
    -------
    ax: ``plt.Axes``
        The matplotlib axes containing the plot.

    """
    # Preprocess optional arguments
    if ood_title is None:
        ood_title = "Out-of-Distribution"

    if ax is None:
        ax = plt.gca()

    scores_str: Iterable[str]
    if scores_and_layers is None:
        scores_str = (f"Score {i + 1}" for i in range(len(confs_ind)))
    else:
        scores_str = starmap(_score_and_layers_str, scores_and_layers)

    # ROCs
    rocs = []
    for conf_ind, conf_ood in zip(confs_ind, confs_ood, strict=False):
        labels = [False] * len(conf_ind) + [True] * len(conf_ood)
        confs = np.concatenate([conf_ind, conf_ood])
        rocs.append(ROC(labels, confs))

    # Layout
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(ood_title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    # Dummy
    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.5)

    colors = sns.color_palette(n_colors=len(rocs))
    for roc, color, score_str in zip(rocs, colors, scores_str, strict=False):
        # Pareto
        ax.scatter(roc.FPR, roc.TPR, s=5, color=color, label=score_str)
        ax.step(*np.c_[[roc.FPR, roc.TPR], [1, 1]], color=color, where="post", lw=1)

        # Convex Hull
        ax.plot(*np.c_[[roc.FPRch, roc.TPRch], [1, 1]], color=color, ls=":")

    if legend:
        ax.legend(title="Scores")

    return ax


def summary_plot(  # noqa: PLR0913 (too many arguments)
    confs_ind: NDArray,
    confs_oods: tuple[NDArray, ...],
    *,
    scores_and_layers: tuple[ScoreClassifAndLayers, ...] | None = None,
    oods_title: tuple[str, ...] | None = None,
    legend: tuple[bool, bool] | bool = True,
    block: bool | None = None,
    **hist_kw: object,
) -> None:
    """Plot and show histograms for each score, ROCs for each OoD set.

    Arguments
    ---------
    confs_ind: ``NDArray``
        First output of :func:`compute_confidence`.
    confs_oods: ``tuple[NDArray, ...]``
        Second output of :func:`compute_confidence`.
    scores_and_layers: ``tuple[ScoreClassifAndLayers, ...]``, optional
        See :func:`roc_scores`.
    oods_title: ``tuple[str, ...]``, optional
        See :func:`histogram_oods`.
    legend: ``tuple[bool, bool] | bool``
        Whether to show legends for histograms and ROCs respectively.
        If a unique ``bool`` is provided, it is used for both. Defaults
        to ``True``.
    block: ``bool``, optional
        Passed to :func:`plt.show`.
    **hist_kw:
        Passed to :func:`sns.histplot`, except the ``ax`` kwarg. Unless
        overidden, the following values are also passed: ``bins=30``,
        ``stat="density"`` and ``common_norm=False``.

    Note
    ----
    In its current state, :func:`summary_plot` may render cropped or
    incomplete legends when working with a lot of scores or OoD sets. In
    this case, you may wish to hide one or both legends with the
    ``legend`` option.

    """
    legend_hist, legend_roc = (legend, legend) if isinstance(legend, bool) else legend

    # Create axes
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1)
    gs_hist = gs[0].subgridspec(1, len(confs_ind))
    gs_rocs = gs[1].subgridspec(1, len(confs_oods))
    axes_hist = list(map(fig.add_subplot, iter(gs_hist)))
    axes_rocs = list(map(fig.add_subplot, iter(gs_rocs)))

    # Plots
    scores_iter = repeat(None) if scores_and_layers is None else scores_and_layers
    hist_kw_final = {"bins": 30, "stat": "density", "common_norm": False} | hist_kw
    hist_kw_final.pop("ax", None)
    for conf_ind, *conf_oods_list, score_and_layers, ax in zip(
        confs_ind,
        *confs_oods,
        scores_iter,
        axes_hist,
        strict=False,
    ):
        conf_oods = tuple(conf_oods_list)
        histogram_oods(
            conf_ind,
            conf_oods,
            oods_title=oods_title,
            score_and_layers=score_and_layers,
            legend=legend_hist,
            ax=ax,
            **hist_kw_final,
        )
        if legend_hist:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(0, 1))

    oods_title_iter = repeat(None) if oods_title is None else oods_title
    for confs_ood, ood_title, ax in zip(
        confs_oods,
        oods_title_iter,
        axes_rocs,
        strict=False,
    ):
        roc_scores(
            confs_ind,
            confs_ood,
            scores_and_layers=scores_and_layers,
            ood_title=ood_title,
            legend=legend_roc,
            ax=ax,
        )

    # Layout
    axes_hist[0].set_yticks([])
    axes_rocs[-1].legend(loc="lower right", bbox_to_anchor=(1, 0))
    for ax in chain(axes_hist[1:], axes_rocs[1:]):
        ax.get_yaxis().set_visible(False)
        if (ax_legend := ax.get_legend()) is not None:
            ax_legend.remove()

    plt.subplots_adjust(0.04, 0.06, 0.99, 0.94, wspace=0)
    plt.show(block=block)


def summary(  # noqa: PLR0913 (too many arguments)
    confs_ind: NDArray,
    confs_oods: tuple[NDArray, ...],
    *,
    scores_and_layers: tuple[ScoreClassifAndLayers, ...] | None = None,
    oods_title: tuple[str, ...] | None = None,
    metrics: tuple[BaseDiscriminativePower, ...] | None = None,
    baseline: int | None = None,
    legend: tuple[bool, bool] | bool = True,
    block: bool | None = None,
    **hist_kw: object,
) -> None:
    """Print evaluation table, plot and show histograms and ROCs.

    For arguments specification, refer to :func:`compute_metrics`,
    :func:`summary_table` and :func:`summary_plot`. If ``metrics`` is
    not provided, no evaluation table is computed, in which case this is
    equivalent to a direct :func:`summary_plot` call.
    """
    if metrics is not None:
        evals = compute_metrics(confs_ind, confs_oods, metrics=metrics)
        summary_table(
            evals,
            scores_and_layers=scores_and_layers,
            oods_title=oods_title,
            metrics=metrics,
            baseline=baseline,
        )

    summary_plot(
        confs_ind,
        confs_oods,
        scores_and_layers=scores_and_layers,
        oods_title=oods_title,
        legend=legend,
        block=block,
        **hist_kw,
    )


def _score_and_layers_str(score: BaseScoreClassif, layers: Collection[DepthIdx]) -> str:
    """Give ``str`` representation for a score with layers.

    Example
    -------
    ::

        DeepMahalanobis(act_norm=2, mode='diff', weights=[1])
        ↳ (1, 10)

    """
    out = str(score)
    if layers:
        out += "\n↳ " + ", ".join(map(str, layers))

    return out
