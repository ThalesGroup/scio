"""Test classification benchmarking functions."""

import matplotlib.pyplot as plt
import numpy as np

from scio.eval import (
    compute_confidence,
    compute_metrics,
    fit_scores,
    histogram_oods,
    roc_scores,
    summary,
    summary_plot,
    summary_table,
)
from test.conftest import parametrize_bool

from .conftest import BASELINE, HIST_KW, OODS_TITLE, OPS


def test_fit_scores(scores_and_layers, net, calib_data, calib_labels, show_progress):
    """Test :func:`fit_scores` works and uses ``calib_data``."""
    scores_fit = fit_scores(
        scores_and_layers,
        net,
        calib_data,
        calib_labels,
        show_progress=show_progress,
    )

    # Check is fit and expectation (see :class:`ScoreFactory`)
    out = net(calib_data)
    for score, op in zip(scores_fit, OPS, strict=True):
        assert score.fit
        expected = op(calib_data) + op(out)
        assert score.observed == expected


@parametrize_bool("with_titles")
def test_compute_confidence(scores_fit, ind, oods, show_progress, with_titles, net):
    """Test :func:`compute_confidence` output."""
    confs_ind, confs_oods = compute_confidence(
        scores_fit,
        ind=ind,
        oods=oods,
        oods_title=OODS_TITLE if with_titles else None,
        show_progress=show_progress,
    )

    # Check shapes
    assert confs_ind.shape == (len(scores_fit), len(ind))
    for confs_ood, ood in zip(confs_oods, oods, strict=True):
        assert confs_ood.shape == (len(scores_fit), len(ood))

    # Check expectations (see :class:`ScoreFactory`)
    out_ind = net(ind)
    out_oods = tuple(map(net, oods))
    for i, op in enumerate(OPS):
        # InD
        expected = op(out_ind, dim=1).numpy(force=True)
        np.testing.assert_equal(confs_ind[i], expected, err_msg="[InD]")

        # OoDs
        for confs_ood, ood_title, out_ood in zip(
            confs_oods,
            OODS_TITLE,
            out_oods,
            strict=True,
        ):
            expected = op(out_ood, dim=1).numpy(force=True)
            np.testing.assert_equal(confs_ood[i], expected, err_msg=f"[{ood_title}]")


def test_compute_metrics(confs_ind, confs_oods, metrics):
    """Test :func:`compute_metrics` works and uses args."""
    evals = compute_metrics(confs_ind, confs_oods, metrics)

    # Check shapes
    assert evals.shape == (len(confs_ind), len(confs_oods), len(metrics))

    # Check expectation (see :class:`TestMetric`)
    for i, confs_ood in enumerate(confs_oods):
        evals_ood = evals[:, i].squeeze()
        expected = np.c_[confs_ood.max(1), confs_ind.min(1)]
        np.testing.assert_equal(evals_ood, expected, err_msg=f"[OoD {i}]")


@parametrize_bool("with_scores, with_titles, with_metrics, with_baseline", lite=1)
def test_summary_table(
    evals,
    scores_and_layers,
    metrics,
    with_scores,
    with_titles,
    with_metrics,
    with_baseline,
    match_outerr_torch,  # noqa: ARG001 (unused argument)
):
    """Test :func:`summary_table` output.

    Warning
    -------
    It is insensitive to markup formatting.

    """
    none = summary_table(
        evals,
        scores_and_layers=scores_and_layers if with_scores else None,
        oods_title=OODS_TITLE if with_titles else None,
        metrics=metrics if with_metrics else None,
        baseline=BASELINE if with_baseline else None,
    )
    assert none is None


@parametrize_bool(
    "with_oods_title, with_score_and_layers, with_hist_kw, with_ax",
    lite=1,
)
def test_histogram_oods(
    confs_ind,
    confs_oods,
    scores_and_layers,
    with_oods_title,
    with_score_and_layers,
    with_hist_kw,
    with_ax,
    match_plots_torch,  # noqa: ARG001 (unused argument)
):
    """Test :func:`histogram_oods` plots."""
    conf_ind = confs_ind[BASELINE]
    conf_oods = tuple(confs_ood[BASELINE] for confs_ood in confs_oods)
    score_and_layers = scores_and_layers[BASELINE]
    hist_kw = HIST_KW.copy() if with_hist_kw else {}
    if with_ax:
        hist_kw["ax"] = ax = plt.subplot(1, 2, 1)
    out = histogram_oods(
        conf_ind,
        conf_oods,
        oods_title=OODS_TITLE if with_oods_title else None,
        score_and_layers=score_and_layers if with_score_and_layers else None,
        **hist_kw,
    )

    if with_ax:
        assert out is ax

    plt.show()


@parametrize_bool(
    "with_scores_and_layers, with_ood_title, convex_hull, with_ax",
    lite=1,
)
def test_roc_scores(
    confs_ind,
    confs_oods,
    scores_and_layers,
    with_scores_and_layers,
    with_ood_title,
    convex_hull,
    with_ax,
    match_plots_torch,  # noqa: ARG001 (unused argument)
):
    """Test :func:`roc_scores` plots."""
    ax = plt.subplot(1, 2, 1) if with_ax else None
    out = roc_scores(
        confs_ind,
        confs_oods[BASELINE],
        scores_and_layers=scores_and_layers if with_scores_and_layers else None,
        ood_title=OODS_TITLE[BASELINE] if with_ood_title else None,
        legend=with_scores_and_layers,
        convex_hull=convex_hull,
        ax=ax,
    )

    if with_ax:
        assert out is ax

    plt.show()


@parametrize_bool(
    "with_scores_and_layers, with_oods_title, convex_hull, show, with_hist_kw",
    lite=1,
)
def test_summary_plot(
    confs_ind,
    confs_oods,
    scores_and_layers,
    with_scores_and_layers,
    with_oods_title,
    convex_hull,
    show,
    with_hist_kw,
    match_plots_torch,  # noqa: ARG001 (unused argument)
):
    """Test :func:`summary_plot` plots."""
    hist_kw = HIST_KW if with_hist_kw else {}
    if any((with_scores_and_layers, with_oods_title, convex_hull, show, with_hist_kw)):
        legends = [True]
    else:
        legends = (True, False, (True, False), (False, True))

    for legend in legends:
        none = summary_plot(
            confs_ind,
            confs_oods,
            scores_and_layers=scores_and_layers if with_scores_and_layers else None,
            oods_title=OODS_TITLE if with_oods_title else None,
            legend=legend,
            convex_hull=convex_hull,
            show=show,
            block=False,
            **hist_kw,
        )
        assert none is None
        plt.show()


# github.com/pytest-dev/pytest/issues/13537
# Error upon double teardown skips when updating both outerr and plots
# No problem, update still works as intended
@parametrize_bool(
    "with_scores_and_layers, with_oods_title, with_metrics, with_baseline, "
    "optimal_only, convex_hull, show, with_hist_kw",
    lite=1,
)
def test_summary(
    confs_ind,
    confs_oods,
    scores_and_layers,
    metrics,
    with_scores_and_layers,
    with_oods_title,
    with_metrics,
    with_baseline,
    optimal_only,
    convex_hull,
    show,
    with_hist_kw,
    match_outerr_torch,  # noqa: ARG001 (unused argument)
    match_plots_torch,  # noqa: ARG001 (unused argument)
):
    """Test :func:`summary` outerr and plots."""
    hist_kw = HIST_KW if with_hist_kw else {}
    none = summary(
        confs_ind,
        confs_oods,
        scores_and_layers=scores_and_layers if with_scores_and_layers else None,
        oods_title=OODS_TITLE if with_oods_title else None,
        metrics=metrics if with_metrics else None,
        baseline=BASELINE if with_baseline else None,
        optimal_only=optimal_only,
        convex_hull=convex_hull,
        show=show,
        block=False,
        **hist_kw,
    )
    assert none is None
    plt.show()
