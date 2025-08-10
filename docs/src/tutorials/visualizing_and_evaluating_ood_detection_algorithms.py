"""
Visualizing & Evaluating OoD Detection algorithms
=================================================
"""

# %%
# .. hint:: We recommend reading
#    :doc:`/auto_tutorials/inferring_with_confidence` first.
#
# In this tutorial, we use the
# :doc:`/api_references/eval/classification` API from ``scio.eval`` to
# compare several confidence score algorithms in a classification setup,
# both **visually** and **quantitatively**.
#
# Let's start with preparing a trained model and some InD calibration
# data. These should be naturally defined by your own use-case. For this
# tutorial, we use a lightweight `Tiniest
# <https://github.com/xvel/cifar10-tiniest>`_ architecture trained on
# CIFAR10 and hosted on `our hub
# <https://github.com/ThalesGroup/scio/tree/hub>`_, and fetch the
# corresponding calibration data from our HuggingFace `dataset
# <https://huggingface.co/datasets/ego-thales/cifar10>`_, not seen
# during training.

# These should be defined by your use-case
import torch
from datasets import load_dataset  # type: ignore[import-untyped]

calib_set = load_dataset("ego-thales/cifar10", name="calibration")["unique_split"]
calib_data, calib_labels, calib_classnames = calib_set.with_format("torch")[:].values()
calib_data = calib_data / 255  # Convert [0, 255] uint8 from HuggingFace to [0, 1] float

sample_shape = calib_data.shape[1:]
net = torch.hub.load("ThalesGroup/scio:hub", "tiniest", trust_repo=True, verbose=False)
net = net.to(calib_data)

# %%
# 1. Configure algorithms to compare
# ----------------------------------
# Let use choose, say :math:`3` **confidence score algorithms** from
# those :ref:`implemented <confidence-score-algorithms-classif>` in
# ``scio.scores``. We will compare them to the
# :class:`~scio.scores.Softmax` baseline. We arbitrarily choose
# :class:`~scio.scores.GradNorm`, :class:`~scio.scores.Gram` and
# :class:`~scio.scores.KNN`. Each algorithm requires defining which
# internal representations it will analyze. Some authors may recommend
# the use of specific layers, while others propose general approaches
# free of choosing (*e.g.* :class:`~scio.scores.Gram`). Let us identify
# layers to select.

from scio.recorder import Recorder

Recorder(net, input_data=calib_data[[0]])  # Visualize layers

# %%

BLOCK3 = (1, 5)
FEATURE_MAPS = (1, 11)
LOGITS = (1, 12)

# %%
# Now we configure our algorithms, using default or standard parameters,
# and the *somewhat arbitrarily* identified layers for
# :class:`~scio.scores.Gram`. See
# :doc:`implementing_your_own_ood_detection_algorithm` to use your own
# algorithm!

from scio.scores import GradNorm, Gram, KNN, Softmax

scores_and_layers = (  # type: ignore[var-annotated]
    (Softmax(), []),
    (GradNorm(), [LOGITS]),
    (Gram(), [BLOCK3, FEATURE_MAPS, LOGITS]),
    (KNN(k=int(len(calib_data) ** 0.4)), [FEATURE_MAPS]),
)

# %%
# 2. Prepare score functions
# --------------------------
# With :func:`~scio.eval.fit_scores`, we can now easily calibrate the
# algorithms.

from scio.eval import fit_scores

scores_fit = fit_scores(scores_and_layers, net, calib_data, calib_labels)

# %%
# 3. Define InD and OoD scenarios
# -------------------------------
# Our InD scenario is already naturally defined as the leftover CIFAR10
# training data. We only take :math:`1\,000` random samples out of the
# :math:`10\,000` available.

test_set = load_dataset("ego-thales/cifar10", name="test")["unique_split"]
ind = test_set.with_format("torch").shuffle()[:1000]["image"] / 255

# %%
# As for OoD, we arbitrarily use vertical flips, darker images and
# uniformly random samples.

oods_title = ("Vertical flip", "Darker", "Uniformly random")
oods = (ind.flip(2), ind * 0.5, torch.rand_like(ind))

# %%
# 4. Compute confidence scores on scenarios
# -----------------------------------------
# We can easily compute confidence scores for all the prepared scenarios
# with :func:`~scio.eval.compute_confidence`.

from scio.eval import compute_confidence

confs_ind, confs_oods = compute_confidence(
    scores_fit,
    ind=ind,
    oods=oods,
    oods_title=oods_title,
)

# %%
# 5. Visualize the results
# ------------------------
# Now, we simply use :func:`~scio.eval.summary_plot` to visualize the
# results.

from matplotlib import rcParams

from scio.eval import summary_plot

rcParams["figure.figsize"] = (15, 9)  # Adjust tutorial layout
summary_plot(
    confs_ind,
    confs_oods,
    scores_and_layers=scores_and_layers,
    oods_title=oods_title,
)

# %%
# The first row shows the actual confidence scores distributions, one
# graph per scores function. Colors inside a given graph represent
# different scenarios: *InD*, *Vertical flip*, *Darker* and *Uniformly
# random*.
#
# The second row shows the ROC curves for the OoD Detection task (which
# is *in fine* a binary classification task). There is one graph per
# InD/OoD pair, so :math:`3` graphs here. In each graph, colors
# represent score functions: *Softmax*, *GradNorm*, *Gram* and *KNN*.
#
# This visualization provides very insightful details for experienced
# users. The next section will help with quantifying these results.
#
# .. important:: Confidence scores evaluation characterizes the ability
#    to identify OoD samples *based on the confidence scores associated
#    with the predictions of the model*. It provides **no information**
#    regarding the *correctness* of the predictions themselves.
#
# 6. Define metrics to get quantified results
# -------------------------------------------
# In ``scio.eval``, we :ref:`implemented
# <implemented-discriminative-power-metrics>` a few standard
# **Discriminative Power metrics**. Let us choose :math:`2` for our
# tutorial: a partial :class:`~scio.eval.AUC` and
# :class:`~scio.eval.TPR`\ :math:`@ 5\%`. See
# :doc:`implementing_your_own_discriminative_power_metric` to use your
# own metric.
#
# Using :func:`~scio.eval.compute_metrics` and
# :func:`~scio.eval.summary_table`, we get quantitative results from our
# confidence scores.

from scio.eval import AUC, TPR, compute_metrics, summary_table

metrics = (AUC(max_fpr=0.2), TPR(max_fpr=0.05))
evals = compute_metrics(confs_ind, confs_oods, metrics)
summary_table(
    evals,
    scores_and_layers=scores_and_layers,
    oods_title=oods_title,
    metrics=metrics,
)

# %%
# In each cell, the :math:`2` values correspond to the :math:`2` chosen
# metrics. Locally, you can also use the ``baseline`` option in
# :func:`~scio.eval.summary_table` for advanced CLI highlighting.
#
# 7. Be lazy
# ----------
# For good measure, we mention that :func:`~scio.eval.summary` directly
# performs the :func:`~scio.eval.summary_plot`,
# :func:`~scio.eval.compute_metrics` and
# :func:`~scio.eval.summary_table` calls described above,  **at once**!
#
# [Bonus] Profiling
# -----------------
# Let us compare the execution times of our :math:`4` algorithms
# (including the baseline) using the
# :attr:`~scio.scores.BaseScore.timer` attribute presented in
# :ref:`Inferring with Confidence
# <inferring-with-confidence-profiling>`.

for score, _ in scores_and_layers:
    score.timer.report  # Report execution times
    print("----")

# %%
# These facilitate overhead computation and provide decisive information
# when choosing an algorithm for a time-sensitive use-case.
