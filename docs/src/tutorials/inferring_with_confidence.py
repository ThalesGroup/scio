"""
Inferring with Confidence
=========================
"""

# %%
# This tutorial shows how to quickly setup confidence scores for
# inference in a classification setup, using algorithms already
# :ref:`implemented <confidence-score-algorithms-classif>` in
# ``scio.scores``.
#
# Let's start with preparing a trained model and some (fake) calibration
# data. Both should be naturally defined by your own use-case. For this
# tutorial, we use a lightweight `Tiniest
# <https://github.com/xvel/cifar10-tiniest>`_ architecture trained on
# CIFAR10 and hosted on `our hub
# <https://github.com/ThalesGroup/scio/tree/hub>`_, and use random
# calibration data.

# These should be defined by your use-case
import torch

sample_shape = (3, 32, 32)
calib_data = torch.rand(10, *sample_shape)
calib_labels = torch.randint(0, 10, calib_data.shape)
net = torch.hub.load("ThalesGroup/scio:hub", "tiniest", trust_repo=True, verbose=False)
net = net.to(calib_data)

# %%
# 1. Choose an algorithm & define internal representations
# --------------------------------------------------------
# Let us choose a **confidence score algorithm** :ref:`implemented
# <confidence-score-algorithms-classif>` in ``scio.scores``, say
# :class:`~scio.scores.KNN`. The only required parameter for this method
# is ``k``, defining which neighbors to look for in latent spaces. As a
# rule of thumb, the scientific community uses
# :math:`(n_{\text{calib}})^{0.4}`.
#
# The authors of the :class:`~scio.scores.KNN` paper recommend using the
# *feature maps* (penultimate layer) as internal representations, so we
# will set this up when creating our :class:`~scio.recorder.Recorder`
# net (responsible for capturing embeddings, see
# :doc:`diving_inside_neural_networks`).

from scio.recorder import Recorder
from scio.scores import KNN

score = KNN(k=int(len(calib_data) ** 0.4))
rnet = Recorder(net, input_data=calib_data[[0]])
rnet  # Visualize layers

# %%

rnet.record((1, 11))  # Record the feature maps

# %%
# 2. Calibrate the score function
# -------------------------------
# To obtain a usable **confidence score function**, we simply need to
# calibrate the ``score`` instance using the
# :meth:`~scio.scores.BaseScore.fit` method:

score.fit(rnet, calib_data, calib_labels)

# %%
# For example in our :class:`~scio.scores.KNN` case, this step
# internally creates a search index populated with the embeddings of
# calibration samples, for efficient query of :math:`k^{\text{th}}`
# nearest neighbor during inference.

# %%
# 3. Infer with confidence!
# -------------------------
# We can now use the ``score`` instance to infer with confidence scores!

test_data = torch.rand(5, *sample_shape)
out, conformity = score(test_data)

# %%
# The first tensor ``out`` is (almost) exactly what ``net(test_data)``
# would have yielded without confidence scores. It corresponds to the
# *logits* for every sample and every class.
#
# The ``conformity`` tensor has the same shape as ``out``: it
# represents, for **every sample**, the conformity assigned **to every
# class** (some methods provide class-specific results, unlike
# :class:`~scio.scores.KNN`).

preds = out.argmax(1)
confs = conformity[:, 0]  # KNN is constant across classes
for i, (pred, conf) in enumerate(zip(preds, confs, strict=True)):
    print(f"Sample {i}: predicted {pred} with confidence {conf}")

# %%
# Note, as mentioned in :doc:`/user_guide/confidence_scores`, that
# scores are not necessarily between :math:`0` and :math:`1`, and that
# **only order matters**.
#
# If you wish to use these scores to perform OoD Detection and compare
# different algorithms, read
# :doc:`visualizing_and_evaluating_ood_detection_algorithms`.
#
# .. _inferring-with-confidence-profiling:
#
# [Bonus] Profiling
# -----------------
# The :attr:`~scio.scores.BaseScore.timer` attribute contains
# information about execution times. Querying its
# :attr:`~scio.utils.ScoreTimer.report` attribute shows a complete
# report over the object's lifetime.

score.timer.report  # Report execution times
