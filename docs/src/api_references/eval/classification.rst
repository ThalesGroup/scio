Visualization & Evaluation in Classification
============================================

.. _discriminative_power:

Discriminative Power
--------------------

The purpose of confidence scores is to discriminate between *In-Districution* (InD) and *Out-of-Distribution* (OoD) samples. To compare different scores in their efficiency relative to this task, we introduce the concept of **Discriminative Power**, which denotes a metric satisfying the following conditions.

1. Be invariant under any (even nonlinear) rescaling of the score. This implies that it only depends on all the possible :math:`(FP, TP)` tuples when thresholding the score with every possible threshold.
2. Only depend on the Pareto front of all the possible :math:`(FP, TP)` tuples.
3. Satisfy basic monoticity principles (*e.g.* if an InD sample's score is increased, all else being equal, then the discriminative power cannot decrease).

In light of the above caracterization, we provide a :class:`~scio.eval.ROC` utility class which allows simple and formalized definition of Disctiminating Power classes.

.. autosummary::
   :toctree: autosummary

   ~scio.eval.ROC

.. rubric:: Implemented Discriminative Power metrics

We also provide implementation for a few standard such metrics, all derived from the base :class:`~scio.eval.BaseDiscriminativePower` class.

.. autosummary::
   :toctree: autosummary
   :nosignatures:

   ~scio.eval.BaseDiscriminativePower

.. autosummary::
   :toctree: autosummary
   :nosignatures:

   ~scio.eval.AUC
   ~scio.eval.MCC
   ~scio.eval.TNR
   ~scio.eval.TPR


Visualization & Evaluation
--------------------------

.. rubric:: Running experiments

In order to evaluate scoring algorithms, it is necessary to gather various datasets, specifically one InD and possibly several OoD datasets, in order to fit :ref:`score <scores>` instances and collect their output. This may be done with the following functions.

.. autosummary::
   :toctree: autosummary

   ~scio.eval.compute_confidence
   ~scio.eval.fit_scores


.. rubric:: Visualization

We provide two visualization possibilities: histograms and ROC curves.

.. autosummary::
   :toctree: autosummary

   ~scio.eval.histogram_oods
   ~scio.eval.roc_scores
   ~scio.eval.summary_plot


.. rubric:: Evaluation

Given experimental results, the following help quantify the Discriminative Power.

.. autosummary::
   :toctree: autosummary

   ~scio.eval.compute_metrics
   ~scio.eval.summary_table

.. rubric:: Visualization & Evaluation

You can do both at the same time with the following.

.. autosummary::
   :toctree: autosummary

   ~scio.eval.summary
