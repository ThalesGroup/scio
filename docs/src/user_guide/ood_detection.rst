Out-of-Distribution Detection
=============================

The concept
-----------

The task of "**Out-of-Distribution** Detection" (or "**OoD** Detection") is a particular case of `novelty detection <https://en.wikipedia.org/wiki/Novelty_detection>`_. It consists in identifying, at runtime, inputs for which a model's prediction should be rejected, by lack of informed support. For example, if a Neural Network was trained to classify images of cats and dogs, it would seem irrelevant to trust its output on a tree image. Although there is no absolute definitive answer to this question (depending on the context, what a person considers OoD might be considered **In-Distribution** (**InD**) by another), it is still possible to perform deviation estimations through confidence scores.

In ``scio``, we aim at performing OoD Detection naturally by **thresholding** confidence scores. That is, defining the following decision function for a given threshold :math:`\tau`:

.. math::

	x\text{ is OoD}\Longleftrightarrow \text{Score}(x)\leqslant\tau.

Ultimately, the choice of threshold is a trade-off between False Positives and True Positives, left to the user's responsibility. However, it may be relevant to compare score functions in their ability to offer good trade-offs. For example, imagine we have at our disposal :math:`X_{\text{InD}}` and :math:`X_{\text{OoD}}` corresponding to sets of samples which we know to be respectively InD and OoD. Then, we can evaluate score functions, say :math:`S_{\text{bad}}` and :math:`S_{\text{good}}`, on those sets and plot the following histograms.

.. image:: /_static/histograms.svg
   :alt: histograms illustration
   :align: center
   :width: 100%

In this case, it seems clear that for the chosen InD and OoD representatives, :math:`S_{\text{good}}` offers better trade-offs than :math:`S_{\text{bad}}`. In ``scio.eval``, we provide tools and metrics to :doc:`visualize and quantify <../api_references/eval>` such observations.

Application
-----------

The ability to robustly identify Out-of-Distribution samples may have many useful applications. Simple examples are:

#. **Monitoring.** Using the OoD Detection as a **reject option** may help using safe callbacks more efficiently for deployed models.
#. **Training.** Some training procedures may benefit from identifying samples for which a model is currently unfit to infer.
