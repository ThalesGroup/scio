Classification
==============

For development
---------------

Please refer to the base class :class:`~scio.scores.BaseScoreClassif` or the template :class:`~scio.scores.classification.TemplateClassif`.

.. autosummary::
   :toctree: autosummary
   :nosignatures:

   ~scio.scores.BaseScoreClassif
   ~scio.scores.classification.TemplateClassif

----

For inference
-------------

You can directly use one of the following algorithms, directly implemented in our library.

.. autosummary::
   :toctree: autosummary
   :nosignatures:

   ~scio.scores.BaselineClassif
   ~scio.scores.DeepMahalanobis
   ~scio.scores.DkNN
   ~scio.scores.Energy
   ~scio.scores.FeatureSqueezing
   ~scio.scores.GradNorm
   ~scio.scores.Gram
   ~scio.scores.IsoMax
   ~scio.scores.JointEnergy
   ~scio.scores.JTLA
   ~scio.scores.KNN
   ~scio.scores.LID
   ~scio.scores.Logit
   ~scio.scores.Odds
   ~scio.scores.ODIN
   ~scio.scores.ReAct
   ~scio.scores.RelativeMahalanobis
   ~scio.scores.Softmax
   ~scio.scores.Trust
