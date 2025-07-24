Index utils
===========

In our context, the word "index" often refers to a data structure designed specifically for fast similarity lookup, given a reference population and query samples. This is especially useful for *Nearest Neighbors Search* (NNS). In ``scio``, we perform such operations using the popular library `faiss <https://github.com/facebookresearch/faiss>`_. The following utilities ease its use when implementing :ref:`scores` relying on NNS.

.. autosummary::
   :toctree: autosummary

   ~scio.scores.utils.Index
   ~scio.scores.utils.make_indexes
