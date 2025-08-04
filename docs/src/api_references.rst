.. _api-references:

API References
==============

Most API references for this package are grouped in the following three sections:

- :doc:`api_references/recorder`. Describes the ``scio.recorder`` module, responsible for handling the internal states of Neural Networks.
- :doc:`api_references/scores`. Details the content of the ``scio.scores`` package, the core of OoD Detection algorithms.
- :doc:`api_references/eval`. Introduces the benchmarking utilities implemented in ``scio.eval``.

You may also refer to the :doc:`api_references/misc` section, detailing most of the remaining internal utilities.

Note that all submodules are lazily imported and accessible as top-level attributes. For example::

   import scio
   scio.eval.summary(...)  # Directly available

.. rubric:: Table of content

.. toctree::
   :includehidden:
   :maxdepth: 2
   :titlesonly:
   :caption: API References

   api_references/recorder
   api_references/scores
   api_references/eval
   api_references/misc
