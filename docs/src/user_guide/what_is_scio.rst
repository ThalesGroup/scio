.. _what-is-scio:

.. role:: bolditalic
   :class: bolditalic

What is ``scio``?
=================

It is an open-source Python library for designing, using and evaluating **confidence scores in Neural Networks**. This topic has recently gained significant attention, as deploying of Deep Learning models raises pressing reliability concerns – specifically for critical applications.

The project was initiated by Élie Goudout in 2024, internally at cortAIx Labs, and its `first open-source release <first-release_>`_ dates back to July 25th, 2025, under the `MIT License <first-license_>`_.

The name "scio" carries two meanings.

1. In Latin, `sciō <scio-latin_>`_ can mean *to understand, to have practical knowledge* (sharing etymology with *science*). The well-known saying *"*:bolditalic:`scio` *me nihil scire"* ("I know that I know nothing") aptly illustrates the purpose of confidence scores: identifying predictions that should not be trusted due to insufficient informed support.
2. It is also an acronym for :bolditalic:`S`\ *tatistical* :bolditalic:`C`\ *onfidence from* :bolditalic:`I`\ *nternal* :bolditalic:`O`\ *bservations*, reflecting both our motivation and approach.

The following sections introduce the broader context and outline the conceptual framework that guides the development and usage of ``scio``.

.. _first-release: https://github.com/ThalesGroup/scio/releases/tag/v1.0.0a1
.. _first-license: https://github.com/ThalesGroup/scio/tree/2766ac958da28cc6acf6eed0a3c84fedec7c565c?tab=License-1-ov-file
.. _scio-latin: https://en.wiktionary.org/wiki/scio#Verb_2
