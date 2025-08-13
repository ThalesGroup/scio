.. _installation-compatibility:

Installation & Compatibility
============================

Installation
------------

We recommend installing ``scio`` from `PyPI <https://pypi.org/project/scio-pypi>`_::

	pip install scio-pypi  # the `-pypi` suffix will be removed soon

If you wish to install from source or wheels manually, you can download `release assets <https://github.com/ThalesGroup/scio/releases>`_ directly on the GitHub repository, or clone its current state with:

.. code-block:: bash

	git clone https://github.com/ThalesGroup/scio.git

OS Compatibility
----------------
The library is available and functional on **Ubuntu**, **Windows** and *MacOS* (minus ``faiss``-related features ─ feel free to show your interest `here <https://github.com/ThalesGroup/scio/issues/2>`_).

Framework Compatiblity
----------------------
Although confidence scores are not a particularity of Neural Networks, ``scio`` is precisely developed to handle such models. More precisely, only *PyTorch* models (inheriting from ``torch.nn.Module``) are currently supported.

GPU Compatibility
-----------------
Our library is **fully compatible** and tested with CUDA devices for native use of GPU acceleration! Note however that features using ``faiss`` currently use CPU-bound indexes, introducing a potential GPU > CPU > GPU bottleneck. This may be improved in future versions ─ feel free to show your interest `here <https://github.com/ThalesGroup/scio/issues/18>`_.

Supported Data Types
--------------------
The package is **fully compatible** and tested with ``torch.half`` (float16), ``torch.float`` (float32) and ``torch.double`` (float64) data types. However, we **discourage** the use of gradient-based algorithms with ``torch.half`` data, as they can easily generate irrelevant results due to potential ``nan`` values. Finally, note that ``faiss``-related operations temporarily convert to 32 bits data, regardless of the input type.

There is currently no official support for ``torch.bfloat16`` even though many features may be compatible. Feel free to show your interest by opening a related issue.
