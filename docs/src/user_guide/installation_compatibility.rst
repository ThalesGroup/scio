.. _installation-compatibility:

Installation & Compatibility
============================

Installation
------------

We recommend installing ``scio`` from `PyPI <https://pypi.org/project/scio-pypi>`_::

	pip install scio-pypi  # the `-pypi` suffix will be removed soon

If you wish to install from source or wheels manually, you can download `release assets <https://github.com/ThalesGroup/scio/releases>`_ directly on the GitHub repository.

OS Compatibility
----------------
The library is available and functional on Ubuntu, Windows and MacOS. However, note the following regarding ``faiss``-related features.

- They are **partially** supported on **Windows** and their use will trigger a warning: due to non-Python dependency conflicts, there might be uncontrolled and silent concurrency issues, potentially altering results. For more information, see this nice ressource about `native dependencies <https://pypackaging-native.github.io/key-issues/native-dependencies>`_. Note that we still run full CI tests on Windows.
- They are **not officially supported** on **MacOS** and might generate random crashes. We should match the current Windows support status *relatively soon*.

Framework Compatiblity
----------------------
Although confidence scores are not a particularity of Neural Networks, ``scio`` is precisely developed to handle such models. More precisely, only *PyTorch* models (inheriting from ``torch.nn.Module``) are currently supported.

GPU Compatibility
-----------------
Our library is **fully compatible** and tested with CUDA devices for native use of GPU acceleration! Note however that features using ``faiss`` currently use CPU-bound indexes, introducing a potential GPU > CPU > GPU bottleneck. This may be improved in future versions. Do not hesitate to express interest by opening a related issue.

Supported Data Types
--------------------
The package is **fully compatible** and tested with ``torch.half`` (float16), ``torch.float`` (float32) and ``torch.double`` (float64) data types. However, we **discourage** the use of gradient-based algorithms with ``torch.half`` data, as they can easily generate irrelevant results due to potential ``nan`` values. Finally, note that ``faiss``-related operations temporarily convert to 32 bits data independently of the input type.
