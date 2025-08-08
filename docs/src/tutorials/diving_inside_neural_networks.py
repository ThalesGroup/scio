"""
Diving inside Neural Networks
=============================
"""

# %%
# This tutorial provides a short practical overview of the
# :class:`~scio.recorder.Recorder` class, which is designed to ease
# interactions with the internal states of *PyTorch* Neural Network
# objects (``torch.nn.Module``) during or after inference.

from scio.recorder import Recorder

# %%
# Wrapping and visualizing your Neural Network
# --------------------------------------------
#
# Let us first load an arbitrary Neural Network. We use a lightweight
# `Tiniest <https://github.com/xvel/cifar10-tiniest>`_ architecture
# trained on CIFAR10 and hosted on
# `our hub <https://github.com/ThalesGroup/scio/tree/hub>`_. We also
# prepare future input data for this tutorial.

import torch

inputs = torch.rand(5, 3, 32, 32)  # Random inputs with 5 samples
net = torch.hub.load("ThalesGroup/scio:hub", "tiniest", trust_repo=True, verbose=False)
net = net.to(inputs)

# %%
# To wrap it into a :class:`~scio.recorder.Recorder` Net, ``rnet``,
# one simply needs to specify an ``input_size`` (including batch
# dimension) or provide ``input_data``. This will directly analyze and
# store the control flow of the model, using the `torchinfo
# <https://github.com/TylerYep/torchinfo>`_ library.

rnet = Recorder(net, input_data=inputs[[0]])
rnet  # Visualize layers

# %%
# .. tip::
#    For summary customization, refer to ``torchinfo.summary`` `options
#    <https://github.com/TylerYep/torchinfo/blob/164597f999662bfbc64e7674156e4dda16ddba9c/torchinfo/torchinfo.py#L54>`_.
#    For example, it is possible to bound the depth of the
#    representation tree with ``depth=2``.
#
# .. note::
#    In the case of dynamic control flow, refer to the
#    ``force_static_flow`` argument of
#    :class:`~scio.recorder.Recorder`.
#
# In many ways, this wrapper is transparent to the user. For example,
# one can naturally process data with ``rnet(inputs)``.
#
# Selecting layers of interest
# ----------------------------
#
# The penultimate line of the above summary reports the layers that are
# currently set to be recorded (stored in :attr:`rnet.recording
# <scio.recorder.Recorder.recording>`). By default after instantiation,
# there are none.

print(repr(rnet).split("\n")[-2])  # Show penultimate summary line

# %%
# One can arbitrarily set this using :meth:`rnet.record()
# <scio.recorder.Recorder.record>` with the ``depth-idx`` identifiers
# from the summary (*e.g.* ``1-9``). For example, the following
# specifies that the output of the first ``Block`` and the penultimate
# layer should be recorded.

rnet.record((1, 3), (1, 11))
print(repr(rnet).split("\n")[-2])  # Show penultimate summary line

# %%
# .. note::
#    Though not shown in the summary, it is possible to select ``0-1``
#    to refer to the entire model.
#
# .. warning::
#    ``torchinfo.summary`` can only detect ``torch.nn.Module`` calls.
#    As such, if a Neural Network uses activation functions, it should
#    call their ``Module`` implementation (instead of their
#    ``functional`` counterpart) and declare them as an attribute,
#    for them to be visible as a layer in the summary. It is not
#    necessary to declare a different attribute for every activation
#    call (*e.g.* one ``self.relu = nn.ReLU()`` can be used multiple
#    times in :meth:`forward`).
#
# Capturing internal states
# -------------------------
#
# Once the recording layers are set, every forward pass will
# automatically store the corresponding internal states in the
# :attr:`rnet.activations <scio.recorder.Recorder.activations>` mapping.
# Its keys are the ``(depth, idx)`` 2-tuples.

out = rnet(inputs)  # Forward pass, records the activations
rnet.activations

# %%
# Other functionalities
# ---------------------
#
# In addition to recording, the following functionalities are available
# and documented in :class:`~scio.recorder.Recorder`:
#
# - One-the-fly postprocessing of activations during inference (*e.g.*
#   clipping).
# - Local disabling of the recording during forward pass.
# - Access the recorded layers' ``Module`` objects directly.
# - Get the named parameters of the recorded layers.
#
# Implementation details
# ----------------------
#
# This utility works by affecting hooks to every layer of ``net`` with
# ``Module.register_forward_hook``. However, since layers are not aware
# of the context in which they are called, these hooks carry references
# to ``rnet`` and with it, the sufficient context to know when to
# trigger. This means that two different
# :class:`~scio.recorder.Recorder` Nets can wrap the same ``net``
# without any conflict. As an implementation detail, note that these
# references are made `weak
# <https://docs.python.org/3/library/weakref.html>`_ in order to be
# properly cleaned up upon deletion of ``rnet``.
