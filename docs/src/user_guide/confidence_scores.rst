Confidence Scores
=================

The concept
-----------

In the context of Machine Learning, confidence scores are real numbers paired with runtime predictions, meant to inform the operator about the "level of trust" they should have regarding individual model outputs. We illustrate very common examples below for Image Classification (left) and Object Detection (right).

|panda| |traffic|

.. |panda| image:: /_static/panda.png
   :width: 21.86%
   :alt: panda

.. |traffic| image:: /_static/traffic.png
   :width: 76.64%
   :alt: trtraffic

Both examples above use scores between :math:`0` and :math:`1` *by design* (computed via the softmax function), but in general they can be **any real number** (or even :math:`\pm\infty`).

The general hope is that if an input is completely irrelevant to the context in which a model was trained, then the score should be very low (and vice-versa).

Only order matters
------------------

In the context of ``scio``, the **only assumption** about confidence scores is that **order matters**. For example, if one trusts a model regarding prediction :math:`A` with confidence :math:`\alpha`, they should be consistent and also trust every prediction :math:`B` with confidence :math:`\beta\geqslant\alpha`.

Besides order, we specifically **ignore scale**, in order avoid any bias regarding the interpretation of confidence scores. We think this approach sound as we might use many different methods for assigning confidence scores, most of which do not have natural scales that any human operator (even field experts) could confidently interpret. Note that there is a lot of scientific literature regarding the recalibration of the softmax output, which is out of scope for our project.

Vocabulary
----------

Throughout the documentation and source code, "confidence score" is loosely used to denote one of **three possible meanings**, which is hopefully clear from the context. These are:

#. **Actual confidence score.** For example :math:`2.78`.
#. **Confidence score function.** That is a function which assigns an actual confidence score to every given input. In Classification, a confidence score function has signature :math:`\text{Score}: \langle\text{input space}\rangle\rightarrow\overline{\mathbb{R}}`. It is much less clear in Object Detection for example, as one could assign confidence scores either to every predicted object (*e.g.* example above), to the entire image, or even at the internal cell level when it makes sense.
#. **Confidence score algorithm.** It is a procedure describing how to obtain a confidence score function, given a trained model and some calibration data (more on that in :doc:`paradigm`). In that sense, a confidence score *function* is an instantiation of a confidence score *algorithm*.
