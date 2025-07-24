``scio`` enums
==============

In many place in this package, we use `Enum <https://docs.python.org/3/library/enum.html#enum.Enum>`_ classes in conjunction with ``Literal`` as follows::

   @unique
   class Arg(str, EnumWithExplicitSupport):
      '''Enum for supported ``arg`` values.'''

      __slots__ = ()

      VALUE1 = "value1"
      VALUE2 = "value2"


   type ArgLike = Arg | Literal["value1", "value2"]

This allows static type checkers to understand direct API calls such as ``func("value1")``, while also providing runtime sanitization for more complex uses such as ``func(Arg(kw[0]))``.

Furthermore, we derive such *enums* from the base :class:`~scio.utils.enums.EnumWithExplicitSupport`, which provides a more detailed error message for unexpected values::

   >>> Arg("value3")  # Also fails type check
   <traceback>
   ValueError: 'value3' is not a valid Arg. Supported: 'value1', 'value2'

.. autosummary::
   :toctree: autosummary
   :nosignatures:
   :template: enums.rst

   ~scio.utils.enums.EnumWithExplicitSupport

.. rubric:: Used enums

.. autosummary::
   :toctree: autosummary
   :nosignatures:
   :template: enums.rst

   ~scio.utils.AggrName
   ~scio.utils.ClassAggr
   ~scio.utils.IndexMetric
   ~scio.utils.InterpolationKind
   ~scio.utils.LabelKind
   ~scio.utils.MultinomialTestMode
   ~scio.utils.ScoreClassifMode
   ~scio.utils.ScoreTimerOperation
   ~scio.utils.ZAggr
