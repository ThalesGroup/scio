"""Enum utils."""

__all__ = [
    "AggrName",
    "AggrNameLike",
    "ClassAggr",
    "ClassAggrLike",
    "EnumWithExplicitSupport",
    "IndexMetric",
    "IndexMetricLike",
    "InterpolationKind",
    "InterpolationKindLike",
    "LabelKind",
    "LabelKindLike",
    "MultinomialTestMode",
    "MultinomialTestModeLike",
    "ScoreClassifMode",
    "ScoreClassifModeLike",
    "ScoreTimerOperation",
    "ScoreTimerOperationLike",
    "ZAggr",
    "ZAggrLike",
]

from enum import Enum, unique
from inspect import Parameter, Signature
from typing import Literal


class EnumWithExplicitSupport(Enum):
    """Base Enum for explicit error message.

    In many place in this package, we use :class:`Enum` in conjunction
    with ``Literal`` as follows::

        @unique
        class Arg(str, EnumWithExplicitSupport):
            '''Enum for supported ``arg`` values.'''

            __slots__ = ()

            VALUE1 = "value1"
            VALUE2 = "value2"


        type ArgLike = Arg | Literal["value1", "value2"]

    This allows static type checkers to understand direct API calls such
    as ``func("value1")``, while also providing runtime sanitization for
    more complex uses such as ``func(Arg(kw[0]))``.

    This :class:`Enum` subclass provides more details about the
    supported values during runtime errors::

        >>> Arg("value3")  # Also fails type check
        <traceback>
        ValueError: 'value3' is not a valid Arg. Supported: 'value1', 'value2'

    """

    __slots__ = ()
    # While enums may technically accept multiple arguments
    # (github.com/python/cpython/issues/132543), we assume only one is
    # expected in our case and override the signature for doc style. If
    # necessary, override again in subclasses.
    __signature__ = Signature([Parameter("value", Parameter.POSITIONAL_OR_KEYWORD)])  # type: ignore[assignment]  # github.com/python/cpython/issues/132543

    @classmethod
    def _missing_(cls, value: object) -> None:
        values = {member.value for member in cls}
        supported_str = ", ".join(sorted(map(repr, values)))
        msg = f"{value!r} is not a valid {cls.__qualname__}. Supported: {supported_str}"
        raise ValueError(msg)


@unique
class InterpolationKind(str, EnumWithExplicitSupport):
    r"""Enum for supported :class:`~scio.eval.ROC` interpolation methods.

    Attributes
    ----------
    CONVEX_HULL:
        Interpolate to the overall ROC convex hull.
    PESSIMISTIC:
        Interpolate to the previous value.

    Note
    ----
    This `discussion
    <https://github.com/scikit-learn/scikit-learn/issues/29252>`_
    motivates the (unpopular) absence of a :attr:`LINEAR` mode.

    """

    __slots__ = ()

    CONVEX_HULL = "convex_hull"
    PESSIMISTIC = "pessimistic"


type InterpolationKindLike = InterpolationKind | Literal["convex_hull", "pessimistic"]


@unique
class ScoreClassifMode(str, EnumWithExplicitSupport):
    """Enum for supported ``mode`` in classification scores.

    The ``mode`` describes an optional per-sample postprocess applied to
    the ``conformity`` output of a classification score's
    :meth:`~scio.scores.BaseScoreClassif.get_conformity`.

    Attributes
    ----------
    DIFF:
        Compute the additional preference to the highest other
        conformity. Only maximum conformities remain nonnegative.
    RATIO:
        Multiplicative analogue of :attr:`DIFF`. Should only be used
        with nonnegative conformities.
    RAW:
        No postprocessing.

    """

    __slots__ = ()

    DIFF = "diff"
    RATIO = "ratio"
    RAW = "raw"


type ScoreClassifModeLike = ScoreClassifMode | Literal["diff", "ratio", "raw"]


@unique
class LabelKind(str, EnumWithExplicitSupport):
    """Enum for supported ``calib_labels`` values with :class:`~scio.scores.Gram`.

    Describes whether the calibration of :class:`~scio.scores.Gram`
    should use the *predicted* or the *true* labels.

    Attributes
    ----------
    PRED:
        Use predicted labels.
    TRUE:
        Use true labels.

    """

    __slots__ = ()

    PRED = "pred"
    TRUE = "true"


type LabelKindLike = LabelKind | Literal["pred", "true"]


@unique
class ClassAggr(str, EnumWithExplicitSupport):
    """Enum for supported ``class_aggregation`` in :class:`~scio.scores.JTLA`.

    See :cite:`JTLA{section 4.4}`.

    Attributes
    ----------
    ADV:
        Designed to detect adversarial samples.
    NAT:
        Designed to detect natural OoD samples.

    """

    __slots__ = ()

    ADV = "adv"
    NAT = "nat"


type ClassAggrLike = ClassAggr | Literal["adv", "nat"]


@unique
class ZAggr(str, EnumWithExplicitSupport):
    """Enum for supported :math:`z`-aggregation schemes in :class:`~scio.scores.Odds`.

    From :cite:`Odds{section 3.3 and Appendix 7.1}`, it is clear that
    the :math:`z`-normalized deviations for :math:`g_{yz}` are
    thresholded using OoD calibration data. This is not allowed in our
    unsupervised setup. As such, we choose to measure nonconformity with
    regard to class :math:`c` by aggregating these statistics against
    the candidate class. The idea is that if an example of true class
    :math:`c_1` was manipulated to appear as :math:`c_2`, then
    :math:`g_{c_1c_2}` would be very low and :math:`g_{c_2c_1}` very
    high, while the other :math:`g_{yz}` would be noisy statistics (?).
    From this, the following three simple aggregation schemes arise.

    Attributes
    ----------
    DETECT:
        By looking the for maximum :math:`g_{yz}` across :math:`z`, try
        to identify suspicious class favorism.
    RECTIFY:
        By looking for the minimum :math:`g_{yz}` across :math:`z`, look
        for unjustly neglected class.
    SUM:
        By summing along :math:`z`, hope to not only detect OoD, but
        also identify the correct class. Sadly, the "noisy" statistics
        :math:`g_{yz}` for :math:`z` different from :math:`c_1` and
        :math:`c_2` do impact the final result. This is why it's not an
        obvious better option.

    """

    __slots__ = ()

    DETECT = "detect"
    RECTIFY = "rectify"
    SUM = "sum"


type ZAggrLike = ZAggr | Literal["detect", "rectify", "sum"]


@unique
class IndexMetric(str, EnumWithExplicitSupport):
    r"""Enum for supported ``metric`` values in :class:`~scio.scores.utils.Index`.

    Attributes
    ----------
    IP:
        **I**\ nner **P**\ roduct metric. For :math:`L^2`\ -normalized
        samples, equivalent to cosine similarity.
    L2:
        :math:`L^2` metric.

    """

    __slots__ = ()

    IP = "ip"
    L2 = "l2"


type IndexMetricLike = IndexMetric | Literal["l2", "ip"]


@unique
class AggrName(str, EnumWithExplicitSupport):
    r"""Enum for supported named :math:`1`\ D aggregation methods.

    Values are valid inputs for :func:`~scio.scores.utils.get_aggregator`.

    Attributes
    ----------
    FISHER:
        Sum of logarithms. Requires nonnegative inputs.
    GEOMETRIC:
        Geometric mean. Requires nonnegative inputs.
    HARMONIC:
        Harmonic mean.
    MAX:
        Explicit.
    MEAN:
        Explicit. Requires floating inputs.
    MIN:
        Explicit.
    PROD:
        Product.
    SUM:
        Explicit.

    """

    __slots__ = ()

    FISHER = "fisher"
    GEOMETRIC = "geometric"
    HARMONIC = "harmonic"
    MAX = "max"
    MEAN = "mean"
    MIN = "min"
    PROD = "prod"
    SUM = "sum"


type AggrNameLike = (
    AggrName
    | Literal[
        "fisher",
        "geometric",
        "harmonic",
        "max",
        "mean",
        "min",
        "prod",
        "sum",
    ]
)


@unique
class MultinomialTestMode(str, EnumWithExplicitSupport):
    r"""Enum for supported ``mode`` in :func:`~scio.scores.utils.multinomial_test`.

    Attributes
    ----------
    DCM:
        **D**\ irichlet **C**\ ompound **M**\ ultinomial. See
        :func:`~scio.scores.utils.dirmult_surprise`.
    MAP:
        **M**\ aximum **A** **P**\ osteriori with Dirichlet prior.
    MLE:
        **M**\ aximum **L**\ ikelihood **E**\ stimation.

    """

    __slots__ = ()

    DCM = "dcm"
    MAP = "map"
    MLE = "mle"


type MultinomialTestModeLike = MultinomialTestMode | Literal["compound", "map", "mle"]


@unique
class ScoreTimerOperation(str, EnumWithExplicitSupport):
    r"""Enum for supported operations timed by :class:`~scio.utils.ScoreTimer`.

    Attributes
    ----------
    CALIBRATION:
        Explicit.
    INFERENCE:
        Explicit.

    """

    __slots__ = ()

    CALIBRATION = "calibration"
    INFERENCE = "inference"


type ScoreTimerOperationLike = ScoreTimerOperation | Literal["calibration", "inference"]
