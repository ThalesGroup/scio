"""JTLA for classification."""

__all__ = ["JTLA", "JTLATestMultinomial"]

from functools import partial
from itertools import repeat
from types import MappingProxyType
from typing import Literal, NotRequired, SupportsFloat, SupportsInt, TypedDict, cast

import torch
from torch import Tensor

from scio.scores.utils import (
    ak_lpe,
    get_aggregator,
    knn_label_count,
    make_indexes,
    multinomial_test,
    normalize_samples,
)
from scio.utils import (
    AggrName,
    ClassAggr,
    ClassAggrLike,
    IndexMetricLike,
    MultinomialTestModeLike,
    check,
)

from .base import BaseScoreClassif


# https://github.com/python/typing/issues/1952
class JTLATestMultinomial(TypedDict):
    """JTLA's multinomial test syntax.

    Precisely describes possible ``test`` values for multinomial test in
    :class:`~scio.scores.JTLA`.

    Arguments
    ---------
    type: ``Literal["multinomial"]``
        The type of test. Must be ``"multinomial"``.
    mode: ``MultinomialTestModeLike``
        See :class:`~scio.utils.MultinomialTestMode`.
    k: ``int``
        Number of neighbors to consider.
    prior: ``Tensor | float``, optional
        If not provided, treated as ``1.0``. Must be broadcastable to
        ``(n_classes, n_classes)``, where each row defines the Dirichlet
        prior for the corresponding class-conditional test. Ignored in
        :attr:`~scio.utils.MultinomialTestMode.MLE` mode.
    special_prior: ``float``, optional
        If provided, refers to the epsilon from the "special prior" in
        the `authors' implementation
        <https://github.com/jayaram-r/adversarial-detection/blob/94fd0881a3eef179e66301629c9a5e348ce46bd1/expts/helpers/multinomial.py#L42-L68>`_.
        In this case, ``prior`` is ignored.

    """

    type: Literal["multinomial"]
    mode: MultinomialTestModeLike
    k: int
    prior: NotRequired[Tensor | float]
    special_prior: NotRequired[float]


class JTLA(BaseScoreClassif):
    r"""JTLA for classification.

    The proposed framework essentially works in 3 steps:

    1. Statistical tests, conditional on both layer and class (true and
       predicted);
    2. Layer aggregation;
    3. Class aggregation.

    Arguments
    ---------
    test: ``JTLATestMultinomial``
        Defines the statistical test to use. Currently, only multinomial
        tests are supported. For possible values, refer to
        :class:`~scio.scores.classification.jtla.JTLATestMultinomial`.
    layer_aggregation: ``str``
        Defines layers aggregation scheme. The following values are
        supported.

        - ``"lpe"``: multi-layer normalization scheme using aK-LPE. Uses
          the same distance as nearest neighbors search. This approach
          requires a hyperparameter ``k`` for nearest neighbors search.
          By default, it uses ``test["k"]``, which can be overriden by
          specifying a suffix ``:k``. Examples: ``"lpe"``, ``"lpe:20"``.
        - Any value from :class:`~scio.utils.AggrName`: :math:`1`\ D
          aggregation of :math:`p`-values. These may be appended ``":"``
          followed by a coma-separated list of integers ``n``,
          indicating that combined :math:`p`-values for ``n``-tuples of
          layers must be computed. Appending ``":1"`` is a no-op.
          Examples: ``"sum"``, ``"harmonic:2"``, ``"fisher:1,2,3"``.

    layer_aggregation_consecutive: ``bool``
        Whether ``n``-tuples for layer aggregation are restricted to
        consecutive layers. Ignored for ``"lpe"`` aggregation. Defaults
        to ``False``.
    class_aggregation: ``ClassAggrLike``
        See :class:`~scio.utils.ClassAggr`. Defaults to ``"nat"``.
    pred_conditional: ``bool``
        Whether to use predicted-class-conditional scores as described
        in :cite:`JTLA`. Note that if the classifier is perfect on
        calibration samples, this has no effect. If ``False``, every
        predicted-class-conditional object from :cite:`JTLA` is replaced
        with its true-class-conditional counterpart. Defaults to
        ``True``.
    index_metric: ``IndexMetricLike``
        Kind of metric to use for nearest neighbors search. See
        :class:`~scio.utils.IndexMetric`. Defaults to ``"ip"``.

    Notes
    -----
    Here are a few key differences between our implementation and `the
    authors' <https://github.com/jayaram-r/adversarial-detection>`_:

    - No bootstraping for :math:`p`-values estimation;
    - No dimensionality reduction of latent spaces before neighbors
      search. Since they use neighborhood preserving projection, it
      should not impact neighbor search results and this is only seen as
      a computational acceleration;
    - New option ``mode="dcm"`` for
      :class:`~scio.utils.MultinomialTestMode`;
    - The layer aggregation can be done on ``n``-tuples with arbitrary
      sets of ``n``, and with or without restricting to consecutive
      layers.
    - It is possible to skip resorting to pred-class-conditional tests.

    References
    ----------
    .. bibliography::
       :filter: false

       JTLA

    """

    test: JTLATestMultinomial
    layer_aggregation: str = "fisher"  # [JTLA] best
    layer_aggregation_consecutive: bool = False  # [JTLA]
    class_aggregation: ClassAggrLike = "nat"
    pred_conditional: bool = True  # [JTLA]
    act_norm: float | None = 2  # [JTLA] default
    index_metric: IndexMetricLike = "ip"  # [JTLA] default

    def _check_params(self, n_calib: int) -> None:
        super()._check_params(n_calib)

        # Sanitize ``test`` dict and make it mappingproxy
        check(0 < self.test["k"] <= n_calib)
        self.test = MappingProxyType(self.test)  # type: ignore[assignment]  # github.com/python/typing/issues/1952

    def _on_param_will_be_set(self, attr: str, future_val: object) -> None:
        """Bypass unfit for specific attributes."""
        if attr not in {"class_aggregation", "layer_aggregation_consecutive"}:
            super()._on_param_will_be_set(attr, future_val)

    @torch.no_grad()
    def calibrate(self, calib_data: Tensor, calib_labels: Tensor) -> None:
        """Calibrate the scoring algorithm with In-Distribution data."""
        out = self.rnet(calib_data)  # Records activations
        all_activations = self.activations()
        self.calib_labels_true = calib_labels
        self.calib_labels_pred = out.argmax(1) if self.pred_conditional else None
        self.prepare_tests(all_activations, out.size(1))
        # Next are tensors of shape (n_classes, n_calib_samples, n_layers) (or None)
        self.calib_tests_true, self.calib_tests_pred = self.run_tests(all_activations)

        self.parse_layer_aggregation()
        if self.layer_aggregation_method == "lpe":
            self.prepare_lpe()

    @torch.no_grad()
    def get_conformity(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute output and associated conformity at inference."""
        out = self.rnet(inputs)
        tests = self.run_tests(self.activations())
        q_values = self.aggregate_layers(tests)
        conformity = self.aggregate_classes(q_values)
        return out, conformity

    def parse_layer_aggregation(self) -> None:
        """Store layer aggregation method and ``n``-tuples to use.

        Handles special case of ``"lpe"`` aggregation.
        """
        method, _, specs = self.layer_aggregation.partition(":")
        self.layer_aggregation_method: Literal["lpe"] | AggrName

        if method == "lpe":
            self.layer_aggregation_method = "lpe"
            self.lpe_k = int(specs if specs else cast("SupportsInt", self.test["k"]))
            return

        self.layer_aggregation_method = AggrName(method)
        tuples = set(map(int, specs.strip(",").split(","))) if specs else [1]
        self.layer_aggregation_tuples = tuples

    def prepare_tests(
        self,
        all_activations: tuple[Tensor, ...],
        n_classes: int,
    ) -> None:
        """Multinomial only for now. Creates indexes and stats tests.

        Note
        ----
        For multinomial test: Sets attributes ``indexes``,
        ``all_llrs_true`` and ``all_llrs_pred``. The last two are nested
        tuples of shape ``(n_layers, n_classes)``, unless
        ``not self.pred_conditional``, in which case
        ``self.all_llrs_true`` is set to ``itertools.repeat(None)``.

        """
        # Retrieve test parameters
        k = int(cast("SupportsInt", self.test["k"]))
        mode = self.test["mode"]
        nc2 = (n_classes, n_classes)
        if "special_prior" in self.test:
            eps = float(cast("SupportsFloat", self.test["special_prior"]))
            # See https://github.com/jayaram-r/adversarial-detection/blob/94fd0881a3eef179e66301629c9a5e348ce46bd1/expts/helpers/multinomial.py#L42-L68
            offdiag = 1 + eps / n_classes
            diag = 2 - (n_classes - 1) * eps / n_classes
            prior = torch.full(nc2, offdiag).to(all_activations[0]).fill_diagonal_(diag)
        else:
            prior_arg = self.test.get("prior", 1)
            prior_tensor = torch.as_tensor(prior_arg).to(all_activations[0])
            prior = torch.broadcast_to(prior_tensor, nc2)

        labels_true = self.calib_labels_true
        labels_pred = self.calib_labels_pred
        test = partial(multinomial_test, mode=mode)
        self.indexes = make_indexes(all_activations, metric=self.index_metric)

        # True-class-conditional tests
        all_llrs_true_list = []
        all_llrs_pred_list = []
        # Loop over layers
        for index, activations in zip(self.indexes, all_activations, strict=False):
            counts = knn_label_count(
                index,
                labels_true,
                n_classes,
                k,
                activations,
                self_query=True,
            )
            all_llrs_true_list.append(
                tuple(
                    test(counts[labels_true == c], prior_c)
                    for c, prior_c in enumerate(prior)
                ),
            )

            if self.pred_conditional:
                all_llrs_pred_list.append(
                    tuple(
                        test(counts[labels_pred == c], prior_c)
                        for c, prior_c in enumerate(prior)
                    ),
                )

        self.all_llrs_true = tuple(all_llrs_true_list)
        self.all_llrs_pred = (
            tuple(all_llrs_pred_list) if self.pred_conditional else repeat(None)
        )

    def run_tests(
        self,
        all_activations: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor | None]:
        r"""Run class-conditional statistical tests.

        Arguments
        ---------
        all_activations: ``tuple[Tensor]``
            Layers' batched activations, for which to run statistical
            tests.

        Returns
        -------
        tests_true: ``Tensor``
            Shape ``(n_classes, n_samples, n_layers)``.
            True-class-conditional tests results.
        tests_pred: ``Tensor | None``
            Not ``None`` only if :attr:`self.pred_conditional`. In this
            case, same shape as ``tests_true``.
            Predicted-class-conditional tests results.

        Note
        ----
        High test result :math:`\longleftrightarrow` high deviation.

        """
        # Retrieve test parameters
        k = int(cast("SupportsInt", self.test["k"]))

        labels_true = self.calib_labels_true
        n_classes = len(self.all_llrs_true[0])

        tests_true_list, tests_pred_list = [], []
        # Loop over layers
        for index, activations, llrs_true, llrs_pred in zip(
            self.indexes,
            all_activations,
            self.all_llrs_true,
            self.all_llrs_pred,
            strict=False,
        ):
            counts = knn_label_count(index, labels_true, n_classes, k, activations)
            tests_true_list.append(
                torch.stack([
                    llr(counts).nan_to_num(nan=torch.inf, posinf=torch.inf)
                    for llr in llrs_true
                ]),
            )
            if llrs_pred is None:
                continue

            tests_pred_list.append(
                torch.stack([
                    llr(counts).nan_to_num(nan=torch.inf, posinf=torch.inf)
                    for llr in llrs_pred
                ]),
            )

        tests_true = torch.stack(tests_true_list, 2)  # (n_classes, n_samples, n_layers)
        tests_pred = torch.stack(tests_pred_list, 2) if self.pred_conditional else None

        return tests_true, tests_pred

    def prepare_lpe(self) -> None:
        """For lpe layer aggregation prepare indexes and query calib."""
        self.lpe_normalizer = partial(
            normalize_samples,
            ord=self.act_norm,
            same_inf=True,
            sample_start_dim=2,
        )

        normed_true = self.lpe_normalizer(self.calib_tests_true)
        self.lpe_true_indxs = make_indexes(tuple(normed_true), metric=self.index_metric)
        self.lpe_true_calib = torch.stack([
            torch.sort(ak_lpe(index, self.lpe_k, tests, self_query=True))[0].nan_to_num(
                nan=torch.inf,
            )
            for index, tests in zip(self.lpe_true_indxs, normed_true, strict=False)
        ])

        if not self.pred_conditional:
            return

        if self.calib_tests_pred is None:  # pragma: no cover
            msg = "With `pred_conditional=True`, `calib_test_pred` should be a tensor"
            raise RuntimeError(msg)

        normed_pred = self.lpe_normalizer(self.calib_tests_pred)
        self.lpe_pred_indxs = make_indexes(tuple(normed_pred), metric=self.index_metric)
        self.lpe_pred_calib = torch.stack([
            torch.sort(ak_lpe(index, self.lpe_k, tests, self_query=True))[0].nan_to_num(
                nan=torch.inf,
            )
            for index, tests in zip(self.lpe_pred_indxs, normed_pred, strict=False)
        ])

    def aggregate_layers(
        self,
        tests: tuple[Tensor, Tensor | None],
    ) -> tuple[Tensor, Tensor | None]:
        r"""Aggregate tests results accross layers.

        Arguments
        ---------
        tests: ``tuple[Tensor, Tensor | None]``
            Output of :meth:`run_tests` on query samples. Shapes
            ``(n_classes, n_samples, n_layers)``.

        Returns
        -------
        q_values_true: ``Tensor``
            Shape ``(n_samples, n_classes)``. True-class-conditional
            aggregated :math:`q`-values.
        q_values_pred: ``Tensor | None``
            Not ``None`` only if :attr:`self.pred_conditional`. In this
            case, same shape as ``q_values_true``.
            Pred-class-conditional aggregated :math:`q`-values.

        Note
        ----
        High :math:`q`-value :math:`\longleftrightarrow` high
        conformity.

        """
        tests_true, tests_pred_or_none = tests

        if self.pred_conditional:
            if tests_pred_or_none is None:  # pragma: no cover
                msg = (
                    "With `pred_conditional=True`, `tests_pred_or_none` should be a "
                    "tensor"
                )
                raise RuntimeError(msg)

            tests_pred = tests_pred_or_none

        # Special aK-LPE method
        if self.layer_aggregation_method == "lpe":
            n_calib = self.lpe_true_calib.size(1)
            similarity_search = self.index_metric == "ip"

            normed_true = self.lpe_normalizer(tests_true)
            lpe_true = torch.stack([
                ak_lpe(index, self.lpe_k, tests).nan_to_num(nan=torch.inf)
                for index, tests in zip(self.lpe_true_indxs, normed_true, strict=False)
            ])
            p_values_true = torch.searchsorted(self.lpe_true_calib, lpe_true) / n_calib
            q_values_true = p_values_true if similarity_search else 1 - p_values_true
            if not self.pred_conditional:
                return q_values_true.T, None

            normed_pred = self.lpe_normalizer(tests_pred)
            lpe_pred = torch.stack([
                ak_lpe(index, self.lpe_k, tests).nan_to_num(nan=torch.inf)
                for index, tests in zip(self.lpe_pred_indxs, normed_pred, strict=False)
            ])
            p_values_pred = torch.searchsorted(self.lpe_pred_calib, lpe_pred) / n_calib
            q_values_pred = p_values_pred if similarity_search else 1 - p_values_pred
            return q_values_true.T, q_values_pred.T

        # Non-LPE layer aggregation method
        aggregator = get_aggregator(self.layer_aggregation_method)

        # True-class-conditional q-values
        calib_tests_true = self.calib_tests_true
        p_values_true_map = map(self.compute_p_values, calib_tests_true, tests_true)
        q_values_true = aggregator(torch.stack(list(p_values_true_map), dim=1), dim=2)
        if not self.pred_conditional:
            return q_values_true, None

        calib_tests_pred = cast("Tensor", self.calib_tests_pred)
        check(isinstance(calib_tests_pred, Tensor))
        # Pred-class-conditional q-values
        p_values_pred_map = map(self.compute_p_values, calib_tests_pred, tests_pred)
        q_values_pred = aggregator(torch.stack(list(p_values_pred_map), dim=1), dim=2)
        return q_values_true, q_values_pred

    def compute_p_values(self, reference_tests: Tensor, query_tests: Tensor) -> Tensor:
        r"""Compute :math:`p`-values for query samples.

        Following :cite:`JTLA{eq. (8)}`, for test values :math:`(t_1,
        ..., t_n)` from :math:`n` layers, we define :math:`p :=
        \mathbb{P}(T_1\geqslant t_1, ..., T_n\geqslant t_n)` where
        :math:`n`\ -tuples :math:`(T_1, ..., T_n)` are sampled across
        calibrations samples. This method computes and stacks all such
        :math:`p`-values across all the ``n``-tuples of layers, for
        ``n`` in :attr:`self.layer_aggregation_tuples`.

        This implemention is a trivial counting approach. See
        `orthogonal range searching
        <https://en.wikipedia.org/wiki/Range_searching#Orthogonal_range_searching>`_
        for more efficient methods.

        Arguments
        ---------
        reference_tests: ``Tensor``
            Reference tests results, against which to compute
            :math:`p`-values. Shape ``(n_reference_samples, n_layers)``.
        query_tests: ``Tensor``
            Query tests results, for which to compute :math:`p`-values.
            Shape ``(n_query_samples, n_layers)``.

        Returns
        -------
        p_values: ``Tensor``
            Shape ``(n_query_samples, n_p_values)``. The number of
            :math:`p`-values can be expressed as ``sum(num_n_tuples for
            n in required_tuples)``.

        """
        device = query_tests.device
        only_consecutive = self.layer_aggregation_consecutive

        def slicer(n: int, mask: Tensor) -> Tensor:
            """Create ``n``-tuples of mask across layers.

            Arguments
            ---------
            n: ``int``
            mask: ``Tensor``
                Boolean mask of shape ``(m0, m1)``.

            Returns
            -------
            out: ``Tensor``
                Boolean mask of shape ``(m0, num_n_tuples, n)`` where
                ``n_layers`` depends on ``m1`` and nonlcal
                ``only_consecutive``.

            Note
            ----
            Uses two nonlocal variables ``device: torch.device`` and
            ``only_consecutive: bool``

            """
            if only_consecutive:
                return mask.unfold(1, n, 1)

            range_n_layers = torch.arange(mask.size(1)).to(device=device)
            combs = torch.combinations(range_n_layers, n)
            return mask[:, combs]

        tuples = self.layer_aggregation_tuples
        samples_counts = []
        for sample_tests in query_tests:
            mask = reference_tests >= sample_tests
            counts = torch.cat([slicer(n, mask).all(2).sum(0) for n in tuples])
            samples_counts.append(counts)

        return torch.stack(samples_counts) / len(reference_tests)

    def aggregate_classes(self, q_values: tuple[Tensor, Tensor | None]) -> Tensor:
        r"""Aggregate :math:`q`-values accross classes.

        Differences with :cite:`JTLA{section 4.4}`: no log scale
        (unimportant) and inverse ratio (convention).

        Arguments
        ---------
        q_values: ``tuple[Tensor, Tensor | None]``
            Output of :meth:`aggregate_layers`. It's a :math:`2`\ -tuple
            with shapes (when ``Tensor``) ``(n_samples, n_classes)``.

        Returns
        -------
        conformity: ``Tensor``
            Shape ``(n_samples, n_classes)``. Result of :math:`q`-values
            aggregation.

        Raises
        ------
        :exc:`ValueError`
            If :attr:`self.class_aggregation` value is unsupported.

        """
        q_values_true = q_values[0]
        if self.pred_conditional:
            q_values_pred = cast("Tensor", q_values[1])
            check(isinstance(q_values_pred, Tensor))
        else:
            q_values_pred = q_values_true

        class_aggregation = ClassAggr(self.class_aggregation)
        if class_aggregation == ClassAggr.NAT:
            conformity = q_values_pred
        elif class_aggregation == ClassAggr.ADV:
            range_ = range(len(q_values_true))
            values, indices = q_values_true.topk(2)
            top1_val, top2_val = values.T
            top1_idx, _ = indices.T
            max_other_true = top1_val[:, None].broadcast_to(q_values_true.shape).clone()
            max_other_true[range_, top1_idx] = top2_val
            conformity = (q_values_pred / max_other_true).nan_to_num()
        else:
            msg = f"Unsupported class aggregation: {self.class_aggregation!r}"
            raise ValueError(msg)

        return conformity
