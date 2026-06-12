# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.decision
=================
V5 Decision Functional — the generalization of the two-token logit diff
(ROADMAP_V5_FOUNDATIONS.md §2.1).

A decision functional D(x) scores a model's decision as

    D(x) = logsumexp(logits over positive set) − logsumexp(logits over negative set)

For singleton sets this reduces *exactly* to the legacy logit difference
(the softmax normalizer cancels in the difference), so every existing
Glassbox result is a special case of this module — backward compatible
by construction, not by approximation.

Three decision shapes are supported:

1. **Verbalizer sets** (single-token variants): computed directly from the
   last-position logits via :meth:`ResolvedDecision.value_from_logits`.
2. **Multi-token variants**: computed from injected sequence log-probability
   scores via :meth:`ResolvedDecision.value_from_scores` (the caller supplies
   a ``score_fn(variant_ids) -> log p(variant | prompt)``, typically teacher
   forcing; this keeps the module free of any model dependency).
3. Score/numeric decisions are deferred to a later phase (see roadmap).

This module is dependency-free (stdlib only). Anything model-specific lives
with the caller, which is what makes the functional architecture-agnostic.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "VerbalizerSet",
    "DecisionFunctional",
    "ResolvedDecision",
]


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp over a non-empty float sequence."""
    m = max(values)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


@dataclass(frozen=True)
class VerbalizerSet:
    """A labeled set of surface realizations of one decision outcome.

    Attributes:
        label: Human-readable outcome name (e.g. ``"approve"``). Used in
            reports and the evidence vault.
        variants: Tuple of strings the model might emit for this outcome
            (e.g. ``(" approved", " approve", " yes")``). Tokenization
            happens at :meth:`DecisionFunctional.resolve` time.
    """

    label: str
    variants: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError(
                f"VerbalizerSet {self.label!r} needs at least one variant"
            )


@dataclass(frozen=True)
class ResolvedDecision:
    """A DecisionFunctional bound to a tokenizer (token ids, not strings).

    Attributes:
        positive_ids: Token-id sequences for each positive variant.
        negative_ids: Token-id sequences for each negative variant.
        positive_texts: Source strings, index-aligned with ``positive_ids``.
        negative_texts: Source strings, index-aligned with ``negative_ids``.
        all_single_token: True when every variant encodes to one token —
            the cheap path where D comes from one position's logits.
    """

    positive_ids: List[List[int]]
    negative_ids: List[List[int]]
    positive_texts: Tuple[str, ...]
    negative_texts: Tuple[str, ...]
    all_single_token: bool

    def value_from_logits(self, last_logits: Sequence[float]) -> float:
        """Compute D from one position's logits (single-token sets only).

        Args:
            last_logits: Logit vector at the decision position (any
                indexable sequence — list, numpy array, torch tensor).

        Returns:
            ``logsumexp(logits[positive]) − logsumexp(logits[negative])``.
            For singleton sets this equals the legacy logit diff exactly.

        Raises:
            ValueError: If any variant is multi-token (use
                :meth:`value_from_scores` instead).
        """
        if not self.all_single_token:
            raise ValueError(
                "value_from_logits requires all-single-token variants; "
                "this decision has multi-token variants — use "
                "value_from_scores with a sequence score function."
            )
        pos = [float(last_logits[ids[0]]) for ids in self.positive_ids]
        neg = [float(last_logits[ids[0]]) for ids in self.negative_ids]
        return _logsumexp(pos) - _logsumexp(neg)

    def value_from_scores(
        self, score_fn: Callable[[Sequence[int]], float]
    ) -> float:
        """Compute D from injected per-variant sequence log-probabilities.

        Args:
            score_fn: Callable mapping a variant's token-id sequence to
                ``log p(variant | prompt)`` (e.g. teacher-forced sum of
                token log-probs). Model-specific; supplied by the caller.

        Returns:
            ``logsumexp(scores_positive) − logsumexp(scores_negative)``.

        Raises:
            ValueError: If ``score_fn`` fails for any variant; the variant
                text is named in the error so failures are diagnosable.
        """
        def _score_all(
            ids_list: List[List[int]], texts: Tuple[str, ...]
        ) -> List[float]:
            out: List[float] = []
            for ids, text in zip(ids_list, texts):
                try:
                    out.append(float(score_fn(ids)))
                except Exception as exc:
                    raise ValueError(
                        f"score_fn failed for variant {text!r}: {exc}"
                    ) from exc
            return out

        pos = _score_all(self.positive_ids, self.positive_texts)
        neg = _score_all(self.negative_ids, self.negative_texts)
        return _logsumexp(pos) - _logsumexp(neg)


@dataclass(frozen=True)
class DecisionFunctional:
    """A decision expressed as two disjoint verbalizer sets.

    Attributes:
        positive: The outcome whose evidence raises D (e.g. approve).
        negative: The outcome whose evidence lowers D (e.g. deny).
    """

    positive: VerbalizerSet
    negative: VerbalizerSet

    @classmethod
    def from_tokens(cls, correct: str, incorrect: str) -> "DecisionFunctional":
        """Backward-compatible constructor matching the legacy
        ``analyze(prompt, correct, incorrect)`` two-token call shape."""
        return cls(
            VerbalizerSet(correct.strip() or correct, (correct,)),
            VerbalizerSet(incorrect.strip() or incorrect, (incorrect,)),
        )

    def resolve(
        self, encode: Callable[[str], Sequence[int]]
    ) -> ResolvedDecision:
        """Bind the functional to a tokenizer.

        Args:
            encode: Callable mapping a variant string to its token ids
                (e.g. ``lambda s: model.to_tokens(s, prepend_bos=False)[0]``).

        Returns:
            A :class:`ResolvedDecision` ready to compute values.

        Raises:
            ValueError: If a variant cannot be encoded, or if the positive
                and negative sets overlap after tokenization (an overlapping
                token sequence would make D ill-defined).
        """
        def _encode_set(vs: VerbalizerSet) -> Tuple[List[List[int]], Tuple[str, ...]]:
            seen: Dict[Tuple[int, ...], None] = {}
            texts: List[str] = []
            for variant in vs.variants:
                try:
                    ids = tuple(int(i) for i in encode(variant))
                except Exception as exc:
                    raise ValueError(
                        f"variant {variant!r} in set {vs.label!r} could not "
                        f"be encoded: {exc}"
                    ) from exc
                if not ids:
                    raise ValueError(
                        f"variant {variant!r} in set {vs.label!r} encoded "
                        "to zero tokens"
                    )
                if ids not in seen:  # dedupe, order-preserving
                    seen[ids] = None
                    texts.append(variant)
            return [list(ids) for ids in seen], tuple(texts)

        pos_ids, pos_texts = _encode_set(self.positive)
        neg_ids, neg_texts = _encode_set(self.negative)

        overlap = {tuple(i) for i in pos_ids} & {tuple(i) for i in neg_ids}
        if overlap:
            raise ValueError(
                f"positive set {self.positive.label!r} and negative set "
                f"{self.negative.label!r} overlap after tokenization: "
                f"{sorted(overlap)}"
            )

        all_single = all(len(i) == 1 for i in pos_ids + neg_ids)
        logger.debug(
            "Resolved decision %s/%s: %d vs %d variants, single_token=%s",
            self.positive.label, self.negative.label,
            len(pos_ids), len(neg_ids), all_single,
        )
        return ResolvedDecision(
            positive_ids=pos_ids,
            negative_ids=neg_ids,
            positive_texts=pos_texts,
            negative_texts=neg_texts,
            all_single_token=all_single,
        )

    def to_dict(self) -> Dict[str, Dict[str, object]]:
        """JSON-safe description for the evidence vault (Annex IV §4)."""
        return {
            "positive": {
                "label": self.positive.label,
                "variants": list(self.positive.variants),
            },
            "negative": {
                "label": self.negative.label,
                "variants": list(self.negative.variants),
            },
        }
