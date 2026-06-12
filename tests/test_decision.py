"""
tests/test_decision.py — V5 decision functional (verbalizer sets).

Pure-logic tests: no torch, no transformer_lens. A fake encode function
stands in for the tokenizer; plain Python floats stand in for logits.

The mathematical contract under test (ROADMAP_V5 §2.1):

    D(x) = logsumexp(logits over A) − logsumexp(logits over B)

which for singleton sets reduces EXACTLY to the legacy logit diff
(the softmax normalizer cancels in the difference), guaranteeing
backward compatibility.
"""

import math

import pytest

from glassbox.decision import (
    DecisionFunctional,
    VerbalizerSet,
)

# ---------------------------------------------------------------------------
# Fake tokenizer: a fixed vocabulary, encode() splits on '+' for multi-token.
# ---------------------------------------------------------------------------

_VOCAB = {
    " Mary": 10, " John": 11,
    " approved": 20, " approve": 21, " yes": 22,
    " denied": 30, " deny": 31, " no": 32,
    "App": 40, "roved": 41,
}


def fake_encode(text):
    """'App+roved' → [40, 41]; known words → single id; else error."""
    if "+" in text:
        return [_VOCAB[part] for part in text.split("+")]
    if text not in _VOCAB:
        raise KeyError(f"unknown token {text!r}")
    return [_VOCAB[text]]


def logits_with(pairs, size=50, fill=-10.0):
    """Dense logit list with selected (token_id, value) entries."""
    out = [fill] * size
    for tid, val in pairs:
        out[tid] = val
    return out


# ---------------------------------------------------------------------------
# VerbalizerSet construction
# ---------------------------------------------------------------------------

class TestVerbalizerSet:
    def test_requires_at_least_one_variant(self):
        with pytest.raises(ValueError):
            VerbalizerSet("empty", ())

    def test_variants_preserved_in_order(self):
        vs = VerbalizerSet("approve", (" approved", " yes"))
        assert vs.variants == (" approved", " yes")

    def test_is_hashable_and_frozen(self):
        vs = VerbalizerSet("a", (" yes",))
        assert hash(vs)
        with pytest.raises(Exception):
            vs.label = "b"  # frozen


# ---------------------------------------------------------------------------
# Resolution against a tokenizer
# ---------------------------------------------------------------------------

class TestResolve:
    def test_singleton_resolves_to_single_ids(self):
        d = DecisionFunctional.from_tokens(" Mary", " John")
        r = d.resolve(fake_encode)
        assert r.positive_ids == [[10]]
        assert r.negative_ids == [[11]]
        assert r.all_single_token is True

    def test_multi_variant_set_resolution(self):
        d = DecisionFunctional(
            VerbalizerSet("approve", (" approved", " approve", " yes")),
            VerbalizerSet("deny", (" denied", " deny", " no")),
        )
        r = d.resolve(fake_encode)
        assert [v[0] for v in r.positive_ids] == [20, 21, 22]
        assert r.all_single_token is True

    def test_multi_token_variant_detected(self):
        d = DecisionFunctional(
            VerbalizerSet("approve", ("App+roved",)),
            VerbalizerSet("deny", (" denied",)),
        )
        r = d.resolve(fake_encode)
        assert r.positive_ids == [[40, 41]]
        assert r.all_single_token is False

    def test_overlapping_sets_rejected(self):
        d = DecisionFunctional(
            VerbalizerSet("a", (" yes",)),
            VerbalizerSet("b", (" yes",)),
        )
        with pytest.raises(ValueError, match="[Oo]verlap"):
            d.resolve(fake_encode)

    def test_duplicate_variant_within_set_deduplicated(self):
        d = DecisionFunctional(
            VerbalizerSet("a", (" yes", " yes")),
            VerbalizerSet("b", (" no",)),
        )
        r = d.resolve(fake_encode)
        assert r.positive_ids == [[22]]

    def test_unencodable_variant_raises_with_variant_named(self):
        d = DecisionFunctional(
            VerbalizerSet("a", (" missing-token",)),
            VerbalizerSet("b", (" no",)),
        )
        with pytest.raises(ValueError, match="missing-token"):
            d.resolve(fake_encode)


# ---------------------------------------------------------------------------
# The value: D = logsumexp(A) − logsumexp(B) over last-position logits
# ---------------------------------------------------------------------------

class TestValueFromLogits:
    def test_singleton_equals_legacy_logit_diff_exactly(self):
        """Backward compatibility: singletons → plain logit difference."""
        d = DecisionFunctional.from_tokens(" Mary", " John").resolve(fake_encode)
        logits = logits_with([(10, 3.25), (11, 1.75)])
        assert d.value_from_logits(logits) == pytest.approx(3.25 - 1.75)

    def test_set_value_is_logsumexp_difference(self):
        d = DecisionFunctional(
            VerbalizerSet("approve", (" approved", " yes")),
            VerbalizerSet("deny", (" denied",)),
        ).resolve(fake_encode)
        logits = logits_with([(20, 2.0), (22, 1.0), (30, 0.5)])
        expected = math.log(math.exp(2.0) + math.exp(1.0)) - 0.5
        assert d.value_from_logits(logits) == pytest.approx(expected)

    def test_logsumexp_is_numerically_stable_for_large_logits(self):
        d = DecisionFunctional(
            VerbalizerSet("a", (" approved", " yes")),
            VerbalizerSet("b", (" denied",)),
        ).resolve(fake_encode)
        logits = logits_with([(20, 1000.0), (22, 999.0), (30, 998.0)])
        v = d.value_from_logits(logits)
        assert math.isfinite(v)
        expected = 1000.0 + math.log(1 + math.exp(-1.0)) - 998.0
        assert v == pytest.approx(expected)

    def test_adding_weak_variant_never_decreases_set_evidence(self):
        """Monotonicity: logsumexp grows when a variant is added."""
        base = DecisionFunctional(
            VerbalizerSet("a", (" approved",)),
            VerbalizerSet("b", (" denied",)),
        ).resolve(fake_encode)
        bigger = DecisionFunctional(
            VerbalizerSet("a", (" approved", " yes")),
            VerbalizerSet("b", (" denied",)),
        ).resolve(fake_encode)
        logits = logits_with([(20, 2.0), (22, -1.0), (30, 1.0)])
        assert bigger.value_from_logits(logits) > base.value_from_logits(logits)

    def test_multi_token_resolution_refuses_value_from_logits(self):
        """Multi-token variants need sequence scores, not one-position logits."""
        d = DecisionFunctional(
            VerbalizerSet("a", ("App+roved",)),
            VerbalizerSet("b", (" denied",)),
        ).resolve(fake_encode)
        with pytest.raises(ValueError, match="single-token"):
            d.value_from_logits(logits_with([(40, 1.0), (30, 0.0)]))


# ---------------------------------------------------------------------------
# Sequence scoring path (multi-token variants) via injected score function
# ---------------------------------------------------------------------------

class TestValueFromScores:
    def test_sequence_value_uses_injected_scores(self):
        d = DecisionFunctional(
            VerbalizerSet("approve", ("App+roved",)),
            VerbalizerSet("deny", (" denied", " no")),
        ).resolve(fake_encode)

        scores = {(40, 41): -0.5, (30,): -2.0, (32,): -3.0}

        def score_fn(variant_ids):
            return scores[tuple(variant_ids)]

        expected = -0.5 - math.log(math.exp(-2.0) + math.exp(-3.0))
        assert d.value_from_scores(score_fn) == pytest.approx(expected)

    def test_score_fn_failure_is_wrapped_with_variant_context(self):
        d = DecisionFunctional(
            VerbalizerSet("a", (" yes",)),
            VerbalizerSet("b", (" no",)),
        ).resolve(fake_encode)

        def bad(_ids):
            raise RuntimeError("model exploded")

        with pytest.raises(ValueError, match="yes"):
            d.value_from_scores(bad)


# ---------------------------------------------------------------------------
# Serialization for the evidence vault
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_round_trip_documents_the_decision(self):
        d = DecisionFunctional(
            VerbalizerSet("approve", (" approved", " yes")),
            VerbalizerSet("deny", (" denied",)),
        )
        payload = d.to_dict()
        assert payload["positive"]["label"] == "approve"
        assert payload["positive"]["variants"] == [" approved", " yes"]
        assert payload["negative"]["variants"] == [" denied"]
