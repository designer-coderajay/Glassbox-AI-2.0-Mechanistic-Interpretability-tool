"""
tests/test_decision_resolution.py — V5: analyze()'s input resolver.

Pure-logic tests for glassbox.core._resolve_decision_tokens — the function
that turns legacy str/str inputs or verbalizer-set sequences into decision
tokens. No torch, no model: encode_single is a dict lookup and to_tokens is a
fake returning a representative last-token id for multi-token variants.
"""

import pytest

from glassbox.core import _resolve_decision_tokens

_VOCAB = {
    " Mary": 10, " John": 11,
    " approved": 20, " approve": 21, " yes": 22,
    " denied": 30, " no": 32,
}


def encode_single(text):
    if text not in _VOCAB:
        raise KeyError(f"not a single token: {text!r}")
    return _VOCAB[text]


class _Toks:
    """Fake model.to_tokens(s): supports [0, -1].item() -> representative id."""

    def __init__(self, last):
        self._last = last

    def __getitem__(self, idx):
        return self

    def item(self):
        return self._last


_FALLBACK = {"NotAToken": 999, " Approved": 38493, " Denied": 47557}


def to_tokens(text):
    return _Toks(_FALLBACK.get(text, abs(hash(text)) % 50000))


class TestLegacyPath:
    def test_str_str_defers_to_legacy_resolution(self):
        t, d, meta, pc, pi = _resolve_decision_tokens(
            encode_single, to_tokens, " Mary", " John"
        )
        assert t is None and d is None and meta is None
        assert (pc, pi) == (" Mary", " John")


class TestSetPath:
    def test_lists_resolve_to_id_lists(self):
        t, d, meta, pc, pi = _resolve_decision_tokens(
            encode_single, to_tokens, [" approved", " yes"], [" denied"]
        )
        assert t == [20, 22]
        assert d == [30]
        assert meta["token_resolution"] == "single_token"

    def test_metadata_documents_the_sets(self):
        _, _, meta, _, _ = _resolve_decision_tokens(
            encode_single, to_tokens, [" approved", " yes"], [" denied"]
        )
        assert meta["positive"]["variants"] == [" approved", " yes"]
        assert meta["negative"]["variants"] == [" denied"]

    def test_primary_strings_are_first_variants(self):
        _, _, _, pc, pi = _resolve_decision_tokens(
            encode_single, to_tokens, [" approved", " yes"], [" denied", " no"]
        )
        assert (pc, pi) == (" approved", " denied")

    def test_mixed_str_and_list_supported(self):
        t, d, meta, _, _ = _resolve_decision_tokens(
            encode_single, to_tokens, " approved", [" denied", " no"]
        )
        assert t == [20]
        assert d == [30, 32]
        assert meta is not None

    def test_overlapping_sets_rejected(self):
        with pytest.raises(ValueError, match="[Oo]verlap"):
            _resolve_decision_tokens(
                encode_single, to_tokens, [" yes"], [" yes", " no"]
            )

    def test_multi_token_variant_now_resolved(self):
        # V5 safe increment: multi-token variants are resolved to a representative
        # (last) token rather than rejected, matching the legacy string fallback.
        t, _, meta, _, _ = _resolve_decision_tokens(
            encode_single, to_tokens, [" approved", "NotAToken"], [" denied"]
        )
        assert t == [20, 999]  # " approved"->20 (single); "NotAToken"->999 (fallback)
        assert meta["token_resolution"] == "representative_token"
        assert "NotAToken" in meta["multi_token_variants"]

    def test_duplicate_variants_deduplicated(self):
        t, _, _, _, _ = _resolve_decision_tokens(
            encode_single, to_tokens, [" yes", " yes"], [" no"]
        )
        assert t == [22]


class TestCFNoiseFloor:
    """V5 Sprint 4: relative noise floor for counterfactual verification."""

    def test_scales_with_clean_magnitude(self):
        from glassbox.core import _cf_noise_floor
        assert _cf_noise_floor(4.0) == pytest.approx(0.04)
        assert _cf_noise_floor(-4.0) == pytest.approx(0.04)

    def test_absolute_floor_when_clean_ld_tiny(self):
        from glassbox.core import _cf_noise_floor
        assert _cf_noise_floor(0.0) == pytest.approx(1e-6)
        assert _cf_noise_floor(1e-9) == pytest.approx(1e-6)
