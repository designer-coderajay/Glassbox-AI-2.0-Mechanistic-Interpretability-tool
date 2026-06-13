"""Tests for core._resolve_decision_tokens multi-token tolerance (V5 safe increment).

The verbalizer-set path previously crashed on multi-token variants (" Approved").
It now resolves them to a representative token, matching the legacy str fallback.
"""
import pytest

from glassbox.core import _resolve_decision_tokens


class _FakeTokens:
    """Mimics model.to_tokens(s) so that [0, -1].item() returns a fixed id."""

    def __init__(self, last_id):
        self._last = last_id

    def __getitem__(self, idx):  # idx == (0, -1)
        return self

    def item(self):
        return self._last


_SINGLE = {" Yes": 5297, " yes": 8505, " No": 1400, " no": 645}
_LAST = {" Approved": 38493, " Denied": 47557, " Approve": 16835, " Deny": 26747}


def _to_single(s):
    if s in _SINGLE:
        return _SINGLE[s]
    raise AssertionError(f"{s!r} is not a single token")


def _to_tokens(s):
    return _FakeTokens(_LAST.get(s, abs(hash(s)) % 50000))


def test_legacy_str_str_passthrough():
    t, d, meta, pc, pi = _resolve_decision_tokens(_to_single, _to_tokens, " Mary", " John")
    assert (t, d, meta) == (None, None, None)
    assert pc == " Mary" and pi == " John"


def test_single_token_set_unchanged():
    t, d, meta, pc, _ = _resolve_decision_tokens(
        _to_single, _to_tokens, [" Yes", " yes"], [" No", " no"]
    )
    assert t == [5297, 8505]
    assert d == [1400, 645]
    assert meta["token_resolution"] == "single_token"
    assert "multi_token_variants" not in meta
    assert pc == " Yes"


def test_multi_token_set_resolves_to_representative():
    t, d, meta, _, _ = _resolve_decision_tokens(
        _to_single, _to_tokens, [" Approved", " Approve"], [" Denied", " Deny"]
    )
    assert t == [38493, 16835]   # last token of each multi-token variant
    assert d == [47557, 26747]
    assert meta["token_resolution"] == "representative_token"
    assert set(meta["multi_token_variants"]) == {" Approved", " Approve", " Denied", " Deny"}
    assert "representative" in meta["resolution_note"]


def test_mixed_single_and_multi():
    t, _, meta, _, _ = _resolve_decision_tokens(
        _to_single, _to_tokens, [" Yes", " Approved"], [" No", " Denied"]
    )
    assert t == [5297, 38493]
    assert meta["token_resolution"] == "representative_token"  # any multi flips it
    assert meta["multi_token_variants"] == sorted([" Approved", " Denied"])


def test_overlap_still_raises():
    # positive and negative resolving to the same token is ill-defined -> raise
    with pytest.raises(ValueError):
        _resolve_decision_tokens(_to_single, _to_tokens, [" Yes"], [" Yes"])
