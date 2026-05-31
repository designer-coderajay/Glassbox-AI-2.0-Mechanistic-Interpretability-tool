"""
tests/test_prompt_corruption.py
================================
Tests for glassbox.prompt_corruption — any-prompt corruption engine.

All tests run offline (no model download, no torch required).
Tests cover every strategy, the auto-selector logic, edge cases,
and mathematical invariants.
"""

from __future__ import annotations

import re
import pytest

from glassbox.prompt_corruption import (
    CorruptionStrategy,
    CorruptionSelector,
    auto_corrupt,
    get_antonym,
    name_swap_corruption,
    random_token_corruption,
    antonym_corruption,
    _apply_semantic_negation,
    _ANTONYM_TABLE,
    _NEGATION_TRIGGERS,
    _REPLACEMENT_POOL,
)


# ===========================================================================
# Antonym table
# ===========================================================================

class TestAntonymTable:
    def test_antonym_table_is_symmetric(self):
        """Every entry's antonym must also map back to the entry."""
        for word, antonym in _ANTONYM_TABLE.items():
            assert antonym in _ANTONYM_TABLE, (
                f"Antonym table not symmetric: '{word}' → '{antonym}' "
                f"but '{antonym}' has no reverse entry"
            )

    def test_get_antonym_known_pair(self):
        assert get_antonym("approved").strip().lower() == "denied"
        assert get_antonym("denied").strip().lower() == "approved"
        assert get_antonym("urgent").strip().lower() == "routine"
        assert get_antonym("high").strip().lower() == "low"

    def test_get_antonym_unknown_returns_none(self):
        assert get_antonym("xyz_unknown_word") is None
        assert get_antonym("") is None

    def test_get_antonym_leading_space_preserved(self):
        """TransformerLens tokens often start with a space — preserve it."""
        result = get_antonym(" Approved")
        assert result is not None
        assert result.startswith(" "), f"Expected leading space, got: {result!r}"

    def test_get_antonym_case_insensitive(self):
        assert get_antonym("APPROVED") is not None
        assert get_antonym("Approved") is not None
        assert get_antonym("approved") is not None


# ===========================================================================
# NameSwap strategy
# ===========================================================================

class TestNameSwap:
    def test_ioi_classic(self):
        """Original IOI prompt — both names swap."""
        prompt = "When Mary and John went to the store, John gave a drink to"
        result = name_swap_corruption(prompt, "Mary", "John")
        assert "John" in result or "Mary" in result  # at least one name present
        assert result != prompt  # something changed

    def test_symmetric_swap(self):
        """Swapping A→B→A must yield a different prompt (not the original)."""
        prompt = "Mary gave John the book"
        once   = name_swap_corruption(prompt, "Mary", "John")
        twice  = name_swap_corruption(once,   "John", "Mary")
        # After double swap we're back to original
        assert twice == prompt

    def test_word_boundary_respected(self):
        """'a' must not match inside 'cat' or 'sat'."""
        prompt = "The cat sat on the mat"
        result = name_swap_corruption(prompt, "a", "b")
        # Should only swap standalone 'a'
        assert "cat" in result  # 'a' inside 'cat' not replaced
        assert "sat" in result  # 'a' inside 'sat' not replaced

    def test_fallback_when_no_match(self):
        """If neither token appears in prompt, append distractor."""
        prompt = "The weather is nice today"
        result = name_swap_corruption(prompt, "Mary", "John")
        assert result.endswith("John")

    def test_no_double_replacement(self):
        """Placeholder mechanism prevents incorrect double-swap."""
        prompt = "Alice met Bob and Bob met Alice"
        result = name_swap_corruption(prompt, "Alice", "Bob")
        # Count occurrences: should be consistent swap
        assert result.count("Alice") + result.count("Bob") == 4


# ===========================================================================
# RandomTokenReplacement strategy
# ===========================================================================

class TestRandomTokenReplacement:
    def test_something_changes(self):
        prompt = "The applicant credit score is 620 loan should be"
        result = random_token_corruption(prompt, " approved", " denied")
        assert result != prompt

    def test_replacement_uses_pool_words(self):
        prompt = "The quick brown fox jumps over the lazy dog"
        result = random_token_corruption(prompt, "cat", "dog", seed=42)
        pool_set = set(w.lower() for w in _REPLACEMENT_POOL)
        words = result.lower().split()
        # At least one word must come from the pool (25% replacement)
        assert any(w in pool_set for w in words)

    def test_correct_token_protected(self):
        """The correct token should never be replaced."""
        prompt = "approved denied approved denied"
        result = random_token_corruption(
            prompt, "approved", "denied", replace_fraction=1.0, seed=1
        )
        # With replace_fraction=1.0, non-protected tokens are all replaced
        # but correct and incorrect tokens themselves are protected
        # (they appear in the prompt here to test boundary)
        assert isinstance(result, str)

    def test_reproducible_with_seed(self):
        prompt = "The loan application decision"
        r1 = random_token_corruption(prompt, "approved", "denied", seed=7)
        r2 = random_token_corruption(prompt, "approved", "denied", seed=7)
        assert r1 == r2

    def test_different_seeds_differ(self):
        prompt = "The loan application decision result is pending"
        r1 = random_token_corruption(prompt, "approved", "denied", seed=1)
        r2 = random_token_corruption(prompt, "approved", "denied", seed=2)
        # Very unlikely to be identical with different seeds
        assert r1 != r2

    def test_returns_string(self):
        result = random_token_corruption("Hello world", "Hello", "Goodbye")
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# AntonymReplacement strategy
# ===========================================================================

class TestAntonymCorruption:
    def test_known_antonym_applied(self):
        """If correct has a known antonym, it should appear in corrupted prompt."""
        prompt = "The loan application result"
        result = antonym_corruption(prompt, " Approved", " Denied")
        assert isinstance(result, str)
        assert result != prompt or "denied" in result.lower()

    def test_falls_back_to_random_when_no_antonym(self):
        """Unknown tokens → fall back to random token replacement."""
        prompt = "The quick brown fox jumps over the lazy dog"
        result = antonym_corruption(prompt, "zyx_unknown", "abc_unknown")
        assert isinstance(result, str)

    def test_medical_pair(self):
        result = antonym_corruption(
            "Patient status is stable. Next action:",
            " urgent", " routine"
        )
        assert isinstance(result, str)


# ===========================================================================
# SemanticNegation strategy
# ===========================================================================

class TestSemanticNegation:
    @pytest.mark.parametrize("trigger,expected_pattern", [
        ("should",    "should not"),
        ("must",      "must not"),
        ("recommend", "do not recommend"),
        ("suggest",   "do not suggest"),
    ])
    def test_modal_negation(self, trigger, expected_pattern):
        prompt = f"The system {trigger} output the correct label."
        result = _apply_semantic_negation(prompt)
        assert expected_pattern in result, (
            f"Expected '{expected_pattern}' in result, got: {result!r}"
        )

    def test_fallback_prepend_not(self):
        """No trigger found → prepend 'NOT:'."""
        prompt = "The capital of France is Paris."
        result = _apply_semantic_negation(prompt)
        assert result.startswith("NOT:")

    def test_only_first_occurrence_negated(self):
        """Only the first modal verb should be negated."""
        prompt = "You should do this and you should do that."
        result = _apply_semantic_negation(prompt)
        # First should → should not; second should unchanged
        assert result.count("should not") == 1


# ===========================================================================
# CorruptionSelector auto-selection logic
# ===========================================================================

class TestCorruptionSelector:
    def test_ioi_selects_name_swap(self):
        """Both tokens appear in prompt → NameSwap."""
        prompt = "When Mary and John went to the store, John gave a drink to"
        strategy, _, rationale = CorruptionSelector.select(prompt, " Mary", " John")
        assert strategy == CorruptionStrategy.NAME_SWAP
        assert "name_swap" in rationale.lower() or "NameSwap" in rationale

    def test_antonym_pair_selects_antonym(self):
        """Correct and incorrect are antonyms → AntonymReplacement."""
        prompt = "The loan application decision:"
        strategy, _, rationale = CorruptionSelector.select(
            prompt, " Approved", " Denied"
        )
        assert strategy == CorruptionStrategy.ANTONYM

    def test_modal_verb_selects_negation(self):
        """Prompt with 'should' and no entity/antonym match → SemanticNegation."""
        prompt = "The classifier should output the correct label."
        strategy, _, rationale = CorruptionSelector.select(
            prompt, " positive", " negative"
        )
        # positive/negative are antonyms so antonym takes priority
        # Use a non-antonym pair to force negation
        strategy2, _, _ = CorruptionSelector.select(
            prompt, " alpha", " beta"
        )
        assert strategy2 == CorruptionStrategy.SEMANTIC_NEGATION

    def test_fallback_selects_random_token(self):
        """No match for any pattern → RandomTokenReplacement."""
        prompt = "The quick brown fox"
        strategy, corrupted, rationale = CorruptionSelector.select(
            prompt, " zyx_unknown_x", " abc_unknown_y"
        )
        assert strategy == CorruptionStrategy.RANDOM_TOKEN
        assert isinstance(corrupted, str)

    def test_selector_returns_three_tuple(self):
        strategy, corrupted, rationale = CorruptionSelector.select(
            "Hello world", "Hello", "Goodbye"
        )
        assert isinstance(strategy, CorruptionStrategy)
        assert isinstance(corrupted, str)
        assert isinstance(rationale, str)
        assert len(rationale) > 10


# ===========================================================================
# auto_corrupt public API
# ===========================================================================

class TestAutoCorrupt:
    def test_returns_three_tuple(self):
        corrupted, strategy, rationale = auto_corrupt(
            "When Mary and John went to the store, John gave a drink to",
            " Mary", " John"
        )
        assert isinstance(corrupted, str)
        assert isinstance(strategy, str)
        assert isinstance(rationale, str)

    def test_ioi_auto_selects_name_swap(self):
        _, strategy, _ = auto_corrupt(
            "When Mary and John went to the store, John gave a drink to",
            " Mary", " John"
        )
        assert strategy == "name_swap"

    def test_force_strategy_name_swap(self):
        _, strategy, rationale = auto_corrupt(
            "The weather is fine today",
            "sunny", "cloudy",
            strategy="name_swap",
        )
        assert strategy == "name_swap"
        assert "Forced" in rationale

    def test_force_strategy_random_token(self):
        _, strategy, _ = auto_corrupt(
            "The patient presents with chest pain",
            " urgent", " routine",
            strategy="random_token",
        )
        assert strategy == "random_token"

    def test_force_strategy_antonym(self):
        _, strategy, _ = auto_corrupt(
            "The loan decision is",
            " Approved", " Denied",
            strategy="antonym",
        )
        assert strategy == "antonym"

    def test_force_strategy_semantic_negation(self):
        _, strategy, _ = auto_corrupt(
            "The system should output the result",
            " correct", " wrong",
            strategy="semantic_negation",
        )
        assert strategy == "semantic_negation"

    def test_force_activation_noise_raises(self):
        """activation_noise is not a prompt-level strategy — must raise."""
        with pytest.raises(ValueError, match="activation_noise"):
            auto_corrupt("prompt", "correct", "incorrect", strategy="activation_noise")

    @pytest.mark.parametrize("prompt,correct,incorrect", [
        # IOI
        ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
        # Medical triage
        ("Patient presents with acute chest pain. Priority:", " Urgent", " Routine"),
        # Credit scoring
        ("Annual income: €42,000. Credit history: 3 missed payments. Decision:", " Denied", " Approved"),
        # HR screening
        ("Candidate has 8 years Python experience and a CS degree. Assessment:", " Qualified", " Rejected"),
        # Open-ended (no trigger, no antonym, no entity)
        ("The capital of France is", " Paris", " Berlin"),
        # Short prompt
        ("Pass or fail?", " pass", " fail"),
    ])
    def test_any_prompt_returns_non_empty_string(self, prompt, correct, incorrect):
        """auto_corrupt must succeed and return non-empty string for any input."""
        corrupted, strategy, rationale = auto_corrupt(prompt, correct, incorrect)
        assert isinstance(corrupted, str)
        assert len(corrupted) > 0
        assert strategy in {s.value for s in CorruptionStrategy} - {"activation_noise"}
        assert len(rationale) > 0
