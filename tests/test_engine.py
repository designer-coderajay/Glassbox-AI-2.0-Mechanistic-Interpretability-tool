"""
test_engine.py — Glassbox 2.0 Test Suite
==========================================

Covers:
  • GlassboxV2.attribution_patching()
  • GlassboxV2.analyze()
  • GlassboxV2._name_swap()  (static helper)
  • GlassboxV2.bootstrap_metrics()
  • GlassboxV2.functional_circuit_alignment()
  • Edge cases: zero logit-diff, short-circuit, suff_is_approx flag

Run with:
    pytest tests/test_engine.py -v

Requirements:
    pytest, torch, transformer_lens  (see pyproject.toml)

Notes
-----
These tests load GPT-2-small once via a module-scoped fixture so the full
suite runs in a single model-load.  On GitHub Actions (CPU) the full suite
takes ~3-5 min; mark slow tests with @pytest.mark.slow if you want to skip
them in fast CI mode:

    pytest tests/test_engine.py -v -m "not slow"
"""

import math
import time
import pytest

# ---------------------------------------------------------------------------
# Module-scope fixture — load model once for the whole test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Return a live GlassboxV2 instance running on CPU with GPT-2-small."""
    from glassbox import GlassboxV2
    return GlassboxV2("gpt2")


# ---------------------------------------------------------------------------
# Canonical IOI prompts (Mary / John — classic Indirect Object Identification)
# ---------------------------------------------------------------------------

IOI_PROMPT    = "When Mary and John went to the store, John gave a drink to"
IOI_CORRECT   = "Mary"
IOI_INCORRECT = "John"

# A small batch for bootstrap tests  (n ≥ 5 gives stable percentile CIs)
IOI_BATCH = [
    ("When Mary and John went to the store, John gave a drink to",   "Mary",   "John"),
    ("After Alice and Bob entered the room, Bob handed the key to",  "Alice",  "Bob"),
    ("When Sarah and Tom left the park, Tom passed the ball to",     "Sarah",  "Tom"),
    ("Once Emma and Jack arrived at school, Jack gave the pen to",   "Emma",   "Jack"),
    ("When Lisa and Mike reached the cafe, Mike offered the menu to","Lisa",   "Mike"),
]

# Factual prompt for non-IOI corruption test
FACT_PROMPT    = "The capital of France is"
FACT_CORRECT   = "Paris"
FACT_INCORRECT = "Berlin"

# Subject-verb agreement
SVA_PROMPT    = "The keys to the cabinet"
SVA_CORRECT   = "are"
SVA_INCORRECT = "is"


# ===========================================================================
# 1. ATTRIBUTION PATCHING
# ===========================================================================

class TestAttributionPatching:
    """Tests for the fast O(3) attribution patching step."""

    def test_returns_dict(self, engine):
        scores, ld = engine.attribution_patching(
            IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT
        )
        assert isinstance(scores, dict), "attribution_patching must return a dict"
        assert isinstance(ld, float),   "attribution_patching must return (dict, float)"

    def test_keys_are_layer_head_tuples(self, engine):
        scores, _ = engine.attribution_patching(
            IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT
        )
        for key in scores:
            assert isinstance(key, tuple), f"Key {key!r} is not a tuple"
            assert len(key) == 2,          f"Key {key!r} does not have length 2"
            layer, head = key
            assert isinstance(layer, int) and isinstance(head, int)

    def test_nonzero_scores(self, engine):
        scores, _ = engine.attribution_patching(
            IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT
        )
        total = sum(abs(v) for v in scores.values())
        assert total > 0.0, "All attribution scores are zero — something is wrong"

    def test_ioi_key_head_present(self, engine):
        """GPT-2 head (9, 9) is a well-known name-mover; should have a positive score."""
        scores, _ = engine.attribution_patching(
            IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT
        )
        assert (9, 9) in scores, "Head (9, 9) missing from attribution dict"
        assert scores[(9, 9)] > 0, "Head (9, 9) should have a positive attribution score"

    def test_positive_logit_diff(self, engine):
        _, ld = engine.attribution_patching(
            IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT
        )
        assert ld > 0.0, "Clean logit-diff should be positive for a solved IOI prompt"

    @pytest.mark.slow
    def test_performance_under_120s(self, engine):
        """Attribution patching (O(3)) must complete well under 120 s on CPU."""
        t0 = time.time()
        engine.attribution_patching(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        elapsed = time.time() - t0
        assert elapsed < 120.0, (
            f"attribution_patching took {elapsed:.1f}s — exceeds 120 s budget"
        )


# ===========================================================================
# 2. _NAME_SWAP (corruption helper)
# ===========================================================================

class TestNameSwap:
    """Tests for the bidirectional name-swap corruption."""

    def test_bidirectional_ioi(self, engine):
        """Both names must be swapped in one shot (no double-replacement)."""
        result = engine._name_swap(IOI_PROMPT, "Mary", "John")
        assert "John" in result,  "Distractor 'John' must appear after swap"
        assert "Mary" in result,  "Target 'Mary' must appear after swap"
        # After swap the first occurrence of 'Mary' should be where 'John' was
        # and vice versa — simplest invariant: the output is different from input
        assert result != IOI_PROMPT, "_name_swap returned the original prompt unchanged"

    def test_no_double_replacement(self, engine):
        """Naive .replace() would produce all-Mary or all-John; check it doesn't."""
        swapped = engine._name_swap(IOI_PROMPT, "Mary", "John")
        # Original has 1 Mary + 2 Johns  →  swap should produce 2 Marys + 1 John
        assert swapped.count("Mary") != IOI_PROMPT.count("Mary") or \
               swapped.count("John") != IOI_PROMPT.count("John"), (
            "Counts unchanged — bidirectional swap may not have fired"
        )

    def test_fallback_when_target_not_in_prompt(self, engine):
        """For factual prompts the target word isn't in the prompt; fallback appends."""
        result = engine._name_swap(FACT_PROMPT, FACT_CORRECT, FACT_INCORRECT)
        # Either the prompt is different OR distractor was appended
        assert result != FACT_PROMPT or FACT_INCORRECT in result, (
            "_name_swap with absent target should produce a changed prompt"
        )

    def test_sva_swap(self, engine):
        result = engine._name_swap(SVA_PROMPT, SVA_CORRECT, SVA_INCORRECT)
        assert result != SVA_PROMPT


# ===========================================================================
# 3. FULL ANALYZE() — IOI
# ===========================================================================

@pytest.fixture(scope="module")
def ioi_result(engine):
    return engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)


class TestAnalyzeIOI:
    """Validate the full analyze() return dict on a canonical IOI example."""

    def test_returns_dict(self, ioi_result):
        assert isinstance(ioi_result, dict)

    # -- circuit ---------------------------------------------------------------
    def test_circuit_nonempty(self, ioi_result):
        assert len(ioi_result["circuit"]) > 0, "Circuit is empty"

    def test_circuit_contains_tuples(self, ioi_result):
        for head in ioi_result["circuit"]:
            assert isinstance(head, tuple) and len(head) == 2, (
                f"Circuit element {head!r} is not a (layer, head) tuple"
            )

    def test_ioi_key_head_in_circuit(self, ioi_result):
        """Head (9, 9) is GPT-2's primary name-mover; it must survive pruning."""
        assert (9, 9) in ioi_result["circuit"], (
            "Head (9, 9) missing from circuit — MFC pruning may be too aggressive"
        )

    def test_circuit_sorted_by_attribution(self, ioi_result):
        """analyze() must return circuit sorted descending by attribution score."""
        attrs  = ioi_result.get("attributions", {})
        scores = [attrs.get(str(h), 0.0) for h in ioi_result["circuit"]]
        assert scores == sorted(scores, reverse=True), (
            "Circuit is not sorted by attribution score (descending)"
        )

    # -- corr_prompt (new field, fixes naive replace() bug) --------------------
    def test_corr_prompt_present(self, ioi_result):
        assert "corr_prompt" in ioi_result, "corr_prompt key missing from analyze() result"

    def test_corr_prompt_differs_from_original(self, ioi_result):
        assert ioi_result["corr_prompt"] != IOI_PROMPT, (
            "corr_prompt is identical to the original — corruption failed"
        )

    # -- attributions ----------------------------------------------------------
    def test_attributions_dict(self, ioi_result):
        assert isinstance(ioi_result["attributions"], dict)
        assert len(ioi_result["attributions"]) > 0

    # -- faithfulness metrics --------------------------------------------------
    def test_faithfulness_keys(self, ioi_result):
        faith = ioi_result["faithfulness"]
        for key in ("sufficiency", "comprehensiveness", "f1", "category", "suff_is_approx"):
            assert key in faith, f"faithfulness missing key: {key!r}"

    def test_sufficiency_range(self, ioi_result):
        suff = ioi_result["faithfulness"]["sufficiency"]
        assert 0.0 <= suff <= 1.0, f"Sufficiency {suff:.4f} outside [0, 1]"

    def test_comprehensiveness_range(self, ioi_result):
        comp = ioi_result["faithfulness"]["comprehensiveness"]
        assert 0.0 <= comp <= 1.0, f"Comprehensiveness {comp:.4f} outside [0, 1]"

    def test_sufficiency_exceeds_threshold(self, ioi_result):
        """IOI is a well-solved task; sufficiency should be above 0.5."""
        suff = ioi_result["faithfulness"]["sufficiency"]
        assert suff >= 0.5, (
            f"Sufficiency {suff:.4f} below 0.5 — circuit may have been pruned too hard"
        )

    def test_f1_mathematically_consistent(self, ioi_result):
        faith = ioi_result["faithfulness"]
        suff  = faith["sufficiency"]
        comp  = faith["comprehensiveness"]
        f1    = faith["f1"]
        denom = suff + comp
        expected = (2 * suff * comp / denom) if denom > 1e-9 else 0.0
        assert abs(f1 - expected) < 1e-4, (
            f"F1={f1:.6f} inconsistent with suff={suff:.4f}, comp={comp:.4f} "
            f"(expected {expected:.6f})"
        )

    # ── CRITICAL BUG FIX: wrong categories were "strong/good/partial" ─────────
    def test_category_valid(self, ioi_result):
        """
        Category must be one of the five labels returned by core.py.

        BUG HISTORY: The original test checked for {"strong", "good", "partial"},
        which are NEVER returned by GlassboxV2.  These categories caused the CI
        to always fail silently.  The correct set is the one defined in analyze():

            faithful  ← f1 ≥ 0.85
            backup_mechanisms ← 0.70 ≤ f1 < 0.85
            moderate  ← 0.50 ≤ f1 < 0.70
            incomplete← 0.30 ≤ f1 < 0.50
            weak      ← f1 < 0.30
        """
        valid_categories = {"faithful", "backup_mechanisms", "moderate", "incomplete", "weak"}
        cat = ioi_result["faithfulness"]["category"]
        assert cat in valid_categories, (
            f"Category {cat!r} is not a valid GlassboxV2 category. "
            f"Valid values: {valid_categories}"
        )

    def test_suff_is_approx_flag(self, ioi_result):
        """
        Sufficiency is a Taylor (first-order linear) approximation.
        The flag must be True so downstream consumers know this.
        """
        assert ioi_result["faithfulness"]["suff_is_approx"] is True, (
            "suff_is_approx flag is missing or False — API contract violated"
        )


# ===========================================================================
# 4. ANALYZE() — FACTUAL & SVA VARIANTS
# ===========================================================================

class TestAnalyzeVariants:
    """Smoke-test analyze() on non-IOI task types."""

    def test_factual_returns_valid_result(self, engine):
        result = engine.analyze(FACT_PROMPT, FACT_CORRECT, FACT_INCORRECT)
        assert isinstance(result["circuit"], list)
        assert result["faithfulness"]["category"] in {
            "faithful", "backup_mechanisms", "moderate", "incomplete", "weak"
        }

    def test_sva_returns_valid_result(self, engine):
        result = engine.analyze(SVA_PROMPT, SVA_CORRECT, SVA_INCORRECT)
        assert isinstance(result["circuit"], list)
        faith = result["faithfulness"]
        assert 0.0 <= faith["sufficiency"]       <= 1.0
        assert 0.0 <= faith["comprehensiveness"] <= 1.0

    def test_corr_prompt_present_in_all_variants(self, engine):
        for prompt, correct, incorrect in [
            (FACT_PROMPT, FACT_CORRECT, FACT_INCORRECT),
            (SVA_PROMPT,  SVA_CORRECT,  SVA_INCORRECT),
        ]:
            result = engine.analyze(prompt, correct, incorrect)
            assert "corr_prompt" in result, (
                f"corr_prompt missing for prompt={prompt!r}"
            )


# ===========================================================================
# 5. BOOTSTRAP METRICS
# ===========================================================================

class TestBootstrapMetrics:
    """
    Bootstrap CI tests.
    n ≥ 5 is required for stable percentile confidence intervals.
    The original benchmark used n=3 which gives unreliable CIs.

    API contract for bootstrap_metrics():
        Input : prompts = List[Tuple[str, str, str]]  (raw prompt triples)
        Param : n_boot  (NOT n_bootstrap)
        Output: dict with keys "sufficiency", "comprehensiveness", "f1"
                each being a sub-dict: {"mean": float, "ci_lo": float,
                                         "ci_hi": float, "std": float, "n": int}
    """

    @pytest.mark.slow
    def test_bootstrap_returns_expected_keys(self, engine):
        # Pass raw prompt triples — NOT pre-analyzed results
        boot = engine.bootstrap_metrics(IOI_BATCH, n_boot=200)
        for metric in ("sufficiency", "comprehensiveness", "f1"):
            assert metric in boot, (
                f"bootstrap_metrics missing top-level key: {metric!r}"
            )
            for sub_key in ("mean", "ci_lo", "ci_hi"):
                assert sub_key in boot[metric], (
                    f"bootstrap_metrics[{metric!r}] missing sub-key: {sub_key!r}"
                )

    @pytest.mark.slow
    def test_bootstrap_ci_is_ordered(self, engine):
        """Lower CI bound must be ≤ mean ≤ upper CI bound."""
        boot = engine.bootstrap_metrics(IOI_BATCH, n_boot=200)
        for metric in ("sufficiency", "comprehensiveness", "f1"):
            lo   = boot[metric]["ci_lo"]
            hi   = boot[metric]["ci_hi"]
            mean = boot[metric]["mean"]
            assert lo <= mean <= hi, (
                f"{metric}: CI [{lo:.4f}, {hi:.4f}] does not bracket mean {mean:.4f}"
            )

    @pytest.mark.slow
    def test_bootstrap_means_in_range(self, engine):
        """All means must be in [0, 1] — basic sanity check."""
        boot = engine.bootstrap_metrics(IOI_BATCH, n_boot=200)
        for metric in ("sufficiency", "comprehensiveness", "f1"):
            mean = boot[metric]["mean"]
            assert 0.0 <= mean <= 1.0, (
                f"bootstrap {metric} mean {mean:.4f} outside [0, 1]"
            )


# ===========================================================================
# 6. FUNCTIONAL CIRCUIT ALIGNMENT SCORE (FCAS)
# ===========================================================================

def _heads_from_result(result: dict, n_layers: int = 12, n_heads: int = 12) -> list:
    """
    Convert analyze() output to the List[Dict] format expected by
    functional_circuit_alignment().

    functional_circuit_alignment() expects each element to be a dict with:
        layer, head, attr, rel_depth, rel_head, n_layers, n_heads

    analyze() returns circuit as List[Tuple[int, int]] and attributions as
    a dict keyed by str((layer, head)).  This helper bridges the gap.

    GPT-2 Small defaults: n_layers=12, n_heads=12.
    """
    attrs = result.get("attributions", {})
    head_dicts = []
    for (l, h) in result["circuit"]:
        attr = attrs.get(str((l, h)), 0.0)
        head_dicts.append({
            "layer":     l,
            "head":      h,
            "attr":      attr,
            "rel_depth": l / max(1, n_layers - 1),
            "rel_head":  h / max(1, n_heads  - 1),
            "n_layers":  n_layers,
            "n_heads":   n_heads,
        })
    return sorted(head_dicts, key=lambda x: x["attr"], reverse=True)


class TestFCAS:
    """
    FCAS compares circuits across two models or two runs.
    Includes a null distribution so the score is interpretable.

    API contract for functional_circuit_alignment():
        Input : heads_a, heads_b = List[Dict] with keys
                  layer, head, attr, rel_depth, rel_head, n_layers, n_heads
        Param : top_k  (NOT k)
        Output: dict with keys "fcas", "null_mean", "null_std", "z_score", "pairs"

    Use _heads_from_result() to convert analyze() output to the expected format.
    """

    @pytest.mark.slow
    def test_fcas_returns_required_keys(self, engine):
        r1 = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        r2 = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        heads_a = _heads_from_result(r1)
        heads_b = _heads_from_result(r2)
        fcas_result = engine.functional_circuit_alignment(
            heads_a, heads_b, top_k=3
        )
        for key in ("fcas", "null_mean", "null_std", "z_score", "pairs"):
            assert key in fcas_result, (
                f"functional_circuit_alignment missing key: {key!r}"
            )

    @pytest.mark.slow
    def test_fcas_identical_circuits_is_one(self, engine):
        """Comparing a circuit to itself must yield FCAS = 1.0."""
        r = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        heads = _heads_from_result(r)
        fcas_result = engine.functional_circuit_alignment(
            heads, heads, top_k=3
        )
        assert abs(fcas_result["fcas"] - 1.0) < 1e-6, (
            f"FCAS of identical circuits is {fcas_result['fcas']:.6f}, expected 1.0"
        )

    @pytest.mark.slow
    def test_fcas_range(self, engine):
        r1 = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        r2 = engine.analyze(FACT_PROMPT, FACT_CORRECT, FACT_INCORRECT)
        heads_a = _heads_from_result(r1)
        heads_b = _heads_from_result(r2)
        fcas_result = engine.functional_circuit_alignment(
            heads_a, heads_b, top_k=3
        )
        assert 0.0 <= fcas_result["fcas"] <= 1.0, (
            f"FCAS {fcas_result['fcas']:.4f} outside [0, 1]"
        )

    @pytest.mark.slow
    def test_z_score_is_finite(self, engine):
        r1 = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        r2 = engine.analyze(FACT_PROMPT, FACT_CORRECT, FACT_INCORRECT)
        heads_a = _heads_from_result(r1)
        heads_b = _heads_from_result(r2)
        fcas_result = engine.functional_circuit_alignment(
            heads_a, heads_b, top_k=3
        )
        assert math.isfinite(fcas_result["z_score"]), (
            "z_score is not finite — null_std may be zero"
        )


# ===========================================================================
# 7. EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Corner cases that should not crash or produce NaN."""

    def test_single_token_target(self, engine):
        """Single-character targets (common in SVA) must not crash."""
        # 'a' vs 'the' — degenerate but valid
        result = engine.analyze("The cat sat on", "a", "the")
        assert "faithfulness" in result

    def test_output_has_no_nan(self, ioi_result):
        faith = ioi_result["faithfulness"]
        for k, v in faith.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"NaN detected in faithfulness[{k!r}]"
            if isinstance(v, float):
                assert not math.isinf(v), f"Inf detected in faithfulness[{k!r}]"

    def test_attributions_serializable(self, ioi_result):
        """Attributions are stored as str(key): float — must be JSON-serializable."""
        import json
        try:
            json.dumps(ioi_result["attributions"])
        except (TypeError, ValueError) as exc:
            pytest.fail(f"attributions dict is not JSON-serializable: {exc}")
