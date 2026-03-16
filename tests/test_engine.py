import math
import re
import subprocess
import sys
import time
import pytest

# ---------------------------------------------------------------------------
# Module-scope fixture — load model once for the whole test session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine():
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2
    model = HookedTransformer.from_pretrained("gpt2")
    return GlassboxV2(model)


# ---------------------------------------------------------------------------
# Canonical IOI prompts (Mary / John — classic Indirect Object Identification)
# ---------------------------------------------------------------------------
IOI_PROMPT    = "When Mary and John went to the store, John gave a drink to"
IOI_CORRECT   = "Mary"
IOI_INCORRECT = "John"

# A small batch for bootstrap tests  (n >= 5 gives stable percentile CIs)
IOI_BATCH = [
    ("When Mary and John went to the store, John gave a drink to",    "Mary",  "John"),
    ("After Alice and Bob entered the room, Bob handed the key to",   "Alice", "Bob"),
    ("When Sarah and Tom left the park, Tom passed the ball to",      "Sarah", "Tom"),
    ("Once Emma and Jack arrived at school, Jack gave the pen to",    "Emma",  "Jack"),
    ("When Lisa and Mike reached the cafe, Mike offered the menu to", "Lisa",  "Mike"),
]

# Factual prompt for non-IOI corruption test
FACT_PROMPT    = "The capital of France is"
FACT_CORRECT   = "Paris"
FACT_INCORRECT = "Berlin"

# Subject-verb agreement
SVA_PROMPT    = "The keys to the cabinet"
SVA_CORRECT   = "are"
SVA_INCORRECT = "is"


# ---------------------------------------------------------------------------
# BUG FIX: ioi_tokens fixture
#
# ORIGINAL BUG: Every test in TestAttributionPatching called:
#     engine.attribution_patching(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
# with 3 plain strings. But attribution_patching() requires:
#     (clean_tokens: torch.Tensor, corrupted_tokens: torch.Tensor,
#      target_token: int, distractor_token: int)
# This caused TypeError on every TestAttributionPatching test.
#
# FIX: This module-scope fixture tokenizes once and all TestAttributionPatching
# tests receive the correctly typed arguments.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ioi_tokens(engine):
    tokens_c    = engine.model.to_tokens(IOI_PROMPT)
    corr_prompt = engine._name_swap(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
    tokens_corr = engine.model.to_tokens(corr_prompt)
    try:
        t_tok = engine.model.to_single_token(IOI_CORRECT)
        d_tok = engine.model.to_single_token(IOI_INCORRECT)
    except Exception:
        t_tok = engine.model.to_tokens(IOI_CORRECT)[0, -1].item()
        d_tok = engine.model.to_tokens(IOI_INCORRECT)[0, -1].item()
    return tokens_c, tokens_corr, t_tok, d_tok


# ===========================================================================
# 1. ATTRIBUTION PATCHING
# ===========================================================================
class TestAttributionPatching:
    """Tests for the fast O(3) attribution patching step."""

    def test_returns_dict(self, engine, ioi_tokens):
        # BUG FIX: was attribution_patching(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        # — 3 strings. Now uses ioi_tokens fixture with correctly typed args.
        tokens_c, tokens_corr, t_tok, d_tok = ioi_tokens
        scores, ld = engine.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)
        assert isinstance(scores, dict), "attribution_patching must return a dict"
        assert isinstance(ld, float),    "attribution_patching must return (dict, float)"

    def test_keys_are_layer_head_tuples(self, engine, ioi_tokens):
        tokens_c, tokens_corr, t_tok, d_tok = ioi_tokens
        scores, _ = engine.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)
        for key in scores:
            assert isinstance(key, tuple), f"Key {key!r} is not a tuple"
            assert len(key) == 2,          f"Key {key!r} does not have length 2"
            layer, head = key
            assert isinstance(layer, int) and isinstance(head, int)

    def test_nonzero_scores(self, engine, ioi_tokens):
        tokens_c, tokens_corr, t_tok, d_tok = ioi_tokens
        scores, _ = engine.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)
        total = sum(abs(v) for v in scores.values())
        assert total > 0.0, "All attribution scores are zero — something is wrong"

    def test_ioi_key_head_present(self, engine, ioi_tokens):
        """GPT-2 head (9, 9) is a well-known name-mover; should have a positive score."""
        tokens_c, tokens_corr, t_tok, d_tok = ioi_tokens
        scores, _ = engine.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)
        assert (9, 9) in scores, "Head (9, 9) missing from attribution dict"
        assert scores[(9, 9)] > 0, "Head (9, 9) should have a positive attribution score"

    def test_positive_logit_diff(self, engine, ioi_tokens):
        tokens_c, tokens_corr, t_tok, d_tok = ioi_tokens
        _, ld = engine.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)
        assert ld > 0.0, "Clean logit-diff should be positive for a solved IOI prompt"

    @pytest.mark.slow
    def test_performance_under_120s(self, engine, ioi_tokens):
        """Attribution patching (O(3)) must complete well under 120 s on CPU."""
        tokens_c, tokens_corr, t_tok, d_tok = ioi_tokens
        t0 = time.time()
        engine.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)
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
        assert result != IOI_PROMPT, "_name_swap returned the original prompt unchanged"

    def test_no_double_replacement(self, engine):
        """Naive .replace() would produce all-Mary or all-John; check it doesn't."""
        swapped = engine._name_swap(IOI_PROMPT, "Mary", "John")
        # Original has 1 Mary + 2 Johns -> swap should produce 2 Marys + 1 John
        assert swapped.count("Mary") != IOI_PROMPT.count("Mary") or \
               swapped.count("John") != IOI_PROMPT.count("John"), (
            "Counts unchanged — bidirectional swap may not have fired"
        )

    def test_fallback_when_target_not_in_prompt(self, engine):
        """For factual prompts the target word isn't in the prompt; fallback appends."""
        result = engine._name_swap(FACT_PROMPT, FACT_CORRECT, FACT_INCORRECT)
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

    # -- corr_prompt -----------------------------------------------------------
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

    def test_category_valid(self, ioi_result):
        """
        Category must be one of the five labels returned by core.py.

        BUG HISTORY: The original test checked for {"strong", "good", "partial"},
        which are NEVER returned by GlassboxV2.  The correct set is:
            backup_mechanisms  <- suff > 0.9 and comp < 0.4
            faithful           <- suff > 0.7 and comp > 0.5
            weak               <- suff < 0.6 and comp < 0.5
            incomplete         <- suff < 0.5
            moderate           <- everything else
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
    n >= 5 is required for stable percentile confidence intervals.

    API contract for bootstrap_metrics():
        Input : prompts = List[Tuple[str, str, str]]  (raw prompt triples)
        Param : n_boot  (NOT n_bootstrap)
        Output: dict with keys "sufficiency", "comprehensiveness", "f1"
                each being a sub-dict: {"mean": float, "ci_lo": float,
                                        "ci_hi": float, "std": float, "n": int}
    """

    @pytest.mark.slow
    def test_bootstrap_returns_expected_keys(self, engine):
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
        """Lower CI bound must be <= mean <= upper CI bound."""
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
        fcas_result = engine.functional_circuit_alignment(heads_a, heads_b, top_k=3)
        for key in ("fcas", "null_mean", "null_std", "z_score", "pairs"):
            assert key in fcas_result, (
                f"functional_circuit_alignment missing key: {key!r}"
            )

    @pytest.mark.slow
    def test_fcas_identical_circuits_is_one(self, engine):
        """Comparing a circuit to itself must yield FCAS = 1.0."""
        r = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        heads = _heads_from_result(r)
        fcas_result = engine.functional_circuit_alignment(heads, heads, top_k=3)
        assert abs(fcas_result["fcas"] - 1.0) < 1e-6, (
            f"FCAS of identical circuits is {fcas_result['fcas']:.6f}, expected 1.0"
        )

    @pytest.mark.slow
    def test_fcas_range(self, engine):
        r1 = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        r2 = engine.analyze(FACT_PROMPT, FACT_CORRECT, FACT_INCORRECT)
        heads_a = _heads_from_result(r1)
        heads_b = _heads_from_result(r2)
        fcas_result = engine.functional_circuit_alignment(heads_a, heads_b, top_k=3)
        assert 0.0 <= fcas_result["fcas"] <= 1.0, (
            f"FCAS {fcas_result['fcas']:.4f} outside [0, 1]"
        )

    @pytest.mark.slow
    def test_z_score_is_finite(self, engine):
        r1 = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)
        r2 = engine.analyze(FACT_PROMPT, FACT_CORRECT, FACT_INCORRECT)
        heads_a = _heads_from_result(r1)
        heads_b = _heads_from_result(r2)
        fcas_result = engine.functional_circuit_alignment(heads_a, heads_b, top_k=3)
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
        result = engine.analyze("The cat sat on", "a", "the")
        assert "faithfulness" in result

    def test_output_has_no_nan(self, ioi_result):
        faith = ioi_result["faithfulness"]
        for k, v in faith.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"NaN detected in faithfulness[{k!r}]"
                assert not math.isinf(v), f"Inf detected in faithfulness[{k!r}]"

    def test_attributions_serializable(self, ioi_result):
        """Attributions are stored as str(key): float — must be JSON-serializable."""
        import json
        try:
            json.dumps(ioi_result["attributions"])
        except (TypeError, ValueError) as exc:
            pytest.fail(f"attributions dict is not JSON-serializable: {exc}")


# ===========================================================================
# 8. LOGIT LENS  (v2.2.0)
# ===========================================================================
@pytest.fixture(scope="module")
def logit_lens_result(engine, ioi_tokens):
    tokens_c, _, _, _ = ioi_tokens
    return engine.logit_lens(tokens_c, " Mary", " John")


class TestLogitLens:
    """Tests for logit_lens() — nostalgebraist 2020 extended with direct effects."""

    def test_returns_dict(self, logit_lens_result):
        assert isinstance(logit_lens_result, dict)

    def test_required_keys(self, logit_lens_result):
        for key in ("logit_diffs", "logit_shifts", "head_direct_effects",
                    "mlp_direct_effects", "target_token", "distractor_token"):
            assert key in logit_lens_result, f"logit_lens missing key: {key!r}"

    def test_logit_diffs_length(self, engine, logit_lens_result):
        """logit_diffs must have n_layers + 1 entries (embedding + after each block)."""
        expected = engine.n_layers + 1
        actual   = len(logit_lens_result["logit_diffs"])
        assert actual == expected, (
            f"logit_diffs length {actual} != expected {expected} (n_layers+1)"
        )

    def test_logit_shifts_length(self, engine, logit_lens_result):
        """logit_shifts must have exactly n_layers entries."""
        expected = engine.n_layers
        actual   = len(logit_lens_result["logit_shifts"])
        assert actual == expected, (
            f"logit_shifts length {actual} != expected {expected} (n_layers)"
        )

    def test_shift_equals_diff_delta(self, logit_lens_result):
        """Each logit_shift must equal logit_diffs[i+1] - logit_diffs[i]."""
        diffs  = logit_lens_result["logit_diffs"]
        shifts = logit_lens_result["logit_shifts"]
        for i, shift in enumerate(shifts):
            expected = diffs[i + 1] - diffs[i]
            assert abs(shift - expected) < 1e-4, (
                f"shift[{i}]={shift:.6f} != diffs[{i+1}]-diffs[{i}]={expected:.6f}"
            )

    def test_head_direct_effects_coverage(self, engine, logit_lens_result):
        """head_direct_effects must have an entry for every layer."""
        hde = logit_lens_result["head_direct_effects"]
        assert len(hde) == engine.n_layers, (
            f"head_direct_effects has {len(hde)} layers, expected {engine.n_layers}"
        )

    def test_head_direct_effects_per_head_count(self, engine, logit_lens_result):
        """Each layer in head_direct_effects must have n_heads values."""
        hde = logit_lens_result["head_direct_effects"]
        for l, effects in hde.items():
            assert len(effects) == engine.n_heads, (
                f"Layer {l}: {len(effects)} effects, expected {engine.n_heads} heads"
            )

    def test_mlp_effects_coverage(self, engine, logit_lens_result):
        """mlp_direct_effects must have an entry for every layer."""
        mde = logit_lens_result["mlp_direct_effects"]
        assert len(mde) == engine.n_layers

    def test_no_nan_or_inf(self, logit_lens_result):
        """All numeric outputs must be finite."""
        for val in logit_lens_result["logit_diffs"] + logit_lens_result["logit_shifts"]:
            assert math.isfinite(val), f"Non-finite value in logit_diffs/shifts: {val}"
        for l, effects in logit_lens_result["head_direct_effects"].items():
            for e in effects:
                assert math.isfinite(e), f"Non-finite head_direct_effect at layer {l}: {e}"

    def test_final_ld_positive(self, logit_lens_result):
        """For IOI on GPT-2, the final logit diff should be positive (Mary > John)."""
        final_ld = logit_lens_result["logit_diffs"][-1]
        assert final_ld > 0.0, (
            f"Final logit diff {final_ld:.4f} is not positive for IOI prompt"
        )

    def test_analyze_include_logit_lens(self, engine):
        """analyze(include_logit_lens=True) must include 'logit_lens' key."""
        result = engine.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT,
                                include_logit_lens=True)
        assert "logit_lens" in result, (
            "analyze(include_logit_lens=True) missing 'logit_lens' key"
        )
        assert "logit_diffs" in result["logit_lens"]


# ===========================================================================
# 9. EDGE ATTRIBUTION PATCHING  (v2.2.0, Syed et al. 2024)
# ===========================================================================
@pytest.fixture(scope="module")
def eap_result(engine, ioi_tokens):
    tokens_c, tokens_corr, t_tok, d_tok = ioi_tokens
    return engine.edge_attribution_patching(tokens_c, tokens_corr, t_tok, d_tok, top_k=20)


class TestEdgeAttributionPatching:
    """Tests for edge_attribution_patching() — EAP (Syed et al. 2024)."""

    def test_returns_dict(self, eap_result):
        assert isinstance(eap_result, dict)

    def test_required_keys(self, eap_result):
        for key in ("edge_scores", "top_edges", "n_edges", "clean_ld"):
            assert key in eap_result, f"EAP result missing key: {key!r}"

    def test_edge_scores_nonempty(self, eap_result):
        assert len(eap_result["edge_scores"]) > 0, "edge_scores is empty"

    def test_top_edges_length(self, eap_result):
        """top_edges must have at most top_k entries."""
        assert len(eap_result["top_edges"]) <= 20

    def test_top_edges_have_required_fields(self, eap_result):
        """Every edge record must have sender, receiver, score fields."""
        for edge in eap_result["top_edges"]:
            for field in ("sender", "receiver", "score"):
                assert field in edge, f"Edge missing field {field!r}: {edge}"

    def test_n_edges_consistent(self, eap_result):
        """n_edges must match the total number of scored edges."""
        assert eap_result["n_edges"] == len(eap_result["edge_scores"]), (
            "n_edges inconsistent with edge_scores length"
        )

    def test_clean_ld_positive(self, eap_result):
        """Clean logit diff must be positive for IOI prompt."""
        assert eap_result["clean_ld"] > 0.0, (
            f"clean_ld {eap_result['clean_ld']:.4f} should be positive for IOI"
        )

    def test_scores_are_finite(self, eap_result):
        """All edge scores must be finite."""
        for edge in eap_result["top_edges"]:
            assert math.isfinite(edge["score"]), (
                f"Non-finite score in edge: {edge}"
            )

    def test_scores_have_nonzero_spread(self, eap_result):
        """Edge scores must not all be identical — variation implies gradient flow."""
        scores = [e["score"] for e in eap_result["top_edges"]]
        if len(scores) > 1:
            assert max(scores) != min(scores), "All edge scores are identical"


# ===========================================================================
# 10. ATTRIBUTION STABILITY  (v2.2.0, novel metric)
# ===========================================================================
class TestAttributionStability:
    """Tests for attribution_stability() — novel Glassbox metric."""

    @pytest.fixture(scope="class")
    def stab_result(self, engine, ioi_tokens):
        tokens_c, _, t_tok, d_tok = ioi_tokens
        return engine.attribution_stability(
            tokens_c, t_tok, d_tok,
            n_corruptions=5,    # small n for CI speed
            replace_fraction=0.25,
            seed=42,
        )

    def test_returns_dict(self, stab_result):
        assert isinstance(stab_result, dict)

    def test_required_keys(self, stab_result):
        for key in ("mean_attributions", "std_attributions", "stability_scores",
                    "rank_consistency", "top_stable_heads", "n_corruptions"):
            assert key in stab_result, f"attribution_stability missing key: {key!r}"

    def test_stability_scores_in_range(self, stab_result):
        """Stability scores S = 1 - std/(|mean|+ε) can be negative but bounded above by 1."""
        for s in stab_result["stability_scores"]:
            assert s <= 1.0 + 1e-6, f"Stability score {s:.4f} exceeds 1.0"

    def test_rank_consistency_in_range(self, stab_result):
        """Kendall τ-b rank consistency must be in [-1, 1]."""
        tau = stab_result["rank_consistency"]
        assert -1.0 - 1e-6 <= tau <= 1.0 + 1e-6, (
            f"rank_consistency {tau:.4f} outside [-1, 1]"
        )

    def test_n_corruptions_matches_request(self, stab_result):
        assert stab_result["n_corruptions"] == 5

    def test_top_stable_heads_format(self, stab_result):
        """top_stable_heads must be a list of dicts with layer and head keys."""
        for h in stab_result["top_stable_heads"]:
            assert "layer"     in h, f"Missing 'layer' in top_stable_heads entry: {h}"
            assert "head"      in h, f"Missing 'head' in top_stable_heads entry: {h}"
            assert "stability" in h, f"Missing 'stability' in top_stable_heads entry: {h}"

    def test_no_nan_in_mean_attributions(self, stab_result):
        for v in stab_result["mean_attributions"]:
            assert math.isfinite(v), f"NaN/Inf in mean_attributions: {v}"


# ===========================================================================
# 11. TOKEN ATTRIBUTION  (v2.3.0, Simonyan et al. 2014)
# ===========================================================================
class TestTokenAttribution:
    """Tests for token_attribution() — gradient × embedding saliency."""

    @pytest.fixture(scope="class")
    def tok_attr_result(self, engine, ioi_tokens):
        tokens_c, _, t_tok, d_tok = ioi_tokens
        return engine.token_attribution(tokens_c, t_tok, d_tok)

    def test_returns_dict(self, tok_attr_result):
        assert isinstance(tok_attr_result, dict)

    def test_required_keys(self, tok_attr_result):
        for key in ("token_ids", "token_strs", "attributions",
                    "abs_attributions", "top_tokens"):
            assert key in tok_attr_result, f"token_attribution missing key: {key!r}"

    def test_length_consistent_with_tokens(self, tok_attr_result):
        """All per-token lists must have the same length."""
        n = len(tok_attr_result["token_ids"])
        assert len(tok_attr_result["token_strs"])      == n
        assert len(tok_attr_result["attributions"])    == n
        assert len(tok_attr_result["abs_attributions"]) == n

    def test_abs_equals_abs_of_attr(self, tok_attr_result):
        """abs_attributions must equal |attributions|."""
        for a, aa in zip(tok_attr_result["attributions"],
                         tok_attr_result["abs_attributions"]):
            assert abs(abs(a) - aa) < 1e-6, (
                f"abs_attributions mismatch: |{a:.6f}| != {aa:.6f}"
            )

    def test_scores_are_finite(self, tok_attr_result):
        for v in tok_attr_result["attributions"]:
            assert math.isfinite(v), f"Non-finite token attribution: {v}"

    def test_top_tokens_length(self, tok_attr_result):
        """top_tokens must return at most 5 entries."""
        assert len(tok_attr_result["top_tokens"]) <= 5

    def test_top_tokens_have_required_fields(self, tok_attr_result):
        for t in tok_attr_result["top_tokens"]:
            for field in ("rank", "token_str", "attribution", "position"):
                assert field in t, f"top_tokens entry missing field {field!r}: {t}"

    def test_top_tokens_sorted_by_abs(self, tok_attr_result):
        """top_tokens must be sorted descending by |attribution|."""
        abs_scores = [abs(t["attribution"]) for t in tok_attr_result["top_tokens"]]
        assert abs_scores == sorted(abs_scores, reverse=True), (
            "top_tokens not sorted by |attribution| descending"
        )


# ===========================================================================
# 12. ATTENTION PATTERNS  (v2.3.0, Elhage et al. 2021 / Olsson et al. 2022)
# ===========================================================================
class TestAttentionPatterns:
    """Tests for attention_patterns() — attention matrices + entropy + head typing."""

    VALID_HEAD_TYPES = {"induction_candidate", "previous_token", "focused",
                        "uniform", "self_attn", "mixed"}

    @pytest.fixture(scope="class")
    def attn_result(self, engine, ioi_tokens):
        tokens_c, _, _, _ = ioi_tokens
        # Explicitly request known IOI heads
        return engine.attention_patterns(tokens_c, heads=[(9, 9), (10, 0), (5, 5)])

    def test_returns_dict(self, attn_result):
        assert isinstance(attn_result, dict)

    def test_required_keys(self, attn_result):
        for key in ("heads", "patterns", "entropy", "last_tok_attn",
                    "head_types", "token_strs"):
            assert key in attn_result, f"attention_patterns missing key: {key!r}"

    def test_heads_count(self, attn_result):
        """Requesting 3 heads must return 3 entries."""
        assert len(attn_result["heads"]) == 3

    def test_patterns_are_square(self, attn_result):
        """Each attention matrix must be [seq, seq]."""
        for label, A in attn_result["patterns"].items():
            assert A.ndim == 2, f"Pattern for {label} is not 2D"
            assert A.shape[0] == A.shape[1], (
                f"Pattern for {label} is not square: {A.shape}"
            )

    def test_attention_rows_sum_to_one(self, attn_result):
        """Attention weights must sum to 1 along the source axis."""
        import numpy as np
        for label, A in attn_result["patterns"].items():
            row_sums = A.sum(axis=-1)
            assert np.allclose(row_sums, 1.0, atol=1e-4), (
                f"Attention rows for {label} don't sum to 1: {row_sums}"
            )

    def test_entropy_non_negative(self, attn_result):
        """Entropy must be non-negative."""
        for label, ent in attn_result["entropy"].items():
            assert ent >= 0.0, f"Negative entropy for {label}: {ent}"

    def test_head_types_valid(self, attn_result):
        """All head types must be from the valid set."""
        for label, htype in attn_result["head_types"].items():
            assert htype in self.VALID_HEAD_TYPES, (
                f"Head {label}: invalid type {htype!r}. "
                f"Valid: {self.VALID_HEAD_TYPES}"
            )

    def test_auto_select_returns_top_k(self, engine, ioi_tokens):
        """attention_patterns(heads=None) must auto-select and return top_k heads."""
        tokens_c, _, _, _ = ioi_tokens
        result = engine.attention_patterns(tokens_c, heads=None, top_k=5)
        assert len(result["heads"]) <= 5


# ===========================================================================
# 13. HEAD COMPOSITION ANALYSIS  (v2.3.0, Elhage et al. 2021)
# ===========================================================================
class TestHeadCompositionAnalyzer:
    """Tests for HeadCompositionAnalyzer — QK / OV virtual-weight composition scores."""

    @pytest.fixture(scope="class")
    def comp(self, engine):
        from glassbox.composition import HeadCompositionAnalyzer
        return HeadCompositionAnalyzer(engine.model)

    # Individual score methods
    def test_q_composition_causally_invalid_returns_zero(self, comp):
        """Receiver at an earlier layer than sender must return 0.0."""
        score = comp.q_composition_score(9, 9, 5, 5)  # receiver layer 5 < sender 9
        assert score == 0.0, f"Expected 0.0 for causally invalid pair, got {score}"

    def test_k_composition_causally_invalid_returns_zero(self, comp):
        score = comp.k_composition_score(9, 9, 5, 5)
        assert score == 0.0

    def test_v_composition_causally_invalid_returns_zero(self, comp):
        score = comp.v_composition_score(9, 9, 5, 5)
        assert score == 0.0

    def test_q_composition_valid_pair_nonnegative(self, comp):
        score = comp.q_composition_score(5, 5, 9, 9)
        assert score >= 0.0, f"Q-composition score must be non-negative, got {score}"

    def test_k_composition_valid_pair_nonnegative(self, comp):
        score = comp.k_composition_score(5, 5, 9, 9)
        assert score >= 0.0

    def test_v_composition_valid_pair_nonnegative(self, comp):
        score = comp.v_composition_score(5, 5, 9, 9)
        assert score >= 0.0

    def test_scores_are_finite(self, comp):
        for score_fn in (comp.q_composition_score,
                         comp.k_composition_score,
                         comp.v_composition_score):
            s = score_fn(5, 5, 9, 9)
            assert math.isfinite(s), f"Non-finite composition score: {s}"

    # Composition matrix
    def test_composition_matrix_shape(self, comp):
        circuit = [(5, 5), (9, 9)]
        result  = comp.composition_matrix(circuit, circuit, kind="q")
        mat     = result["matrix"]
        assert mat.shape == (2, 2), f"Unexpected matrix shape: {mat.shape}"

    def test_composition_matrix_labels(self, comp):
        circuit = [(5, 5), (9, 9)]
        result  = comp.composition_matrix(circuit, circuit, kind="q")
        assert result["senders"]   == ["L05H05", "L09H09"]
        assert result["receivers"] == ["L05H05", "L09H09"]

    def test_composition_matrix_kind_validation(self, comp):
        with pytest.raises(ValueError, match="kind must be"):
            comp.composition_matrix([(5, 5)], [(9, 9)], kind="xyz")

    def test_full_circuit_composition_returns_significant_edges(self, comp):
        circuit = [(5, 5), (7, 3), (9, 9), (9, 6)]
        result  = comp.full_circuit_composition(circuit, kind="q", min_score=0.0)
        assert "significant_edges" in result
        assert "matrix"            in result
        assert "head_labels"       in result

    def test_all_composition_scores_has_three_kinds(self, comp):
        circuit = [(5, 5), (9, 9)]
        result  = comp.all_composition_scores(circuit, min_score=0.0)
        for key in ("q", "k", "v", "combined_edges", "head_labels"):
            assert key in result, f"all_composition_scores missing key: {key!r}"


# ===========================================================================
# TestNegativeInputs — validate error handling and edge cases
# ===========================================================================

class TestNegativeInputs:
    """
    Negative-path tests: verify Glassbox raises or recovers gracefully on
    bad inputs rather than producing silent wrong answers.
    """

    def test_none_prompt_raises(self, engine):
        """None is not a valid prompt — should raise TypeError or AttributeError."""
        with pytest.raises((TypeError, AttributeError, Exception)):
            engine.analyze(None, " Mary", " John")

    def test_integer_prompt_raises(self, engine):
        """Integer is not a valid prompt."""
        with pytest.raises((TypeError, AttributeError, Exception)):
            engine.analyze(42, " Mary", " John")

    def test_identical_tokens_zero_ld(self, engine):
        """Same target and distractor → LD ≈ 0, circuit may be empty, should not crash."""
        result = engine.analyze(
            "When Mary and John went to the store, John gave a drink to",
            " John", " John",
        )
        assert "faithfulness" in result
        # With LD = 0, sufficiency must be 0.0 (no logit difference to explain)
        assert result["faithfulness"]["sufficiency"] == 0.0

    def test_model_metadata_present(self, engine):
        """Every analyze() result must include model_metadata for reproducibility."""
        result = engine.analyze(
            "When Mary and John went to the store, John gave a drink to",
            " Mary", " John",
        )
        meta = result.get("model_metadata")
        assert meta is not None, "model_metadata missing from analyze() result"
        for key in ("model_name", "n_layers", "n_heads", "d_model", "glassbox_version"):
            assert key in meta, f"model_metadata missing key: {key!r}"
        assert meta["n_layers"] > 0
        assert meta["n_heads"]  > 0
        assert meta["d_model"]  > 0

    def test_batch_analyze_returns_list(self, engine):
        """batch_analyze() must return a list of the same length as the input."""
        prompts = [
            ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
            ("The capital of France is",                                   " Paris", " London"),
        ]
        results = engine.batch_analyze(prompts, show_progress=False)
        assert isinstance(results, list)
        assert len(results) == len(prompts)

    def test_batch_analyze_skips_errors(self, engine):
        """batch_analyze(skip_errors=True) should record error dicts, not crash."""
        prompts = [
            ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
            # Deliberately bad — None will cause an error
            (None, " Mary", " John"),
        ]
        results = engine.batch_analyze(prompts, skip_errors=True, show_progress=False)
        assert len(results) == 2
        # First prompt should succeed
        assert "faithfulness" in results[0]
        # Second prompt should have an 'error' key, not crash
        assert "error" in results[1]

    def test_normalize_token_int_passthrough(self):
        """normalize_token with int input must return that int unchanged."""
        from glassbox.utils import normalize_token

        class _FakeModel:
            def to_single_token(self, s):
                raise RuntimeError("should not be called")

        assert normalize_token(_FakeModel(), 42) == 42

    def test_normalize_token_type_error(self):
        """normalize_token with a list input must raise TypeError."""
        from glassbox.utils import normalize_token

        class _FakeModel:
            pass

        with pytest.raises(TypeError):
            normalize_token(_FakeModel(), [1, 2, 3])

    def test_format_head_label(self):
        """format_head_label must produce zero-padded L/H strings."""
        from glassbox.utils import format_head_label, parse_head_label
        assert format_head_label(0, 0)   == "L00H00"
        assert format_head_label(9, 9)   == "L09H09"
        assert format_head_label(11, 11) == "L11H11"
        # Round-trip
        for l, h in [(0, 0), (9, 9), (11, 11)]:
            assert parse_head_label(format_head_label(l, h)) == (l, h)

    def test_estimate_memory(self):
        """estimate_forward_pass_memory_mb must return a positive float."""
        from glassbox.utils import estimate_forward_pass_memory_mb
        mb = estimate_forward_pass_memory_mb(
            n_layers=12, n_heads=12, d_model=768, seq_len=20
        )
        assert mb > 0.0
        assert isinstance(mb, float)

    def test_stable_api_decorator(self):
        """@stable_api must set __glassbox_stable__ = True without changing behaviour."""
        from glassbox.utils import stable_api

        @stable_api
        def my_fn(x):
            return x * 2

        assert my_fn(3) == 6
        assert getattr(my_fn, "__glassbox_stable__", False) is True

    def test_valid_head_types_set(self):
        """VALID_HEAD_TYPES exported from glassbox must be non-empty."""
        from glassbox import VALID_HEAD_TYPES
        assert len(VALID_HEAD_TYPES) > 0
        assert "induction_candidate" in VALID_HEAD_TYPES

    def test_faithfulness_categories_set(self):
        """FAITHFULNESS_CATEGORIES must contain all expected strings."""
        from glassbox import FAITHFULNESS_CATEGORIES
        assert "faithful" in FAITHFULNESS_CATEGORIES
        assert "backup_mechanisms" in FAITHFULNESS_CATEGORIES


# ===========================================================================
# TestCLI — smoke tests for glassbox-ai CLI subcommands
# ===========================================================================

class TestCLI:
    """
    CLI smoke tests — verifies the CLI entry point works end-to-end.
    Uses subprocess to invoke the same way a user would.
    These tests are marked slow because they each load GPT-2.
    """

    def test_help_exits_zero(self):
        """glassbox-ai --help must exit 0 and print usage information."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "glassbox.cli", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "glassbox" in result.stdout.lower()

    def test_doctor_exits_zero(self):
        """glassbox-ai doctor must exit 0 when all required deps are present."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "glassbox.cli", "doctor"],
            capture_output=True, text=True, timeout=30,
        )
        # doctor should succeed in this environment (all deps installed)
        assert result.returncode == 0
        assert "glassbox" in result.stdout.lower() or "glassbox" in result.stderr.lower()

    def test_version_exits_zero(self):
        """glassbox-ai version must exit 0 and print a version string."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "glassbox.cli", "version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        # Output must contain the version string (digits separated by dots)
        import re
        assert re.search(r"\d+\.\d+", result.stdout), (
            f"No version number found in CLI version output: {result.stdout!r}"
        )

    def test_analyze_subcommand_help(self):
        """glassbox-ai analyze --help must not crash."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "glassbox.cli", "analyze", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
