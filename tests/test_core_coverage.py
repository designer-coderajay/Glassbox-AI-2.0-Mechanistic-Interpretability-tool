# SPDX-License-Identifier: MIT
"""
tests/test_core_coverage.py — model-based coverage for core.py methods that the
main engine suite doesn't exercise: model_info/_fmt, the integrated-gradients
attribution path, stability_suite, batch_analyze, and token_attribution.

These require a real transformer_lens model (skipped cleanly otherwise), so they
run in the full CI job, not offline. Assertions are intentionally shape-level —
the goal is to drive the untested code paths, not re-verify numeric results.
"""

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def gb():
    import sys
    _tl = sys.modules.get("transformer_lens")
    if _tl is not None and not hasattr(_tl, "__file__"):
        pytest.skip("transformer_lens is a test stub, not the real package")
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer

    from glassbox import GlassboxV2
    model = HookedTransformer.from_pretrained("gpt2")
    return GlassboxV2(model)


IOI_PROMPT = "When Mary and John went to the store, John gave a drink to"
IOI_CORRECT = " Mary"
IOI_INCORRECT = " John"

# Semantically-equivalent IOI paraphrases for stability_suite.
VARIANTS = [
    ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
    ("After Alice and Bob arrived at the party, Bob handed the keys to", " Alice", " Bob"),
    ("When Sarah and Tom left the office, Tom passed the report to", " Sarah", " Tom"),
    ("Once Emma and Jack sat down, Jack gave the pen to", " Emma", " Jack"),
    ("While Lisa and Mike waited outside, Mike offered the seat to", " Lisa", " Mike"),
]


class TestModelInfo:
    def test_model_info_returns_dict(self, gb):
        info = gb.model_info()
        assert isinstance(info, dict)
        assert info  # non-empty


class TestIntegratedGradients:
    def test_analyze_integrated_gradients(self, gb):
        # Drives the integrated_gradients branch of attribution_patching.
        result = gb.analyze(
            IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT,
            method="integrated_gradients", n_steps=3,
        )
        assert isinstance(result, dict)
        assert "faithfulness" in result
        assert "circuit" in result


class TestStabilitySuite:
    def test_stability_suite_keys(self, gb):
        out = gb.stability_suite(VARIANTS, seed=42)
        for key in ("jaccard_mean", "jaccard_std", "stability_rate",
                    "consensus_circuit", "per_variant", "n_pairs"):
            assert key in out, f"missing key: {key}"
        assert len(out["per_variant"]) == len(VARIANTS)

    def test_stability_suite_too_few_raises(self, gb):
        with pytest.raises(ValueError):
            gb.stability_suite(VARIANTS[:1])


class TestBatchAndToken:
    def test_batch_analyze(self, gb):
        results = gb.batch_analyze(
            [(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)],
            show_progress=False,
        )
        assert isinstance(results, list) and len(results) == 1

    def test_token_attribution(self, gb):
        tokens = gb.model.to_tokens(IOI_PROMPT)
        tgt = gb.model.to_single_token(IOI_CORRECT)
        dis = gb.model.to_single_token(IOI_INCORRECT)
        out = gb.token_attribution(tokens, tgt, dis)
        assert isinstance(out, dict)
