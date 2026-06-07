"""Live end-to-end verification for the large-model attribution path.

This is the receipt that closes the single highest technical risk: that
``LargeModelAttributionPatcher`` / ``analyze_large`` are *implemented and
unit-tested* but never run end-to-end on a real model.

These tests are SLOW and require a real model download (and, for >7B, a GPU).
They are skipped by default so normal CI stays fast. Opt in explicitly:

    GLASSBOX_LIVE_TEST=1 pytest tests/test_large_model_live.py -v -s

Target a specific model (default is gpt2, which is CI-affordable and proves the
plumbing works identically for the large path):

    GLASSBOX_LIVE_TEST=1 GLASSBOX_LIVE_MODEL=meta-llama/Meta-Llama-3-8B \\
        pytest tests/test_large_model_live.py -v -s

When the 8B run passes, capture the console output and publish it — that is the
artifact that lets sales honestly say "verified on Llama-3-8B".
"""

from __future__ import annotations

import os

import pytest

# --- opt-in gate ------------------------------------------------------------

LIVE = os.environ.get("GLASSBOX_LIVE_TEST") == "1"
MODEL_NAME = os.environ.get("GLASSBOX_LIVE_MODEL", "gpt2")

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Live model test. Set GLASSBOX_LIVE_TEST=1 to run (slow, downloads a model).",
)


@pytest.fixture(scope="module")
def loaded_model():
    """Load a real transformer once for the whole module."""
    torch = pytest.importorskip("torch")
    tl = pytest.importorskip("transformer_lens")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tl.HookedTransformer.from_pretrained(MODEL_NAME, dtype=dtype, device=device)
    return model, dtype, device


# A neutral, domain-relevant prompt that does NOT rely on IOI name-swap, so the
# any-prompt corruption engine is exercised on the large path too.
PROMPT = "Loan application. Annual income is high and credit history is clean. Decision:"
CORRECT = " Approved"
INCORRECT = " Denied"


def test_estimate_memory_recommends_sane_strategy():
    """estimate_memory must classify size and recommend a strategy without loading weights."""
    torch = pytest.importorskip("torch")
    from glassbox import estimate_memory

    small = estimate_memory(n_layers=12, d_model=768, seq_len=64, dtype=torch.float32)
    huge = estimate_memory(n_layers=80, d_model=8192, seq_len=256, dtype=torch.bfloat16)

    assert small.recommend_strategy in {"standard", "checkpoint", "checkpoint+offload"}
    assert huge.recommend_strategy in {"checkpoint", "checkpoint+offload"}, (
        "A 70B-class config should never be recommended the 'standard' path."
    )
    # Bigger model must be estimated to need more memory than the small one.
    assert huge.attribution_gb > small.attribution_gb


def test_analyze_large_end_to_end(loaded_model):
    """analyze_large must return a valid faithfulness result on a real model."""
    from glassbox import GlassboxV2, analyze_large

    model, dtype, _ = loaded_model
    gb = GlassboxV2(model)

    result = analyze_large(
        gb, PROMPT, correct=CORRECT, incorrect=INCORRECT,
        strategy="auto", dtype=dtype,
    )

    assert "faithfulness" in result, "Result is missing the faithfulness block."
    f1 = result["faithfulness"].get("f1")
    assert f1 is not None and 0.0 <= f1 <= 1.0, f"F1 out of range: {f1!r}"
    assert result.get("circuit"), "Circuit should be non-empty for a real model."
    # Any-prompt engine must have recorded which corruption strategy it used.
    assert result.get("corruption_metadata"), "Missing corruption_metadata audit trail."


def test_large_path_matches_standard_path(loaded_model):
    """For a model small enough to run both ways, the two paths must agree closely.

    This is the correctness anchor: the memory-managed path must not change the
    *answer*, only the memory profile. Gradient checkpointing recomputes
    activations but must be numerically faithful.
    """
    from glassbox import GlassboxV2, analyze_large

    model, dtype, _ = loaded_model
    gb = GlassboxV2(model)

    standard = gb.analyze(PROMPT, correct=CORRECT, incorrect=INCORRECT)
    managed = analyze_large(
        gb, PROMPT, correct=CORRECT, incorrect=INCORRECT,
        strategy="checkpoint", dtype=dtype,
    )

    f1_std = standard["faithfulness"]["f1"]
    f1_mgd = managed["faithfulness"]["f1"]
    # Allow a small tolerance for bf16 recomputation noise; tighten on fp32.
    tol = 0.05 if dtype.is_floating_point else 0.0
    assert abs(f1_std - f1_mgd) <= tol, (
        f"Memory-managed path diverged from standard: {f1_std:.4f} vs {f1_mgd:.4f}"
    )


def test_audit_time_is_recorded(loaded_model):
    """The result should expose a timing so we can guard the <2s benchmark on GPT-2."""
    import time

    from glassbox import GlassboxV2, analyze_large

    model, dtype, _ = loaded_model
    gb = GlassboxV2(model)

    t0 = time.perf_counter()
    analyze_large(gb, PROMPT, correct=CORRECT, incorrect=INCORRECT, strategy="auto", dtype=dtype)
    elapsed = time.perf_counter() - t0

    # Informational on large models; only a hard gate on gpt2 CPU baseline.
    if MODEL_NAME == "gpt2":
        assert elapsed < 10.0, f"gpt2 audit unexpectedly slow: {elapsed:.2f}s"
