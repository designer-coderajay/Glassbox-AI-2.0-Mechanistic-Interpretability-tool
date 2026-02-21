import time
import pytest
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

PROMPT   = "When Mary and John went to the store, John gave a bottle to"
CORR_P   = "When Mary and John went to the store, Mary gave a bottle to"
TARGET   = "Mary"
DIST     = "John"

@pytest.fixture(scope="session")
def gb():
    model = HookedTransformer.from_pretrained("gpt2")
    return GlassboxV2(model)

@pytest.fixture(scope="session")
def tokens(gb):
    """Pre-tokenized inputs for attribution_patching tests."""
    clean_tok = gb.model.to_tokens(PROMPT)
    corr_tok  = gb.model.to_tokens(CORR_P)
    t_tok     = gb.model.to_single_token(TARGET)
    d_tok     = gb.model.to_single_token(DIST)
    return clean_tok, corr_tok, t_tok, d_tok

@pytest.fixture(scope="session")
def ioi_result(gb):
    # analyze() takes a single prompt string, not a list
    return gb.analyze(PROMPT, TARGET, DIST)

# ── attribution_patching ──────────────────────────────────────────────
def test_attribution_returns_scores(gb, tokens):
    clean_tok, corr_tok, t_tok, d_tok = tokens
    scores, clean_ld = gb.attribution_patching(clean_tok, corr_tok, t_tok, d_tok)
    assert isinstance(scores, dict)

def test_attribution_scores_nonzero(gb, tokens):
    clean_tok, corr_tok, t_tok, d_tok = tokens
    scores, _ = gb.attribution_patching(clean_tok, corr_tok, t_tok, d_tok)
    assert any(abs(v) > 1e-6 for v in scores.values())

def test_o3_completes_under_60s(gb, tokens):
    clean_tok, corr_tok, t_tok, d_tok = tokens
    t0 = time.time()
    gb.attribution_patching(clean_tok, corr_tok, t_tok, d_tok)
    assert time.time() - t0 < 60

# ── analyze / minimum_faithful_circuit ───────────────────────────────
def test_mfc_circuit_nonempty(ioi_result):
    assert len(ioi_result["circuit"]) > 0

def test_mfc_top_ioi_head(ioi_result):
    assert (9, 9) in ioi_result["circuit"]

def test_circuit_entries_are_tuples(ioi_result):
    for entry in ioi_result["circuit"]:
        assert isinstance(entry, tuple) and len(entry) == 2

def test_required_keys_present(ioi_result):
    assert "circuit" in ioi_result
    assert "faithfulness" in ioi_result

def test_sufficiency_in_range(ioi_result):
    v = ioi_result["faithfulness"]["sufficiency"]
    assert 0.0 <= v <= 1.0

def test_comprehensiveness_in_range(ioi_result):
    v = ioi_result["faithfulness"]["comprehensiveness"]
    assert 0.0 <= v <= 1.0

def test_ioi_sufficiency_high(ioi_result):
    assert ioi_result["faithfulness"]["sufficiency"] > 0.5

def test_f1_consistent(ioi_result):
    f = ioi_result["faithfulness"]
    s, c = f["sufficiency"], f["comprehensiveness"]
    expected = 2 * s * c / (s + c + 1e-9)
    assert abs(f["f1"] - expected) < 1e-3

def test_category_valid(ioi_result):
    assert ioi_result["faithfulness"]["category"] in {
        "strong", "good", "moderate", "partial", "weak"
    }
