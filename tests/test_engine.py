import pytest, time, torch
from transformer_lens import HookedTransformer

@pytest.fixture(scope="session")
def gb():
    from glassbox import GlassboxV2
    model = HookedTransformer.from_pretrained("gpt2")
    return GlassboxV2(model)

@pytest.fixture(scope="session")
def ioi_tokens(gb):
    clean  = gb.model.to_tokens(
        "When Mary and John went to the store, John gave a gift to")
    corr   = gb.model.to_tokens(
        "When Alice and Bob went to the store, Bob gave a gift to")
    target = gb.model.to_single_token(" Mary")
    dist   = gb.model.to_single_token(" John")
    return clean, corr, target, dist

@pytest.fixture(scope="session")
def ioi_result(gb):
    return gb.analyze(
        prompt="When Mary and John went to the store, John gave a gift to",
        correct=" Mary",
        incorrect=" John"
    )

# ── Attribution patching ──────────────────────────────────────────

def test_attribution_returns_scores(gb, ioi_tokens):
    clean, corr, target, dist = ioi_tokens
    scores, clean_ld = gb.attribution_patching(clean, corr, target, dist)
    assert len(scores) > 0, "attribution_patching must return non-empty scores dict"
    assert isinstance(clean_ld, float), "Second return value should be clean_ld float"

def test_attribution_scores_nonzero(gb, ioi_tokens):
    clean, corr, target, dist = ioi_tokens
    scores, _ = gb.attribution_patching(clean, corr, target, dist)
    assert any(abs(v) > 0.01 for v in scores.values()), (
        "All attribution scores near zero -- patching not working")

def test_o3_completes_under_60s(gb, ioi_tokens):
    clean, corr, target, dist = ioi_tokens
    t0 = time.time()
    gb.attribution_patching(clean, corr, target, dist)
    assert time.time() - t0 < 60.0

# ── Circuit discovery (MFC) ───────────────────────────────────────
# circuit is a list of (layer, head) tuples

def test_mfc_circuit_nonempty(ioi_result):
    assert len(ioi_result["circuit"]) >= 1, "MFC must return at least 1 head"

def test_mfc_top_ioi_head(ioi_result):
    circuit = ioi_result["circuit"]
    # L9H9 is the dominant IOI head (Wang et al. 2022) — must be in the MFC
    assert (9, 9) in circuit, (
        f"L9H9 (Wang et al. 2022 ground truth) not in MFC circuit: {circuit}")

def test_circuit_head_keys_are_tuples(ioi_result):
    for head in ioi_result["circuit"]:
        assert isinstance(head, tuple) and len(head) == 2, (
            f"Circuit entry should be (layer, head) tuple, got {head}")

# ── Faithfulness metrics ──────────────────────────────────────────
# actual keys: 'sufficiency', 'comprehensiveness', 'f1', 'category'

def test_analyze_returns_required_keys(ioi_result):
    assert "circuit"      in ioi_result
    assert "faithfulness" in ioi_result
    for k in ("sufficiency", "comprehensiveness", "f1", "category"):
        assert k in ioi_result["faithfulness"], (
            f"Missing key '{k}' in faithfulness: {ioi_result['faithfulness']}")

def test_sufficiency_in_range(ioi_result):
    suff = ioi_result["faithfulness"]["sufficiency"]
    assert 0.0 <= suff <= 1.0, f"Sufficiency {suff:.4f} outside [0, 1]"

def test_comprehensiveness_in_range(ioi_result):
    comp = ioi_result["faithfulness"]["comprehensiveness"]
    assert 0.0 <= comp <= 1.0, f"Comprehensiveness {comp:.4f} outside [0, 1]"

def test_ioi_sufficiency_high(ioi_result):
    suff = ioi_result["faithfulness"]["sufficiency"]
    assert suff >= 0.8, f"IOI sufficiency {suff:.1%} below 80%"

def test_f1_consistent(ioi_result):
    faith = ioi_result["faithfulness"]
    suff = faith["sufficiency"]
    comp = faith["comprehensiveness"]
    if suff + comp > 0:
        expected_f1 = 2 * suff * comp / (suff + comp)
        assert abs(faith["f1"] - expected_f1) < 0.01, (
            f"F1 {faith['f1']:.4f} inconsistent with "
            f"suff={suff:.4f} comp={comp:.4f}")

def test_faithfulness_category_valid(ioi_result):
    cat = ioi_result["faithfulness"]["category"]
    assert cat in {"faithful", "moderate", "weak", "incomplete"}, (
        f"Unknown category: {cat!r}")
