import pytest, time
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

def test_attribution_returns_scores(gb, ioi_tokens):
    clean, corr, target, dist = ioi_tokens
    scores, clean_ld = gb.attribution_patching(clean, corr, target, dist)
    assert len(scores) > 0
    assert isinstance(clean_ld, float)

def test_attribution_scores_nonzero(gb, ioi_tokens):
    clean, corr, target, dist = ioi_tokens
    scores, _ = gb.attribution_patching(clean, corr, target, dist)
    assert any(abs(v) > 0.01 for v in scores.values())

def test_o3_completes_under_60s(gb, ioi_tokens):
    clean, corr, target, dist = ioi_tokens
    t0 = time.time()
    gb.attribution_patching(clean, corr, target, dist)
    assert time.time() - t0 < 60.0

def test_mfc_circuit_nonempty(ioi_result):
    assert len(ioi_result["circuit"]) >= 1

def test_mfc_top_ioi_head(ioi_result):
    assert (9, 9) in ioi_result["circuit"], (
        f"L9H9 missing from circuit: {ioi_result['circuit']}")

def test_circuit_entries_are_tuples(ioi_result):
    for head in ioi_result["circuit"]:
        assert isinstance(head, tuple) and len(head) == 2

def test_required_keys_present(ioi_result):
    assert "circuit" in ioi_result
    assert "faithfulness" in ioi_result
    for k in ("sufficiency", "comprehensiveness", "f1", "category"):
        assert k in ioi_result["faithfulness"], (
            f"Missing '{k}'. Got: {list(ioi_result['faithfulness'].keys())}")

def test_sufficiency_in_range(ioi_result):
    v = ioi_result["faithfulness"]["sufficiency"]
    assert 0.0 <= v <= 1.0

def test_comprehensiveness_in_range(ioi_result):
    v = ioi_result["faithfulness"]["comprehensiveness"]
    assert 0.0 <= v <= 1.0

def test_ioi_sufficiency_high(ioi_result):
    v = ioi_result["faithfulness"]["sufficiency"]
    assert v >= 0.8, f"IOI sufficiency {v:.1%} below 80%"

def test_f1_consistent(ioi_result):
    f = ioi_result["faithfulness"]
    s, c = f["sufficiency"], f["comprehensiveness"]
    if s + c > 0:
        assert abs(f["f1"] - 2*s*c/(s+c)) < 0.01

def test_category_valid(ioi_result):
    assert ioi_result["faithfulness"]["category"] in {
        "faithful", "moderate", "weak", "incomplete"}
