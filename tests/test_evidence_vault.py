"""Tests for glassbox/evidence_vault.py — Annex IV evidence vault builder."""
import json

from glassbox.evidence_vault import (
    AnnexIVEvidenceVault,
    VaultEntry,
    build_annex_iv_vault,
)

# A realistic GlassboxV2.analyze() result (GPT-2 IOI numbers).
GB = {
    "faithfulness": {"sufficiency": 1.0, "comprehensiveness": 0.22, "f1": 0.64},
    "n_heads": 4,
    "circuit": {(9, 6): 0.584, (9, 9): 0.431, (10, 0): 0.312, (3, 0): 0.067},
}


# ── VaultEntry ─────────────────────────────────────────────────────────────
def test_vault_entry_to_dict():
    e = VaultEntry(
        section="§4", article_refs=["Article 9"], title="t",
        description="d", evidence_type="faithfulness",
        metric_name="f1", metric_value=0.64, threshold=0.65, passed=False,
    )
    d = e.to_dict()
    assert d["section"] == "§4"
    assert d["passed"] is False
    assert d["metric_value"] == 0.64
    assert "timestamp_utc" in d


# ── construction + baseline build ──────────────────────────────────────────
def test_empty_vault_has_baseline_entries():
    v = AnnexIVEvidenceVault(model_name="gpt2", provider="Acme").build_vault()
    d = v.to_dict()
    assert d["model_name"] == "gpt2"
    assert d["provider"] == "Acme"
    assert d["n_entries"] >= 3  # general description + standards + conformity
    assert d["n_entries"] == len(v.entries)
    assert isinstance(d["sections_covered"], list)


def test_build_vault_clears_between_runs():
    v = AnnexIVEvidenceVault()
    v.build_vault(gb_result=GB)
    first = len(v.entries)
    v.build_vault(gb_result=GB)  # should reset, not accumulate
    assert len(v.entries) == first


# ── gb_result population ───────────────────────────────────────────────────
def test_build_from_gb_result_metrics():
    v = AnnexIVEvidenceVault().build_vault(gb_result=GB)
    by_metric = {e.metric_name: e for e in v.entries if e.metric_name}
    assert by_metric["sufficiency"].passed is True       # 1.00 >= 0.70
    assert by_metric["comprehensiveness"].passed is False  # 0.22 < 0.60
    assert by_metric["f1"].passed is False                # 0.64 < 0.65
    assert by_metric["n_heads"].metric_value == 4.0
    # top-heads circuit entry is present
    assert any("circuit attention heads" in e.title.lower() for e in v.entries)


def test_compliance_summary_noncompliant():
    s = AnnexIVEvidenceVault().build_vault(gb_result=GB).to_dict()["compliance_summary"]
    assert s["overall_status"] == "NON-COMPLIANT"  # only 1 of 3 thresholds pass
    assert 0.0 <= s["pass_rate"] <= 1.0
    assert s["n_failed"] >= 2


def test_compliance_summary_compliant():
    good = {
        "faithfulness": {"sufficiency": 1.0, "comprehensiveness": 0.85, "f1": 0.92},
        "n_heads": 4, "circuit": {(9, 6): 0.5},
    }
    s = AnnexIVEvidenceVault().build_vault(gb_result=good).to_dict()["compliance_summary"]
    assert s["overall_status"] == "COMPLIANT"
    assert s["pass_rate"] == 1.0


# ── other input channels ───────────────────────────────────────────────────
def test_stability_entries_skip_non_numeric():
    v = AnnexIVEvidenceVault().build_vault(
        stability_result={"jaccard": 0.9, "rank_corr": 0.8, "note": "ignored"}
    )
    stab = [e for e in v.entries if e.evidence_type == "stability"]
    assert len(stab) == 2  # the string value is skipped


def test_custom_entries_appended():
    ce = VaultEntry(section="§9", article_refs=["Article 72"], title="Custom item",
                    description="d", evidence_type="general")
    v = AnnexIVEvidenceVault().build_vault(custom_entries=[ce])
    assert any(e.title == "Custom item" for e in v.entries)


# ── serialisation ──────────────────────────────────────────────────────────
def test_to_json_parses():
    parsed = json.loads(AnnexIVEvidenceVault().build_vault(gb_result=GB).to_json())
    assert parsed["n_entries"] > 0
    assert "compliance_summary" in parsed
    assert "entries" in parsed


def test_to_html_contains_status():
    html = AnnexIVEvidenceVault().build_vault(gb_result=GB).to_html()
    assert "<" in html
    assert any(s in html for s in ("COMPLIANT", "MARGINAL", "NON-COMPLIANT"))


def test_articles_covered_collected():
    arts = AnnexIVEvidenceVault().build_vault(gb_result=GB).to_dict()["articles_covered"]
    assert any("Article 15" in a for a in arts)


# ── static helper ──────────────────────────────────────────────────────────
def test_truncate_circuit_stringifies_keys_for_json_safety():
    big = {(i, 0): float(i) for i in range(30)}
    out = AnnexIVEvidenceVault._truncate_circuit(big, max_items=20)
    assert len(out) == 20
    assert all(isinstance(k, str) for k in out)  # JSON-serialisable keys
    # small circuits are also stringified now (the bug fix)
    assert AnnexIVEvidenceVault._truncate_circuit({(1, 0): 0.5}) == {"(1, 0)": 0.5}
    assert AnnexIVEvidenceVault._truncate_circuit("notadict") == "notadict"


# ── convenience function + file output ─────────────────────────────────────
def test_build_annex_iv_vault_writes_files(tmp_path):
    jp = tmp_path / "out" / "vault.json"
    hp = tmp_path / "out" / "vault.html"
    vault = build_annex_iv_vault(
        gb_result=GB, model_name="gpt2", provider="Acme Corp",
        output_json=str(jp), output_html=str(hp),
    )
    assert jp.exists() and hp.exists()
    assert vault.model_name == "gpt2"
    data = json.loads(jp.read_text())
    assert data["provider"] == "Acme Corp"
    assert "<" in hp.read_text()


def test_save_json_creates_dirs(tmp_path):
    v = AnnexIVEvidenceVault().build_vault(gb_result=GB)
    p = tmp_path / "nested" / "dir" / "v.json"
    v.save_json(str(p))
    assert p.exists()
    assert json.loads(p.read_text())["n_entries"] > 0
