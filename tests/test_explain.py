# SPDX-License-Identifier: MIT
"""
tests/test_explain.py — coverage for the Natural Language Explainer.

explain.py is pure-Python (no torch): it maps Glassbox result dicts to
plain-English / HTML compliance text. These tests exercise every verbosity
level, every faithfulness grade branch, every risk-flag path, and the
module-level convenience function, with no model required.
"""

import pytest

from glassbox.explain import NaturalLanguageExplainer, explain


# ---------------------------------------------------------------------------
# Sample result fixtures spanning the grade bands
# ---------------------------------------------------------------------------

def _result(suff, comp, f1, n_heads, category="faithful", circuit=None, **extra):
    res = {
        "faithfulness": {
            "sufficiency": suff,
            "comprehensiveness": comp,
            "f1": f1,
            "category": category,
        },
        "n_heads": n_heads,
        "circuit": circuit if circuit is not None else [(0, 1), (2, 3), (5, 7)],
        "model_name": "gpt2",
    }
    res.update(extra)
    return res


@pytest.fixture
def excellent():
    return _result(0.95, 0.80, 0.87, 3, "faithful")


@pytest.fixture
def good():
    return _result(0.78, 0.72, 0.75, 5, "faithful")


@pytest.fixture
def marginal():
    return _result(0.60, 0.40, 0.48, 6, "moderate")


@pytest.fixture
def poor():
    return _result(0.30, 0.10, 0.15, 0, "incomplete", circuit=[])


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        ex = NaturalLanguageExplainer()
        assert ex.language == "en"
        assert ex.verbosity == "standard"
        assert ex.include_article_refs is True

    def test_non_english_raises(self):
        with pytest.raises(NotImplementedError):
            NaturalLanguageExplainer(language="de")

    def test_bad_verbosity_raises(self):
        with pytest.raises(ValueError):
            NaturalLanguageExplainer(verbosity="verbose")

    @pytest.mark.parametrize("v", ["brief", "standard", "detailed"])
    def test_valid_verbosities(self, v):
        assert NaturalLanguageExplainer(verbosity=v).verbosity == v


# ---------------------------------------------------------------------------
# headline()
# ---------------------------------------------------------------------------

class TestHeadline:
    def test_headline_is_string(self, excellent):
        h = NaturalLanguageExplainer().headline(excellent)
        assert isinstance(h, str) and h

    def test_headline_mentions_heads_and_percent(self, excellent):
        h = NaturalLanguageExplainer().headline(excellent)
        assert "head" in h and "%" in h
        assert "Behaviour category" in h

    def test_headline_singular_head(self):
        h = NaturalLanguageExplainer().headline(_result(0.9, 0.8, 0.85, 1))
        assert "1 attention head " in h  # singular, no trailing 's'

    def test_headline_empty_result(self):
        # missing faithfulness/n_heads must not crash
        h = NaturalLanguageExplainer().headline({})
        assert isinstance(h, str) and "0" in h


# ---------------------------------------------------------------------------
# Grade bands (_suff_grade via headline text)
# ---------------------------------------------------------------------------

class TestGradeBands:
    @pytest.mark.parametrize("suff,grade", [
        (0.95, "Excellent"),
        (0.80, "Good"),
        (0.60, "Marginal"),
        (0.30, "Poor"),
    ])
    def test_grade_word_in_headline(self, suff, grade):
        h = NaturalLanguageExplainer().headline(_result(suff, 0.5, 0.5, 4))
        assert grade.lower() in h.lower()


# ---------------------------------------------------------------------------
# explain() across verbosities
# ---------------------------------------------------------------------------

class TestExplain:
    def test_brief_is_just_verdict(self, good):
        ex = NaturalLanguageExplainer(verbosity="brief")
        out = ex.explain(good)
        assert out == ex.explain_sections(good)["verdict"]

    def test_standard_has_multiple_paragraphs(self, good):
        out = NaturalLanguageExplainer(verbosity="standard").explain(good)
        assert out.count("\n\n") >= 2

    def test_detailed_includes_technical_detail(self, good):
        out = NaturalLanguageExplainer(verbosity="detailed").explain(good)
        assert "Technical metric summary" in out

    def test_standard_excludes_technical_detail(self, good):
        out = NaturalLanguageExplainer(verbosity="standard").explain(good)
        assert "Technical metric summary" not in out

    def test_article_refs_toggle(self, good):
        with_refs = NaturalLanguageExplainer(include_article_refs=True).explain(good)
        without = NaturalLanguageExplainer(include_article_refs=False).explain(good)
        assert "Regulation (EU) 2024/1689" in with_refs
        assert "Regulation (EU) 2024/1689" not in without

    def test_module_level_explain(self, good):
        out = explain(good)
        assert isinstance(out, str) and out


# ---------------------------------------------------------------------------
# explain_sections()
# ---------------------------------------------------------------------------

class TestSections:
    REQUIRED = {
        "verdict", "circuit_description", "faithfulness_analysis",
        "compliance_summary", "risk_flags", "stability_summary", "technical_detail",
    }

    def test_all_keys_present(self, excellent):
        sec = NaturalLanguageExplainer().explain_sections(excellent)
        assert self.REQUIRED.issubset(sec.keys())

    def test_values_are_strings(self, excellent):
        sec = NaturalLanguageExplainer().explain_sections(excellent)
        assert all(isinstance(v, str) for v in sec.values())

    def test_prompt_truncation_long(self, good):
        long_prompt = "x" * 200
        sec = NaturalLanguageExplainer().explain_sections(good, prompt=long_prompt)
        assert "..." in sec["verdict"]

    def test_prompt_short(self, good):
        sec = NaturalLanguageExplainer().explain_sections(good, prompt="hi")
        assert '"hi"' in sec["verdict"]

    def test_model_name_override(self, good):
        sec = NaturalLanguageExplainer().explain_sections(good, model_name="Llama")
        assert "Llama" in sec["verdict"]


# ---------------------------------------------------------------------------
# Verdict / circuit / faithfulness branches
# ---------------------------------------------------------------------------

class TestVerdictBranches:
    @pytest.mark.parametrize("suff", [0.95, 0.80, 0.60, 0.30])
    def test_verdict_nonempty_each_band(self, suff):
        sec = NaturalLanguageExplainer().explain_sections(_result(suff, 0.5, 0.5, 4))
        assert sec["verdict"]

    def test_no_circuit_description(self, poor):
        sec = NaturalLanguageExplainer().explain_sections(poor)
        assert "No causal circuit" in sec["circuit_description"]

    def test_sparse_circuit(self):
        sec = NaturalLanguageExplainer().explain_sections(_result(0.9, 0.8, 0.85, 2))
        assert "sparse" in sec["circuit_description"].lower()

    def test_moderate_circuit(self):
        sec = NaturalLanguageExplainer().explain_sections(_result(0.9, 0.8, 0.85, 6))
        assert "moderately complex" in sec["circuit_description"].lower()

    def test_distributed_circuit_lists_more(self):
        circuit = [(i, i) for i in range(10)]
        sec = NaturalLanguageExplainer().explain_sections(
            _result(0.9, 0.8, 0.85, 10, circuit=circuit))
        assert "others" in sec["circuit_description"]
        assert "distributed" in sec["circuit_description"].lower()


# ---------------------------------------------------------------------------
# Compliance summary branches
# ---------------------------------------------------------------------------

class TestComplianceSummary:
    def test_meets(self, good):
        sec = NaturalLanguageExplainer().explain_sections(good)
        assert "MEETS" in sec["compliance_summary"]

    def test_partially_meets(self, marginal):
        sec = NaturalLanguageExplainer().explain_sections(marginal)
        assert "PARTIALLY MEETS" in sec["compliance_summary"]

    def test_does_not_meet(self, poor):
        sec = NaturalLanguageExplainer().explain_sections(poor)
        assert "DOES NOT MEET" in sec["compliance_summary"]


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------

class TestRiskFlags:
    def test_clean_result_no_flags(self, excellent):
        sec = NaturalLanguageExplainer().explain_sections(excellent)
        assert sec["risk_flags"] == ""

    def test_low_faithfulness_flag(self, poor):
        sec = NaturalLanguageExplainer().explain_sections(poor)
        assert "LOW FAITHFULNESS" in sec["risk_flags"]

    def test_no_circuit_flag(self, poor):
        sec = NaturalLanguageExplainer().explain_sections(poor)
        assert "NO CIRCUIT FOUND" in sec["risk_flags"]

    def test_low_comprehensiveness_flag(self):
        sec = NaturalLanguageExplainer().explain_sections(
            _result(0.8, 0.10, 0.4, 4))
        assert "LOW COMPREHENSIVENESS" in sec["risk_flags"]

    def test_backup_mechanism_flag(self):
        sec = NaturalLanguageExplainer().explain_sections(
            _result(0.8, 0.5, 0.6, 4, category="backup_mechanisms"))
        assert "REDUNDANT PROCESSING" in sec["risk_flags"]

    def test_distributed_circuit_flag(self):
        sec = NaturalLanguageExplainer().explain_sections(
            _result(0.8, 0.5, 0.6, 12))
        assert "DISTRIBUTED CIRCUIT" in sec["risk_flags"]

    def test_instability_flag(self):
        res = _result(0.8, 0.5, 0.6, 4, stability={"mean_jaccard": 0.4})
        sec = NaturalLanguageExplainer().explain_sections(res)
        assert "CIRCUIT INSTABILITY" in sec["risk_flags"]


# ---------------------------------------------------------------------------
# Stability summary
# ---------------------------------------------------------------------------

class TestStabilitySummary:
    def test_no_stability_empty(self, good):
        sec = NaturalLanguageExplainer().explain_sections(good)
        assert sec["stability_summary"] == ""

    @pytest.mark.parametrize("mj,word", [
        (0.90, "highly stable"),
        (0.78, "stable"),
        (0.60, "moderately stable"),
        (0.40, "unstable"),
    ])
    def test_stability_quality_words(self, mj, word):
        res = _result(0.8, 0.5, 0.6, 4, stability={
            "mean_jaccard": mj, "std_jaccard": 0.05,
            "stability_rate": 0.8, "n_prompts": 10})
        sec = NaturalLanguageExplainer().explain_sections(res)
        assert word in sec["stability_summary"]

    def test_stability_missing_mean_jaccard(self, good):
        res = dict(good)
        res["stability"] = {"std_jaccard": 0.1}  # no mean_jaccard
        sec = NaturalLanguageExplainer().explain_sections(res)
        assert sec["stability_summary"] == ""


# ---------------------------------------------------------------------------
# to_html()
# ---------------------------------------------------------------------------

class TestToHtml:
    def test_returns_html(self, good):
        html = NaturalLanguageExplainer().to_html(good, model_name="gpt2", prompt="hi")
        assert "<div" in html and "Glassbox Explainability Report" in html

    def test_html_contains_metrics_table(self, good):
        html = NaturalLanguageExplainer().to_html(good)
        assert "Sufficiency" in html and "F1 Score" in html

    def test_html_grade_badge(self, excellent):
        html = NaturalLanguageExplainer().to_html(excellent)
        assert "Excellent Faithfulness" in html


# ---------------------------------------------------------------------------
# Technical detail
# ---------------------------------------------------------------------------

class TestTechnicalDetail:
    def test_detail_lists_metrics(self, good):
        res = dict(good)
        res["logit_diff"] = 3.14
        res["logit_diff_corrupted"] = 0.5
        out = NaturalLanguageExplainer(verbosity="detailed").explain(res)
        assert "logit_diff (clean)" in out
        assert "circuit_size" in out


# ---------------------------------------------------------------------------
# Category formatting
# ---------------------------------------------------------------------------

class TestCategoryFormatting:
    def test_empty_category(self):
        h = NaturalLanguageExplainer().headline(_result(0.9, 0.8, 0.85, 3, category=""))
        assert "Unclassified" in h

    def test_underscore_category_titlecased(self):
        h = NaturalLanguageExplainer().headline(
            _result(0.9, 0.8, 0.85, 3, category="backup_mechanisms"))
        assert "Backup Mechanisms" in h
