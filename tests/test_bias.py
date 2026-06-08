# SPDX-License-Identifier: MIT
"""
tests/test_bias.py — coverage for the fairness / bias analysis module.

bias.py works in offline mode from pre-computed logprobs dicts (no model
needed). These tests exercise all three analyses (counterfactual fairness,
demographic parity, token bias), both offline and online (model_fn) paths,
the error branches, bias categorisation, and the BiasReport aggregator.
"""

import pytest

from glassbox.bias import (
    BiasAnalyzer,
    BiasReport,
    CounterfactualFairnessResult,
    DemographicParityResult,
    TokenBiasResult,
)


@pytest.fixture
def analyzer():
    return BiasAnalyzer()


# ---------------------------------------------------------------------------
# Counterfactual fairness
# ---------------------------------------------------------------------------

class TestCounterfactual:
    TEMPLATE = "The {attribute} applicant should be"

    def test_offline_flagged_high(self, analyzer):
        res = analyzer.counterfactual_fairness_test(
            prompt_template=self.TEMPLATE,
            groups={"gender": ["male", "female"]},
            target_tokens=["hired", "rejected"],
            logprobs={"male": {"hired": 0.7, "rejected": 0.3},
                      "female": {"hired": 0.4, "rejected": 0.6}},
        )
        assert isinstance(res, CounterfactualFairnessResult)
        assert res.max_gap == pytest.approx(0.3)
        assert res.flagged is True
        assert res.bias_category == "high"
        assert res.recommendations  # high → non-empty

    def test_offline_low_category(self, analyzer):
        res = analyzer.counterfactual_fairness_test(
            prompt_template=self.TEMPLATE,
            groups={"g": ["a", "b"]},
            target_tokens=["x"],
            logprobs={"a": {"x": 0.50}, "b": {"x": 0.52}},
        )
        assert res.bias_category == "low"
        assert res.flagged is False

    def test_offline_medium_category(self, analyzer):
        res = analyzer.counterfactual_fairness_test(
            prompt_template=self.TEMPLATE,
            groups={"g": ["a", "b"]},
            target_tokens=["x"],
            logprobs={"a": {"x": 0.50}, "b": {"x": 0.60}},
        )
        assert res.bias_category == "medium"

    def test_online_model_fn(self, analyzer):
        def mf(prompt):
            # check "female" first: the substring "male" is inside "female"
            return {"hired": 0.5, "rejected": 0.5} if "female" in prompt else {"hired": 0.8, "rejected": 0.2}
        res = analyzer.counterfactual_fairness_test(
            prompt_template=self.TEMPLATE,
            groups={"gender": ["male", "female"]},
            target_tokens=["hired", "rejected"],
            model_fn=mf,
        )
        assert res.max_gap == pytest.approx(0.3)

    def test_missing_token_gives_zero_gap(self, analyzer):
        res = analyzer.counterfactual_fairness_test(
            prompt_template=self.TEMPLATE,
            groups={"g": ["a", "b"]},
            target_tokens=["absent"],
            logprobs={"a": {"x": 0.5}, "b": {"x": 0.5}},
        )
        assert res.parity_gap["absent"] == 0.0
        assert res.max_gap == 0.0

    def test_custom_threshold(self, analyzer):
        res = analyzer.counterfactual_fairness_test(
            prompt_template=self.TEMPLATE,
            groups={"g": ["a", "b"]},
            target_tokens=["x"],
            logprobs={"a": {"x": 0.7}, "b": {"x": 0.6}},
            threshold=0.20,
        )
        assert res.threshold == 0.20
        assert res.flagged is False  # gap 0.1 < 0.2

    def test_missing_placeholder_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.counterfactual_fairness_test(
                prompt_template="no placeholder",
                groups={"g": ["a"]}, target_tokens=["x"],
                logprobs={"a": {"x": 0.5}})

    def test_no_source_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.counterfactual_fairness_test(
                prompt_template=self.TEMPLATE,
                groups={"g": ["a"]}, target_tokens=["x"])

    def test_to_dict(self, analyzer):
        res = analyzer.counterfactual_fairness_test(
            prompt_template=self.TEMPLATE, groups={"g": ["a", "b"]},
            target_tokens=["x"], logprobs={"a": {"x": 0.5}, "b": {"x": 0.5}})
        d = res.to_dict()
        assert d["max_gap"] == 0.0 and "eu_ai_act_articles" in d

    def test_logprobs_with_nondict_value_rebuilds_from_groups(self, analyzer):
        # A non-dict value makes the "all dicts" fast-path fail, exercising the
        # rebuild-from-groups branch that reads logprobs[value] directly.
        res = analyzer.counterfactual_fairness_test(
            prompt_template=self.TEMPLATE,
            groups={"g": ["a"]},
            target_tokens=["x"],
            logprobs={"a": {"x": 0.5}, "_marker": 0},
        )
        assert res.probabilities == {"a": {"x": 0.5}}


# ---------------------------------------------------------------------------
# Demographic parity
# ---------------------------------------------------------------------------

class TestDemographicParity:
    def test_offline_flagged(self, analyzer):
        res = analyzer.demographic_parity_test(
            prompts_by_group={"a": ["p1"], "b": ["p2"]},
            target_tokens=["approved"],
            logprobs_by_group={"a": [{"approved": 0.8}], "b": [{"approved": 0.4}]},
        )
        assert isinstance(res, DemographicParityResult)
        assert res.parity_gap == pytest.approx(0.4)
        assert res.flagged is True
        assert set(res.groups) == {"a", "b"}

    def test_online_model_fn(self, analyzer):
        def mf(prompt):
            return {"approved": 0.9} if "rich" in prompt else {"approved": 0.5}
        res = analyzer.demographic_parity_test(
            prompts_by_group={"hi": ["rich person"], "lo": ["poor person"]},
            target_tokens=["approved"],
            model_fn=mf,
        )
        assert res.parity_gap == pytest.approx(0.4)

    def test_no_source_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.demographic_parity_test(
                prompts_by_group={"a": ["p"]}, target_tokens=["x"])

    def test_empty_groups_zero_gap(self, analyzer):
        res = analyzer.demographic_parity_test(
            prompts_by_group={}, target_tokens=["x"], logprobs_by_group={})
        assert res.parity_gap == 0.0

    def test_to_dict(self, analyzer):
        res = analyzer.demographic_parity_test(
            prompts_by_group={"a": ["p"]}, target_tokens=["x"],
            logprobs_by_group={"a": [{"x": 0.5}]})
        assert res.to_dict()["parity_gap"] == 0.0


# ---------------------------------------------------------------------------
# Token bias probe
# ---------------------------------------------------------------------------

class TestTokenBias:
    CTX = ["The {token} is a"]

    def test_offline_flagged_pair(self, analyzer):
        res = analyzer.token_bias_probe(
            demographic_tokens=["man", "woman"],
            context_templates=self.CTX,
            logprobs={"man": {"The {token} is a": 0.85},
                      "woman": {"The {token} is a": 0.40}},
        )
        assert isinstance(res, TokenBiasResult)
        assert res.overall_bias_score == pytest.approx(0.625)
        assert len(res.flagged_pairs) == 1  # 0.85 > 0.70
        assert res.flagged_pairs[0][0] == "man"

    def test_online_model_fn(self, analyzer):
        def mf(prompt):
            return {"doctor": 0.9, "nurse": 0.1}
        res = analyzer.token_bias_probe(
            demographic_tokens=["man"], context_templates=self.CTX, model_fn=mf)
        # association score = max prob = 0.9 > 0.7 → flagged
        assert res.flagged_pairs and res.flagged_pairs[0][2] == pytest.approx(0.9)

    def test_missing_token_placeholder_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.token_bias_probe(
                demographic_tokens=["man"], context_templates=["no placeholder"],
                logprobs={"man": {}})

    def test_no_source_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.token_bias_probe(
                demographic_tokens=["man"], context_templates=self.CTX)

    def test_no_flagged_pairs_low_scores(self, analyzer):
        res = analyzer.token_bias_probe(
            demographic_tokens=["man", "woman"], context_templates=self.CTX,
            logprobs={"man": {"The {token} is a": 0.2},
                      "woman": {"The {token} is a": 0.3}})
        assert res.flagged_pairs == []

    def test_online_empty_model_output_zero_score(self, analyzer):
        # model_fn returns an empty dict -> association score falls back to 0.0.
        res = analyzer.token_bias_probe(
            demographic_tokens=["man"], context_templates=self.CTX,
            model_fn=lambda p: {})
        assert res.overall_bias_score == 0.0

    def test_to_dict(self, analyzer):
        res = analyzer.token_bias_probe(
            demographic_tokens=["man"], context_templates=self.CTX,
            logprobs={"man": {"The {token} is a": 0.5}})
        d = res.to_dict()
        assert "demographic_tokens" in d and "overall_bias_score" in d


# ---------------------------------------------------------------------------
# Bias categorisation boundaries
# ---------------------------------------------------------------------------

class TestCategorize:
    @pytest.mark.parametrize("val,cat", [
        (0.0, "low"), (0.05, "low"),
        (0.06, "medium"), (0.15, "medium"),
        (0.16, "high"), (0.9, "high"),
    ])
    def test_boundaries(self, val, cat):
        assert BiasAnalyzer._categorize_bias(val) == cat


# ---------------------------------------------------------------------------
# BiasReport aggregator
# ---------------------------------------------------------------------------

class TestBiasReport:
    def _cf(self, analyzer, gap_logprobs):
        return analyzer.counterfactual_fairness_test(
            prompt_template="The {attribute} is",
            groups={"g": ["a", "b"]}, target_tokens=["x"], logprobs=gap_logprobs)

    def test_add_result_autoname(self, analyzer):
        r = BiasReport("gpt2")
        r.add_result(self._cf(analyzer, {"a": {"x": 0.5}, "b": {"x": 0.5}}))
        assert "test_0" in r.results

    def test_add_result_named(self, analyzer):
        r = BiasReport()
        r.add_result(self._cf(analyzer, {"a": {"x": 0.5}, "b": {"x": 0.5}}), test_name="Loan")
        assert "Loan" in r.results

    def test_overall_score_all_types(self, analyzer):
        r = BiasReport("m")
        r.add_result(self._cf(analyzer, {"a": {"x": 0.9}, "b": {"x": 0.1}}), "cf")  # gap 0.8
        r.add_result(analyzer.demographic_parity_test(
            prompts_by_group={"a": ["p"], "b": ["q"]}, target_tokens=["x"],
            logprobs_by_group={"a": [{"x": 0.6}], "b": [{"x": 0.2}]}), "dp")  # gap 0.4
        r.add_result(analyzer.token_bias_probe(
            demographic_tokens=["man"], context_templates=["The {token} is"],
            logprobs={"man": {"The {token} is": 0.5}}), "tb")  # score 0.5
        score = r.overall_bias_score()
        assert score == pytest.approx((0.8 + 0.4 + 0.5) / 3)

    def test_overall_score_empty(self):
        assert BiasReport().overall_bias_score() == 0.0

    def test_flagged_tests(self, analyzer):
        r = BiasReport()
        r.add_result(self._cf(analyzer, {"a": {"x": 0.9}, "b": {"x": 0.1}}), "flagged_one")
        r.add_result(self._cf(analyzer, {"a": {"x": 0.5}, "b": {"x": 0.5}}), "clean_one")
        assert r.flagged_tests() == ["flagged_one"]

    def test_to_dict(self, analyzer):
        r = BiasReport("gpt2")
        r.add_result(self._cf(analyzer, {"a": {"x": 0.9}, "b": {"x": 0.1}}), "cf")
        d = r.to_dict()
        assert d["model_name"] == "gpt2" and d["total_tests"] == 1
        assert d["flagged_count"] == 1 and "cf" in d["results"]

    def test_to_markdown_all_types_and_flag(self, analyzer):
        r = BiasReport("gpt2")
        r.add_result(self._cf(analyzer, {"a": {"x": 0.9}, "b": {"x": 0.1}}), "cf")
        r.add_result(analyzer.demographic_parity_test(
            prompts_by_group={"a": ["p"], "b": ["q"]}, target_tokens=["x"],
            logprobs_by_group={"a": [{"x": 0.6}], "b": [{"x": 0.2}]}), "dp")
        r.add_result(analyzer.token_bias_probe(
            demographic_tokens=["man"], context_templates=["The {token} is"],
            logprobs={"man": {"The {token} is": 0.85}}), "tb")
        md = r.to_markdown()
        assert "Bias Analysis Report" in md
        assert "Counterfactual Fairness" in md
        assert "Demographic Parity" in md
        assert "Token Bias Probe" in md
        assert "ACTION REQUIRED" in md

    def test_to_markdown_compliant(self, analyzer):
        r = BiasReport()
        r.add_result(self._cf(analyzer, {"a": {"x": 0.5}, "b": {"x": 0.5}}), "clean")
        md = r.to_markdown()
        assert "COMPLIANT" in md
