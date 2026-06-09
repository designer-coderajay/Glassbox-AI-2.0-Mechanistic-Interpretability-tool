"""Tests for glassbox/multiagent.py — multi-agent handoff liability audit."""
import json

import pytest

from glassbox.multiagent import (
    AgentCall,
    BiasSignals,
    LiabilityReport,
    MultiAgentAudit,
)


def _chain():
    return [
        AgentCall("researcher", "gpt-4", "Assess this loan applicant.",
                  "The applicant appears risky and unreliable."),
        AgentCall("analyst", "gpt-4", "The applicant appears risky and unreliable.",
                  "This person will likely default; recommend denial."),
        AgentCall("decider", "gpt-4", "This person will likely default; recommend denial.",
                  "Decision: application denied."),
    ]


# ── AgentCall ──────────────────────────────────────────────────────────────
def test_agent_call_defaults():
    c = AgentCall("a", "m", "in", "out")
    assert c.agent_id == "a"
    assert isinstance(c.timestamp, float)   # auto-set
    assert c.metadata == {}


# ── BiasSignals.overall_bias_score ─────────────────────────────────────────
def test_overall_bias_score_weighted():
    b = BiasSignals(
        category_scores={"gender": 0.1, "race": 0.2},
        toxicity_score=0.5, sentiment_score=0.4,
        top_categories=["race"], flagged_tokens=["x"],
    )
    # 0.5*mean(0.15) + 0.3*0.5 + 0.2*0.4 = 0.305
    assert b.overall_bias_score == pytest.approx(0.305, abs=1e-3)


def test_overall_bias_score_empty_categories():
    b = BiasSignals(category_scores={}, toxicity_score=0.2,
                    sentiment_score=0.1, top_categories=[], flagged_tokens=[])
    # cat_mean -> 0.0
    assert b.overall_bias_score == pytest.approx(0.3 * 0.2 + 0.2 * 0.1, abs=1e-3)


# ── audit_chain ────────────────────────────────────────────────────────────
def test_audit_empty_raises():
    with pytest.raises(ValueError):
        MultiAgentAudit().audit_chain([])


def test_audit_single_agent():
    report = MultiAgentAudit().audit_chain([AgentCall("solo", "m", "in", "out")])
    assert isinstance(report, LiabilityReport)
    assert report.n_agents == 1
    assert len(report.agent_scores) == 1
    assert report.handoff_analyses == []   # no handoffs with one agent
    assert report.chain_risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_audit_full_chain_structure():
    report = MultiAgentAudit().audit_chain(_chain())
    assert report.n_agents == 3
    assert len(report.agent_scores) == 3
    assert len(report.handoff_analyses) == 2          # n-1 handoffs
    assert report.most_liable_agent in {"researcher", "analyst", "decider"}
    assert isinstance(report.article_violations, list)
    assert isinstance(report.annex_iv_text, str) and report.annex_iv_text
    assert len(report.chain_id) == 12                 # sha256[:12].upper()
    assert report.chain_id == report.chain_id.upper()


def test_handoff_verdicts_are_valid():
    report = MultiAgentAudit().audit_chain(_chain())
    for h in report.handoff_analyses:
        assert h.verdict in {"CLEAN", "FORWARDED", "AMPLIFIED", "INTRODUCED"}
        assert 0.0 <= h.contamination_score <= 1.0
        assert h.from_agent and h.to_agent


def test_agent_verdicts_are_valid():
    report = MultiAgentAudit().audit_chain(_chain())
    for a in report.agent_scores:
        assert a.verdict in {"CLEAN", "MINOR", "MODERATE", "HIGH", "CRITICAL"}
        assert 0.0 <= a.responsibility_score <= 1.0


# ── serialisation ──────────────────────────────────────────────────────────
def test_report_to_dict_and_json():
    report = MultiAgentAudit().audit_chain(_chain())
    d = report.to_dict()
    assert d["n_agents"] == 3
    assert "agent_scores" in d and "handoff_analyses" in d
    parsed = json.loads(report.to_json())
    assert parsed["chain_id"] == report.chain_id


def test_to_html_renders():
    audit = MultiAgentAudit()
    report = audit.audit_chain(_chain())
    html = audit.to_html(report)
    assert isinstance(html, str)
    assert "<" in html


# ── PII-safety toggle ──────────────────────────────────────────────────────
def test_include_full_text_false_runs():
    report = MultiAgentAudit(include_full_text=False).audit_chain(_chain())
    assert report.n_agents == 3  # still produces a full report


def test_clean_chain_low_risk():
    clean = [
        AgentCall("a", "m", "Summarise the quarterly report.",
                  "The quarterly report covers revenue and costs."),
        AgentCall("b", "m", "The quarterly report covers revenue and costs.",
                  "Revenue and costs were within the expected range."),
    ]
    report = MultiAgentAudit().audit_chain(clean)
    assert report.n_agents == 2
    assert report.chain_risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
