# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ajay Pravin Mahale
"""
glassbox/multiagent.py — Multi-Agent Causal Handoff Tracing
=============================================================

The "Black Box Flight Recorder" for agentic AI systems.

In 2026, enterprise AI is swarms: Researcher → Analyst → CFO.
When the output is wrong or harmful, nobody knows which agent caused it.
This module traces *causal responsibility* across an agent chain and
produces a per-agent Liability Report for EU AI Act Article 9 compliance.

No weight access required. Works with any API-accessible agent chain.
White-box mode available when TransformerLens weights are loaded.

Core concepts
-------------
Contamination Score
    How much of Agent B's output bias/drift is *causally explained* by
    Agent A's output vs. Agent B's own processing.
    Score = 0.0 → B introduced everything itself.
    Score = 1.0 → B purely forwarded what A gave it.

Semantic Drift
    Vocabulary-space shift between an agent's input and output.
    High drift = agent significantly transformed the content.
    Low drift + high bias = agent is *laundering* upstream bias.

Bias Signal
    Keyword-density proxy for protected-attribute toxicity.
    Covers: gender, race/ethnicity, nationality, religion, age,
    disability, sexuality — all protected under EU AI Act Article 10.

Usage
-----
::

    from glassbox.multiagent import MultiAgentAudit, AgentCall

    audit = MultiAgentAudit()

    chain = [
        AgentCall("researcher",   "gpt-4",        user_query,   researcher_output),
        AgentCall("analyst",      "claude-3-opus", researcher_output, analyst_output),
        AgentCall("decision-bot", "llama-3-8b",   analyst_output,    final_decision),
    ]

    report = audit.audit_chain(chain)
    print(report.most_liable_agent)           # "analyst"
    print(report.chain_risk_level)            # "HIGH"
    print(report.annex_iv_text)               # full Article 9 section

    # HTML liability dashboard
    html = audit.to_html(report)

EU AI Act mapping
-----------------
Article 9   — Risk management system (system-level, not just model-level)
Article 13  — Transparency and provision of information to deployers
Article 17  — Quality management system for post-market monitoring
Article 72  — Post-market monitoring (continuous agentic system drift)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Protected-attribute keyword sets (EU AI Act Article 10 — 8 categories)
# ---------------------------------------------------------------------------

_BIAS_LEXICON: Dict[str, List[str]] = {
    "gender": [
        "woman", "women", "man", "men", "female", "male", "girl", "boy",
        "she", "he", "her", "his", "transgender", "nonbinary", "gender",
        "feminine", "masculine", "housewife", "breadwinner",
    ],
    "race_ethnicity": [
        "black", "white", "asian", "hispanic", "latino", "latina", "african",
        "caucasian", "indigenous", "native", "arab", "jewish", "race",
        "ethnicity", "minority", "diversity",
    ],
    "nationality": [
        "immigrant", "foreigner", "illegal", "alien", "refugee", "citizen",
        "nationality", "migrant", "deportation",
    ],
    "religion": [
        "christian", "muslim", "jewish", "hindu", "buddhist", "atheist",
        "religion", "faith", "prayer", "mosque", "church", "synagogue",
        "radical", "extremist", "jihad",
    ],
    "age": [
        "old", "elderly", "senior", "aged", "young", "millennial", "boomer",
        "ageist", "retirement", "youth",
    ],
    "disability": [
        "disabled", "handicapped", "wheelchair", "blind", "deaf",
        "mentally", "disorder", "autism", "adhd", "disability",
    ],
    "sexuality": [
        "gay", "lesbian", "bisexual", "queer", "lgbtq", "homosexual",
        "heterosexual", "straight", "sexuality",
    ],
    "socioeconomic": [
        "poor", "rich", "wealthy", "homeless", "welfare", "poverty",
        "privileged", "underprivileged", "class", "working-class",
    ],
}

_TOXICITY_LEXICON: List[str] = [
    "hate", "kill", "murder", "attack", "threat", "violent", "dangerous",
    "terrorist", "bomb", "weapon", "abuse", "harass", "illegal", "fraud",
    "manipulate", "deceive", "corrupt",
]

_SENTIMENT_NEGATIVE: List[str] = [
    "bad", "wrong", "terrible", "awful", "horrible", "worst", "fail",
    "failure", "deny", "reject", "refuse", "cannot", "should not",
    "unqualified", "risky", "suspicious",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentCall:
    """
    One step in a multi-agent chain.

    Parameters
    ----------
    agent_id : str
        Human-readable identifier (e.g. ``"researcher"``, ``"analyst"``).
    model_name : str
        Model used for this call (e.g. ``"gpt-4"``, ``"llama-3-8b"``).
    input_text : str
        Text the agent received as input.
    output_text : str
        Text the agent produced as output.
    timestamp : float, optional
        Unix timestamp of the call. Auto-set if not provided.
    metadata : dict, optional
        Any additional context (tool calls, system prompt excerpt, etc.)
    """
    agent_id:    str
    model_name:  str
    input_text:  str
    output_text: str
    timestamp:   float = field(default_factory=time.time)
    metadata:    Dict[str, Any] = field(default_factory=dict)


@dataclass
class BiasSignals:
    """Per-text bias keyword density scores."""
    category_scores:  Dict[str, float]      # category → density (0–1)
    toxicity_score:   float                 # 0–1
    sentiment_score:  float                 # 0–1 (higher = more negative)
    top_categories:   List[str]             # categories above 0.01 threshold
    flagged_tokens:   List[str]             # all matched bias keywords

    @property
    def overall_bias_score(self) -> float:
        """Weighted composite bias score."""
        cat_mean = (
            sum(self.category_scores.values()) / len(self.category_scores)
            if self.category_scores else 0.0
        )
        return round(0.5 * cat_mean + 0.3 * self.toxicity_score + 0.2 * self.sentiment_score, 4)


@dataclass
class HandoffAnalysis:
    """
    Analysis of one agent-to-agent handoff (A → B).
    """
    from_agent:          str
    to_agent:            str
    contamination_score: float      # 0–1: how much of B's bias came from A
    drift_score:         float      # 0–1: semantic drift in B's processing
    bias_introduced:     float      # bias present in B's output but not A's output
    bias_amplified:      float      # ratio B_bias / A_bias (>1 = amplified)
    input_bias:          BiasSignals
    output_bias:         BiasSignals
    verdict:             str        # "CLEAN" / "FORWARDED" / "AMPLIFIED" / "INTRODUCED"
    article_flags:       List[str]  # EU AI Act articles triggered


@dataclass
class AgentLiabilityScore:
    """
    Per-agent liability assessment.
    """
    agent_id:             str
    model_name:           str
    responsibility_score: float          # 0–1: share of chain liability
    drift_introduced:     float          # semantic drift in this agent's processing
    bias_score_input:     float          # bias in what this agent received
    bias_score_output:    float          # bias in what this agent produced
    bias_delta:           float          # output_bias - input_bias (positive = introduced)
    contamination_to_next: Optional[float]  # how much this agent contaminated the next
    verdict:              str            # "CLEAN" / "MINOR" / "MODERATE" / "HIGH" / "CRITICAL"
    article_violations:   List[str]
    evidence_summary:     str


@dataclass
class LiabilityReport:
    """
    Full multi-agent liability report.
    """
    chain_id:            str
    timestamp_utc:       str
    n_agents:            int
    agent_scores:        List[AgentLiabilityScore]
    handoff_analyses:    List[HandoffAnalysis]
    most_liable_agent:   str
    chain_risk_level:    str             # "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
    chain_bias_score:    float           # end-to-end bias amplification
    article_violations:  List[str]       # deduplicated across all agents
    annex_iv_text:       str             # Article 9 compliant system-level risk section
    raw_chain:           List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Core audit engine
# ---------------------------------------------------------------------------

class MultiAgentAudit:
    """
    Multi-agent causal handoff tracing and liability attribution.

    Traces bias, semantic drift, and legal risk across an agent chain.
    No model weights required — works with text inputs/outputs only.

    Parameters
    ----------
    bias_threshold : float
        Minimum bias signal density to flag an agent. Default 0.02 (2%).
    contamination_threshold : float
        Minimum contamination score to flag a handoff. Default 0.5.
    include_full_text : bool
        Include input/output text in the report. Set False for PII safety.
    """

    def __init__(
        self,
        bias_threshold:          float = 0.02,
        contamination_threshold: float = 0.50,
        include_full_text:       bool  = True,
    ) -> None:
        self.bias_threshold          = bias_threshold
        self.contamination_threshold = contamination_threshold
        self.include_full_text       = include_full_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit_chain(self, calls: List[AgentCall]) -> LiabilityReport:
        """
        Audit a full multi-agent chain and return a LiabilityReport.

        Parameters
        ----------
        calls : list of AgentCall
            Ordered list of agent calls, from first to last in the chain.

        Returns
        -------
        LiabilityReport
        """
        if len(calls) < 1:
            raise ValueError("Need at least one AgentCall to audit.")

        chain_id = hashlib.sha256(
            "".join(c.agent_id + c.output_text[:50] for c in calls).encode()
        ).hexdigest()[:12].upper()

        # ── Compute per-agent bias signals ─────────────────────────────
        agent_bias: List[BiasSignals] = [
            self._bias_signals(c.output_text) for c in calls
        ]
        input_bias: List[BiasSignals] = [
            self._bias_signals(c.input_text) for c in calls
        ]

        # ── Handoff analyses ───────────────────────────────────────────
        handoffs: List[HandoffAnalysis] = []
        for i in range(len(calls) - 1):
            handoffs.append(
                self._analyse_handoff(calls[i], calls[i + 1],
                                      agent_bias[i], agent_bias[i + 1],
                                      input_bias[i + 1])
            )

        # ── Per-agent liability scores ─────────────────────────────────
        agent_scores: List[AgentLiabilityScore] = []
        for i, call in enumerate(calls):
            in_bias  = input_bias[i].overall_bias_score
            out_bias = agent_bias[i].overall_bias_score
            delta    = round(out_bias - in_bias, 4)
            drift    = self._semantic_drift(call.input_text, call.output_text)

            contam_next = handoffs[i].contamination_score if i < len(handoffs) else None

            # Responsibility = how much bias DELTA this agent contributed
            # relative to the total chain delta
            responsibility = max(0.0, delta)

            verdict = self._agent_verdict(delta, out_bias)
            articles = self._article_flags(agent_bias[i], delta)

            evidence = self._evidence_summary(call, in_bias, out_bias, delta, drift, verdict)

            agent_scores.append(AgentLiabilityScore(
                agent_id              = call.agent_id,
                model_name            = call.model_name,
                responsibility_score  = round(responsibility, 4),
                drift_introduced      = round(drift, 4),
                bias_score_input      = round(in_bias, 4),
                bias_score_output     = round(out_bias, 4),
                bias_delta            = delta,
                contamination_to_next = round(contam_next, 4) if contam_next else None,
                verdict               = verdict,
                article_violations    = articles,
                evidence_summary      = evidence,
            ))

        # ── Normalise responsibility scores ────────────────────────────
        total_resp = sum(s.responsibility_score for s in agent_scores) or 1.0
        for s in agent_scores:
            s.responsibility_score = round(s.responsibility_score / total_resp, 4)

        # ── Chain-level metrics ────────────────────────────────────────
        first_bias = input_bias[0].overall_bias_score
        last_bias  = agent_bias[-1].overall_bias_score
        chain_bias = round(last_bias - first_bias, 4)

        all_violations = sorted(set(
            v for s in agent_scores for v in s.article_violations
        ))
        most_liable = max(agent_scores, key=lambda s: s.responsibility_score).agent_id
        risk_level  = self._chain_risk_level(chain_bias, all_violations, agent_scores)

        annex_iv = self._generate_annex_iv(
            chain_id, calls, agent_scores, handoffs, chain_bias, risk_level, all_violations
        )

        raw_chain = [
            {
                "agent_id":   c.agent_id,
                "model_name": c.model_name,
                "input_preview":  c.input_text[:200] if self.include_full_text else "[redacted]",
                "output_preview": c.output_text[:200] if self.include_full_text else "[redacted]",
                "timestamp":  c.timestamp,
                "metadata":   c.metadata,
            }
            for c in calls
        ]

        return LiabilityReport(
            chain_id           = chain_id,
            timestamp_utc      = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            n_agents           = len(calls),
            agent_scores       = agent_scores,
            handoff_analyses   = handoffs,
            most_liable_agent  = most_liable,
            chain_risk_level   = risk_level,
            chain_bias_score   = chain_bias,
            article_violations = all_violations,
            annex_iv_text      = annex_iv,
            raw_chain          = raw_chain,
        )

    def to_html(self, report: LiabilityReport) -> str:
        """Generate a self-contained HTML liability dashboard."""
        risk_color = {
            "LOW": "#22C55E", "MEDIUM": "#F59E0B",
            "HIGH": "#EF4444", "CRITICAL": "#DC2626"
        }.get(report.chain_risk_level, "#94A3B8")

        rows = ""
        for s in report.agent_scores:
            v_color = {
                "CLEAN": "#22C55E", "MINOR": "#84CC16",
                "MODERATE": "#F59E0B", "HIGH": "#EF4444", "CRITICAL": "#DC2626"
            }.get(s.verdict, "#94A3B8")
            bar_w = int(s.responsibility_score * 200)
            rows += f"""
            <tr>
              <td style="padding:12px;color:#C8DCF5;font-weight:600;">{s.agent_id}</td>
              <td style="padding:12px;color:#7B96B8;font-size:0.8rem;">{s.model_name}</td>
              <td style="padding:12px;">
                <div style="background:#1E3A5F;border-radius:4px;height:8px;width:200px;">
                  <div style="background:{v_color};border-radius:4px;height:8px;width:{bar_w}px;"></div>
                </div>
                <span style="color:{v_color};font-size:0.75rem;">{s.responsibility_score:.0%}</span>
              </td>
              <td style="padding:12px;color:{v_color};font-weight:700;">{s.verdict}</td>
              <td style="padding:12px;color:#7B96B8;font-size:0.78rem;">{s.bias_delta:+.3f}</td>
              <td style="padding:12px;color:#7B96B8;font-size:0.78rem;">{", ".join(s.article_violations) or "—"}</td>
            </tr>"""

        handoff_rows = ""
        for h in report.handoff_analyses:
            v_color = {"CLEAN": "#22C55E", "FORWARDED": "#84CC16",
                       "AMPLIFIED": "#F59E0B", "INTRODUCED": "#EF4444"}.get(h.verdict, "#94A3B8")
            handoff_rows += f"""
            <tr>
              <td style="padding:10px;color:#C8DCF5;">{h.from_agent} → {h.to_agent}</td>
              <td style="padding:10px;color:#7B96B8;">{h.contamination_score:.2f}</td>
              <td style="padding:10px;color:#7B96B8;">{h.drift_score:.2f}</td>
              <td style="padding:10px;color:#7B96B8;">{h.bias_introduced:+.3f}</td>
              <td style="padding:10px;color:{v_color};font-weight:700;">{h.verdict}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Glassbox — Multi-Agent Liability Report {report.chain_id}</title>
<style>
  body{{background:#0A1628;color:#E8F0FE;font-family:Inter,-apple-system,sans-serif;margin:0;padding:24px;}}
  table{{width:100%;border-collapse:collapse;}}
  thead th{{background:#1E3A5F;padding:12px 16px;text-align:left;color:#7B96B8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;border-bottom:2px solid #02C39A;}}
  tbody tr:hover{{background:rgba(2,195,154,0.04);}}
  .card{{background:#0F1F3D;border:1px solid #1E3A5F;border-radius:12px;padding:20px;margin-bottom:20px;}}
  .risk-badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.8rem;}}
</style>
</head>
<body>
<div style="border-bottom:2px solid #02C39A;padding-bottom:16px;margin-bottom:24px;">
  <h1 style="color:#02C39A;font-size:1.6rem;font-weight:800;margin:0;">Glassbox — Multi-Agent Liability Report</h1>
  <p style="color:#7B96B8;margin:4px 0 0;font-size:0.85rem;">Chain ID: {report.chain_id} · {report.timestamp_utc} · {report.n_agents} agents · Regulation (EU) 2024/1689 Article 9</p>
</div>

<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;">
  <div class="card" style="text-align:center;">
    <div style="font-size:2rem;font-weight:800;color:{risk_color};">{report.chain_risk_level}</div>
    <div style="color:#7B96B8;font-size:0.75rem;margin-top:4px;">Chain Risk Level</div>
  </div>
  <div class="card" style="text-align:center;">
    <div style="font-size:2rem;font-weight:800;color:#02C39A;">{report.most_liable_agent}</div>
    <div style="color:#7B96B8;font-size:0.75rem;margin-top:4px;">Most Liable Agent</div>
  </div>
  <div class="card" style="text-align:center;">
    <div style="font-size:2rem;font-weight:800;color:#C8DCF5;">{report.chain_bias_score:+.3f}</div>
    <div style="color:#7B96B8;font-size:0.75rem;margin-top:4px;">Chain Bias Delta</div>
  </div>
  <div class="card" style="text-align:center;">
    <div style="font-size:2rem;font-weight:800;color:#F59E0B;">{len(report.article_violations)}</div>
    <div style="color:#7B96B8;font-size:0.75rem;margin-top:4px;">Article Violations</div>
  </div>
</div>

<div class="card">
  <h2 style="color:#C8DCF5;font-size:1rem;margin:0 0 16px;">Agent Liability Breakdown</h2>
  <table>
    <thead><tr>
      <th>Agent</th><th>Model</th><th>Responsibility</th><th>Verdict</th><th>Bias Delta</th><th>Articles</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>

<div class="card">
  <h2 style="color:#C8DCF5;font-size:1rem;margin:0 0 16px;">Handoff Analysis</h2>
  <table>
    <thead><tr>
      <th>Handoff</th><th>Contamination</th><th>Drift</th><th>Bias Introduced</th><th>Verdict</th>
    </tr></thead>
    <tbody>{handoff_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2 style="color:#C8DCF5;font-size:1rem;margin:0 0 12px;">EU AI Act Article 9 — System-Level Risk Section</h2>
  <div style="color:#E8F0FE;font-size:0.85rem;line-height:1.7;white-space:pre-wrap;">{report.annex_iv_text}</div>
</div>

<p style="color:#3B556B;font-size:0.7rem;margin-top:24px;">
  Generated by Glassbox AI v3.4.0 · Multi-Agent Causal Handoff Tracing · Regulation (EU) 2024/1689
</p>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _bias_signals(self, text: str) -> BiasSignals:
        """Compute bias keyword density across all protected categories."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        total  = max(len(tokens), 1)

        category_scores: Dict[str, float] = {}
        flagged: List[str] = []

        for category, keywords in _BIAS_LEXICON.items():
            hits = [t for t in tokens if t in keywords]
            flagged.extend(hits)
            category_scores[category] = round(len(hits) / total, 5)

        toxicity_hits = [t for t in tokens if t in _TOXICITY_LEXICON]
        toxicity      = round(len(toxicity_hits) / total, 5)
        flagged.extend(toxicity_hits)

        sentiment_hits = [t for t in tokens if t in _SENTIMENT_NEGATIVE]
        sentiment      = round(len(sentiment_hits) / total, 5)

        top_cats = [c for c, s in category_scores.items() if s >= 0.01]

        return BiasSignals(
            category_scores = category_scores,
            toxicity_score  = toxicity,
            sentiment_score = sentiment,
            top_categories  = top_cats,
            flagged_tokens  = list(set(flagged)),
        )

    def _semantic_drift(self, text_a: str, text_b: str) -> float:
        """
        Vocabulary-space drift between two texts.
        Uses Jaccard distance on unigram token sets.
        Drift = 1 - Jaccard(A, B)  → 0 = identical, 1 = no shared words.
        """
        if not text_a.strip() or not text_b.strip():
            return 0.0
        tokens_a = set(re.findall(r'\b\w+\b', text_a.lower()))
        tokens_b = set(re.findall(r'\b\w+\b', text_b.lower()))
        # Remove stop words for more meaningful comparison
        stops = {"the","a","an","is","are","was","were","be","been","being",
                 "have","has","had","do","does","did","will","would","could",
                 "should","may","might","shall","can","to","of","in","for",
                 "on","with","at","by","from","up","about","into","through",
                 "and","or","but","if","as","it","its","that","this","these",
                 "those","i","you","we","they","he","she","not","no"}
        tokens_a -= stops
        tokens_b -= stops
        if not tokens_a and not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union        = tokens_a | tokens_b
        jaccard      = len(intersection) / len(union) if union else 1.0
        return round(1.0 - jaccard, 4)

    def _contamination_score(
        self,
        output_a: str,
        input_b:  str,
        output_b: str,
    ) -> float:
        """
        Estimate how much of Agent B's output is causally explained by
        Agent A's output vs. B's own processing.

        Method: If B's input contains A's output, measure what fraction of
        B's bias keywords also appeared in A's output (forwarded) vs. were
        new (introduced by B).

        Returns 0.0 (B generated everything fresh) to 1.0 (B just forwarded A).
        """
        if not output_a.strip():
            return 0.0

        tokens_a_out = set(re.findall(r'\b\w+\b', output_a.lower()))
        tokens_b_out = set(re.findall(r'\b\w+\b', output_b.lower()))

        all_bias_tokens: set = set()
        for kws in _BIAS_LEXICON.values():
            all_bias_tokens.update(kws)
        all_bias_tokens.update(_TOXICITY_LEXICON)

        # Bias tokens in B's output
        b_bias = tokens_b_out & all_bias_tokens
        if not b_bias:
            return 0.0  # B has no bias tokens — contamination irrelevant

        # Of B's bias tokens, how many came from A's output?
        forwarded = b_bias & tokens_a_out
        return round(len(forwarded) / len(b_bias), 4)

    def _analyse_handoff(
        self,
        call_a:     AgentCall,
        call_b:     AgentCall,
        bias_a_out: BiasSignals,
        bias_b_out: BiasSignals,
        bias_b_in:  BiasSignals,
    ) -> HandoffAnalysis:
        contamination = self._contamination_score(
            call_a.output_text, call_b.input_text, call_b.output_text
        )
        drift = self._semantic_drift(call_b.input_text, call_b.output_text)

        a_score = bias_a_out.overall_bias_score
        b_score = bias_b_out.overall_bias_score
        in_score = bias_b_in.overall_bias_score

        bias_introduced = round(b_score - in_score, 4)
        bias_amplified  = round(b_score / a_score, 4) if a_score > 1e-6 else 0.0

        # Verdict logic
        if b_score < self.bias_threshold and bias_introduced < self.bias_threshold:
            verdict = "CLEAN"
        elif bias_introduced < self.bias_threshold and contamination >= self.contamination_threshold:
            verdict = "FORWARDED"   # B passed on A's bias without adding much
        elif bias_amplified > 1.5 and bias_introduced > self.bias_threshold:
            verdict = "AMPLIFIED"   # B took A's bias and made it worse
        elif bias_introduced > self.bias_threshold:
            verdict = "INTRODUCED"  # B generated bias from scratch
        else:
            verdict = "FORWARDED"

        articles = self._article_flags(bias_b_out, bias_introduced)

        return HandoffAnalysis(
            from_agent          = call_a.agent_id,
            to_agent            = call_b.agent_id,
            contamination_score = contamination,
            drift_score         = drift,
            bias_introduced     = bias_introduced,
            bias_amplified      = bias_amplified,
            input_bias          = bias_b_in,
            output_bias         = bias_b_out,
            verdict             = verdict,
            article_flags       = articles,
        )

    @staticmethod
    def _agent_verdict(delta: float, out_bias: float) -> str:
        if out_bias < 0.005 and abs(delta) < 0.005:
            return "CLEAN"
        if abs(delta) < 0.005:
            return "MINOR"
        if delta < 0.02:
            return "MODERATE"
        if delta < 0.05:
            return "HIGH"
        return "CRITICAL"

    @staticmethod
    def _article_flags(bias: BiasSignals, delta: float) -> List[str]:
        flags = []
        if bias.category_scores.get("gender", 0) > 0.01 or \
           bias.category_scores.get("race_ethnicity", 0) > 0.01:
            flags.append("Article 10(2)(f)")   # data governance / bias
        if bias.category_scores.get("disability", 0) > 0.01:
            flags.append("Article 10(5)")       # special categories
        if bias.toxicity_score > 0.01:
            flags.append("Article 15(1)")       # robustness / safety
        if delta > 0.03:
            flags.append("Article 9(2)")        # risk management — significant bias introduction
        if bias.overall_bias_score > 0.05:
            flags.append("Article 13(1)")       # transparency — deployer must be informed
        return sorted(set(flags))

    @staticmethod
    def _chain_risk_level(
        chain_bias: float,
        violations: List[str],
        scores: List[AgentLiabilityScore],
    ) -> str:
        critical_agents = sum(1 for s in scores if s.verdict in ("HIGH", "CRITICAL"))
        if critical_agents >= 2 or "Article 9(2)" in violations or chain_bias > 0.05:
            return "CRITICAL"
        if critical_agents >= 1 or chain_bias > 0.02 or len(violations) >= 3:
            return "HIGH"
        if chain_bias > 0.005 or len(violations) >= 1:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _evidence_summary(
        call: AgentCall,
        in_bias: float, out_bias: float,
        delta: float, drift: float, verdict: str,
    ) -> str:
        direction = "introduced" if delta > 0 else ("reduced" if delta < 0 else "did not change")
        return (
            f"Agent '{call.agent_id}' ({call.model_name}) {direction} bias "
            f"(Δ={delta:+.3f}, in={in_bias:.3f}→out={out_bias:.3f}). "
            f"Semantic drift: {drift:.2f}. Verdict: {verdict}."
        )

    @staticmethod
    def _generate_annex_iv(
        chain_id: str,
        calls: List[AgentCall],
        scores: List[AgentLiabilityScore],
        handoffs: List[HandoffAnalysis],
        chain_bias: float,
        risk_level: str,
        violations: List[str],
    ) -> str:
        agent_list = "\n".join(
            f"  {i+1}. {s.agent_id} ({s.model_name}) — "
            f"Responsibility: {s.responsibility_score:.0%}, Verdict: {s.verdict}, "
            f"Bias delta: {s.bias_delta:+.3f}"
            for i, s in enumerate(scores)
        )
        handoff_list = "\n".join(
            f"  {h.from_agent} → {h.to_agent}: "
            f"contamination={h.contamination_score:.2f}, drift={h.drift_score:.2f}, "
            f"verdict={h.verdict}"
            for h in handoffs
        ) or "  (single agent — no handoffs)"

        violations_str = ", ".join(violations) or "None identified"

        return f"""EU AI Act Article 9 — Multi-Agent System Risk Assessment
Chain ID: {chain_id}
Analysis date: {time.strftime("%Y-%m-%d", time.gmtime())}

1. SYSTEM COMPOSITION
   {len(calls)}-agent pipeline analysed by Glassbox v3.4.0 Multi-Agent Causal Handoff Tracing.

{agent_list}

2. CAUSAL HANDOFF ANALYSIS

{handoff_list}

3. CHAIN-LEVEL RISK ASSESSMENT
   Chain Bias Delta:  {chain_bias:+.4f} (end-to-end bias amplification)
   System Risk Level: {risk_level}
   Article Violations: {violations_str}

4. LIABILITY ATTRIBUTION
   Most liable agent: {scores[0].agent_id if scores else "N/A"} — see per-agent scores above.
   Attribution method: Glassbox keyword-density contamination tracing.
   Confidence: Indicative (behavioural, not mechanistic). For mechanistic confirmation,
   re-run with white-box analysis using GlassboxV2 on each agent's model weights.

5. REQUIRED ACTIONS UNDER EU AI ACT
   - Article 9(2): System-level risk register must document this multi-agent configuration.
   - Article 13(1): Deployers must be informed of agent-level bias scores.
   - Article 17: Quality management system must log this audit result.
   - Article 72: Post-market monitoring must re-run this audit after each agent update.

Generated by Glassbox AI v3.4.0 · Regulation (EU) 2024/1689
"""
