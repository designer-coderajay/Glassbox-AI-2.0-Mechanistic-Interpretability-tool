# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ajay Pravin Mahale
"""
glassbox/explain.py — Natural Language Explainer
=================================================

Translates mechanistic interpretability results into plain English that
compliance officers, legal teams, and non-technical stakeholders can read.

No LLM dependency. Pure template + heuristic logic. Fast, deterministic,
auditable. Every output sentence cites the metric it came from.

Usage
-----
    from glassbox import GlassboxV2, NaturalLanguageExplainer

    gb     = GlassboxV2(model)
    result = gb.analyze(prompt, correct, incorrect)

    ex  = NaturalLanguageExplainer()
    txt = ex.explain(result)
    print(txt)

    # Structured sections dict
    sections = ex.explain_sections(result)
    print(sections["verdict"])
    print(sections["circuit_description"])
    print(sections["compliance_summary"])
    print(sections["risk_flags"])

    # One-liner for the audit log
    headline = ex.headline(result)

    # Full HTML for embedding in an Annex IV report
    html = ex.to_html(result, model_name="gpt2", prompt="...")
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal threshold constants (tuneable)
# ---------------------------------------------------------------------------

_SUFF_EXCELLENT  = 0.90
_SUFF_GOOD       = 0.75
_SUFF_MARGINAL   = 0.50
_COMP_HIGH       = 0.70
_F1_GOOD         = 0.65
_JACCARD_STABLE  = 0.75
_N_HEADS_FEW     = 3
_N_HEADS_MODERATE= 8

# ---------------------------------------------------------------------------
# Article references for EU AI Act citations
# ---------------------------------------------------------------------------

_ARTICLE_REFS = {
    "explainability": "Article 13(3)(b) Regulation (EU) 2024/1689",
    "risk":           "Article 9   Regulation (EU) 2024/1689",
    "monitoring":     "Article 72  Regulation (EU) 2024/1689",
    "accuracy":       "Article 15  Regulation (EU) 2024/1689",
    "transparency":   "Article 13  Regulation (EU) 2024/1689",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NaturalLanguageExplainer:
    """
    Converts Glassbox analysis results into plain-English compliance reports.

    Parameters
    ----------
    language : str
        Report language. Only ``'en'`` is supported in the current release.
        ``'de'``, ``'fr'``, ``'nl'`` are planned for a future release.
    verbosity : str
        ``'brief'``    — 1–2 sentence verdict only.
        ``'standard'`` — 4–6 paragraph structured report (default).
        ``'detailed'`` — Full technical narrative with metric citations.
    include_article_refs : bool
        Append EU AI Act article citations to each relevant sentence.
    """

    def __init__(
        self,
        language: str = "en",
        verbosity: str = "standard",
        include_article_refs: bool = True,
    ) -> None:
        if language != "en":
            raise NotImplementedError("Only language='en' is supported in the current release.")
        if verbosity not in ("brief", "standard", "detailed"):
            raise ValueError("verbosity must be 'brief', 'standard', or 'detailed'.")
        self.language             = language
        self.verbosity            = verbosity
        self.include_article_refs = include_article_refs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def headline(self, result: Dict[str, Any]) -> str:
        """Return a single-sentence verdict headline (≤ 20 words)."""
        faith   = result.get("faithfulness", {})
        suff    = faith.get("sufficiency", 0.0)
        cat     = faith.get("category", "unknown")
        n_heads = result.get("n_heads", 0)
        grade   = self._suff_grade(suff)

        return (
            f"Decision explained by {n_heads} attention head{'s' if n_heads != 1 else ''} "
            f"with {grade} causal faithfulness ({suff:.0%}). "
            f"Behaviour category: {self._fmt_category(cat)}."
        )

    def explain(self, result: Dict[str, Any], **kwargs) -> str:
        """Return the full plain-English explanation as a single string."""
        sections = self.explain_sections(result, **kwargs)
        if self.verbosity == "brief":
            return sections["verdict"]
        parts = [
            sections["verdict"],
            sections["circuit_description"],
            sections["faithfulness_analysis"],
            sections["compliance_summary"],
        ]
        if sections.get("risk_flags"):
            parts.append(sections["risk_flags"])
        if sections.get("stability_summary"):
            parts.append(sections["stability_summary"])
        if self.verbosity == "detailed" and sections.get("technical_detail"):
            parts.append(sections["technical_detail"])
        return "\n\n".join(p for p in parts if p)

    def explain_sections(
        self,
        result: Dict[str, Any],
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Return explanation as a dict of named sections.

        Keys
        ----
        verdict                — top-line pass/fail sentence
        circuit_description    — what the circuit is
        faithfulness_analysis  — sufficiency / comprehensiveness interpretation
        compliance_summary     — EU AI Act Article 13 / 15 assessment
        risk_flags             — flagged concerns (empty string if none)
        stability_summary      — stability_suite() summary if present
        technical_detail       — raw metric citations (detailed verbosity only)
        """
        faith   = result.get("faithfulness", {})
        suff    = faith.get("sufficiency", 0.0)
        comp    = faith.get("comprehensiveness", 0.0)
        f1      = faith.get("f1", 0.0)
        cat     = faith.get("category", "unknown")
        n_heads = result.get("n_heads", 0)
        circuit = result.get("circuit", [])
        model   = model_name or result.get("model_name", "the model")
        prompt_str = f'"{prompt[:60]}..."' if prompt and len(prompt) > 60 else (f'"{prompt}"' if prompt else "the input prompt")

        return {
            "verdict":             self._verdict(suff, f1, n_heads, model, prompt_str),
            "circuit_description": self._circuit_description(circuit, n_heads, model),
            "faithfulness_analysis": self._faithfulness_analysis(suff, comp, f1, cat),
            "compliance_summary":  self._compliance_summary(suff, comp, f1, n_heads),
            "risk_flags":          self._risk_flags(suff, comp, n_heads, cat, result),
            "stability_summary":   self._stability_summary(result),
            "technical_detail":    self._technical_detail(result) if self.verbosity == "detailed" else "",
        }

    def to_html(
        self,
        result: Dict[str, Any],
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Return a self-contained HTML block suitable for embedding in an
        Annex IV compliance report or a web dashboard.
        """
        sections = self.explain_sections(result, model_name=model_name, prompt=prompt)
        faith    = result.get("faithfulness", {})
        suff     = faith.get("sufficiency", 0.0)
        grade    = self._suff_grade(suff)
        color    = {"Excellent": "#22C55E", "Good": "#84CC16",
                    "Marginal": "#F59E0B", "Poor": "#EF4444"}.get(grade, "#94A3B8")

        rows = ""
        for k, v in [
            ("Sufficiency",        f"{suff:.1%}"),
            ("Comprehensiveness",  f"{faith.get('comprehensiveness', 0):.1%}"),
            ("F1 Score",           f"{faith.get('f1', 0):.1%}"),
            ("Circuit Heads",      str(result.get("n_heads", 0))),
            ("Category",           self._fmt_category(faith.get("category", ""))),
            ("Faithfulness Grade", grade),
        ]:
            rows += f"<tr><td style='padding:6px 12px;color:#94A3B8;'>{k}</td><td style='padding:6px 12px;font-weight:600;'>{v}</td></tr>"

        body = ""
        labels = {
            "verdict": "Verdict",
            "circuit_description": "Circuit Description",
            "faithfulness_analysis": "Faithfulness Analysis",
            "compliance_summary": "EU AI Act Compliance",
            "risk_flags": "Risk Flags",
            "stability_summary": "Circuit Stability",
        }
        for key, label in labels.items():
            if sections.get(key):
                body += f"""
                <div style='margin-bottom:16px;'>
                  <h4 style='color:#7B96B8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>{label}</h4>
                  <p style='color:#E8F0FE;font-size:0.875rem;line-height:1.6;'>{sections[key]}</p>
                </div>"""

        return f"""
<div style='background:#0F1F3D;border:1px solid #1E3A5F;border-radius:12px;padding:24px;font-family:Inter,-apple-system,sans-serif;'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;flex-wrap:wrap;gap:12px;'>
    <div>
      <h3 style='color:#02C39A;font-size:1rem;font-weight:700;margin:0;'>Glassbox Explainability Report</h3>
      <p style='color:#7B96B8;font-size:0.8rem;margin:4px 0 0;'>Mechanistic interpretability — plain English summary</p>
    </div>
    <span style='background:rgba(2,195,154,0.15);color:{color};border:1px solid {color};padding:4px 14px;border-radius:20px;font-size:0.8rem;font-weight:700;'>{grade} Faithfulness</span>
  </div>
  <table style='width:100%;border-collapse:collapse;background:#0A1628;border-radius:8px;overflow:hidden;margin-bottom:20px;'>{rows}</table>
  {body}
  <p style='color:#3B556B;font-size:0.7rem;margin-top:20px;border-top:1px solid #1E3A5F;padding-top:12px;'>
    Generated by Glassbox AI v4.2.6 · {_ARTICLE_REFS["explainability"]} · {_ARTICLE_REFS["transparency"]}
  </p>
</div>"""

    # ------------------------------------------------------------------
    # Section builders (private)
    # ------------------------------------------------------------------

    def _verdict(self, suff: float, f1: float, n_heads: int,
                 model: str, prompt_str: str) -> str:
        grade = self._suff_grade(suff)
        if suff >= _SUFF_EXCELLENT:
            verdict_core = (
                f"{model.title()} produced a decision on {prompt_str} that is "
                f"causally explained with {grade.lower()} faithfulness ({suff:.0%}) "
                f"by a circuit of {n_heads} attention head{'s' if n_heads != 1 else ''}."
            )
        elif suff >= _SUFF_GOOD:
            verdict_core = (
                f"{model.title()} produced a decision on {prompt_str} that is "
                f"substantially explained ({suff:.0%} causal faithfulness) "
                f"by {n_heads} attention head{'s' if n_heads != 1 else ''}. "
                f"Some residual variance is unexplained by the identified circuit."
            )
        elif suff >= _SUFF_MARGINAL:
            verdict_core = (
                f"The decision made by {model.title()} on {prompt_str} is only "
                f"partially explained ({suff:.0%} causal faithfulness). "
                f"The identified circuit of {n_heads} head{'s' if n_heads != 1 else ''} "
                f"accounts for a minority of the decision signal."
            )
        else:
            verdict_core = (
                f"Causal faithfulness is low ({suff:.0%}). "
                f"The identified {n_heads}-head circuit does not adequately explain "
                f"the decision made by {model.title()} on {prompt_str}. "
                f"Manual review is recommended before deployment in high-risk contexts."
            )
        ref = f" [{_ARTICLE_REFS['explainability']}]" if self.include_article_refs else ""
        return verdict_core + ref

    def _circuit_description(self, circuit: list, n_heads: int, model: str) -> str:
        if not circuit or n_heads == 0:
            return (
                "No causal circuit was identified for this decision. "
                "This may indicate the model relies on distributed representations "
                "that cannot be localised to specific attention heads using the "
                "current attribution threshold."
            )
        head_labels = [
            f"L{h[0]}H{h[1]}" if isinstance(h, (list, tuple)) and len(h) >= 2
            else str(h)
            for h in circuit[:6]
        ]
        more = f" and {n_heads - 6} others" if n_heads > 6 else ""
        heads_str = ", ".join(head_labels) + more

        if n_heads <= _N_HEADS_FEW:
            complexity = (
                f"This is a sparse, localised circuit — "
                f"fewer than {_N_HEADS_FEW + 1} heads carry the causal signal, "
                f"which is easier to audit and monitor over time."
            )
        elif n_heads <= _N_HEADS_MODERATE:
            complexity = (
                f"This is a moderately complex circuit. "
                f"{n_heads} heads collaborate on the decision, "
                f"which is within normal range for this task type."
            )
        else:
            complexity = (
                f"This is a distributed circuit involving {n_heads} heads, "
                f"which may indicate the model uses parallel strategies or backup mechanisms. "
                f"Broader monitoring scope is recommended."
            )
        return (
            f"The causal circuit responsible for this decision involves {n_heads} "
            f"attention head{'s' if n_heads != 1 else ''}: {heads_str}. "
            f"{complexity}"
        )

    def _faithfulness_analysis(
        self, suff: float, comp: float, f1: float, cat: str
    ) -> str:
        suff_sent = (
            f"Causal sufficiency is {suff:.0%} — "
            + {
                True:  "when only the identified circuit is active and all other components are ablated, the model still makes the correct decision.",
                False: "the identified circuit alone is insufficient to reproduce the full decision signal.",
            }[suff >= _SUFF_GOOD]
        )

        comp_sent = (
            f"Comprehensiveness is {comp:.0%} — "
            + (
                "ablating the circuit significantly disrupts the decision, confirming it is causally necessary."
                if comp >= _COMP_HIGH else
                "ablating the circuit has limited effect, suggesting distributed or redundant processing."
            )
        )

        cat_sent = f"Behaviour category: {self._fmt_category(cat)}."

        f1_sent = (
            f"The combined faithfulness F1 score is {f1:.0%}, "
            + (
                "indicating a well-balanced circuit that is both necessary and sufficient."
                if f1 >= _F1_GOOD else
                "suggesting an imbalance between sufficiency and comprehensiveness."
            )
        )

        ref = f" [{_ARTICLE_REFS['accuracy']}]" if self.include_article_refs else ""
        return f"{suff_sent} {comp_sent} {cat_sent} {f1_sent}{ref}"

    def _compliance_summary(
        self, suff: float, comp: float, f1: float, n_heads: int
    ) -> str:
        if suff >= _SUFF_GOOD and n_heads > 0:
            status = "MEETS"
            detail = (
                f"The causal circuit provides a mechanistically grounded explanation "
                f"that satisfies the explainability standard."
            )
        elif suff >= _SUFF_MARGINAL:
            status = "PARTIALLY MEETS"
            detail = (
                f"The circuit partially explains the decision. "
                f"Supplementary documentation of residual decision factors is recommended."
            )
        else:
            status = "DOES NOT MEET"
            detail = (
                f"Causal faithfulness ({suff:.0%}) is insufficient for deployment "
                f"in high-risk contexts without additional explainability measures."
            )

        ref_e = f" [{_ARTICLE_REFS['explainability']}]" if self.include_article_refs else ""
        ref_r = f" [{_ARTICLE_REFS['risk']}]" if self.include_article_refs else ""

        return (
            f"EU AI Act Annex IV Explainability Assessment: {status}.{ref_e} "
            f"{detail} "
            f"Risk management documentation should record this faithfulness score as evidence "
            f"of technical explainability capability.{ref_r}"
        )

    def _risk_flags(
        self, suff: float, comp: float, n_heads: int, cat: str, result: Dict[str, Any]
    ) -> str:
        flags: List[str] = []

        if suff < _SUFF_MARGINAL:
            flags.append(
                f"LOW FAITHFULNESS: Sufficiency {suff:.0%} is below the 50% threshold. "
                f"Manual explainability review required before high-risk deployment."
            )

        if comp < 0.20 and n_heads > 0:
            flags.append(
                f"LOW COMPREHENSIVENESS: The identified circuit ({comp:.0%}) is weakly causally necessary. "
                f"The model may rely on backup mechanisms not captured in this circuit."
            )

        if cat in ("backup_mechanisms", "high_redundancy"):
            flags.append(
                f"REDUNDANT PROCESSING DETECTED: Category '{self._fmt_category(cat)}' indicates "
                f"the model uses multiple parallel strategies. Circuit monitoring should cover "
                f"a broader head set to satisfy Article 72 post-market monitoring."
            )

        if n_heads > _N_HEADS_MODERATE:
            flags.append(
                f"DISTRIBUTED CIRCUIT: {n_heads} heads involved. "
                f"Large circuits increase audit scope and may signal model instability across fine-tuning."
            )

        if n_heads == 0:
            flags.append(
                "NO CIRCUIT FOUND: The attribution threshold did not identify any causally relevant heads. "
                "Lower the threshold or use a more specific contrastive prompt pair."
            )

        # Check for stability results
        stab = result.get("stability", {})
        if stab:
            mean_j = stab.get("mean_jaccard", 1.0)
            if mean_j < _JACCARD_STABLE:
                flags.append(
                    f"CIRCUIT INSTABILITY: Mean Jaccard stability across prompts is {mean_j:.2f}, "
                    f"below the 0.75 threshold. Circuit may not generalise reliably. "
                    f"Review stability_suite() results before using this circuit for compliance documentation."
                )

        if not flags:
            return ""

        ref = f" [{_ARTICLE_REFS['risk']}]" if self.include_article_refs else ""
        header = f"Risk flags identified ({len(flags)}){ref}:\n"
        return header + "\n".join(f"• {f}" for f in flags)

    def _stability_summary(self, result: Dict[str, Any]) -> str:
        stab = result.get("stability", {})
        if not stab:
            return ""
        mean_j   = stab.get("mean_jaccard", None)
        std_j    = stab.get("std_jaccard", None)
        rate     = stab.get("stability_rate", None)
        n        = stab.get("n_prompts", None)

        if mean_j is None:
            return ""

        quality = (
            "highly stable" if mean_j >= 0.85 else
            "stable"        if mean_j >= _JACCARD_STABLE else
            "moderately stable" if mean_j >= 0.55 else
            "unstable"
        )

        n_str  = f" across {n} prompt variants" if n else ""
        std_str = f" (±{std_j:.2f})" if std_j is not None else ""
        rate_str = f" {rate:.0%} of prompt variants produced a circuit with Jaccard ≥ 0.75." if rate is not None else ""

        ref = f" [{_ARTICLE_REFS['monitoring']}]" if self.include_article_refs else ""
        return (
            f"Circuit stability{n_str}: The circuit is {quality} with mean Jaccard "
            f"similarity {mean_j:.2f}{std_str}.{rate_str} "
            f"This supports post-market monitoring documentation.{ref}"
        )

    def _technical_detail(self, result: Dict[str, Any]) -> str:
        faith   = result.get("faithfulness", {})
        lines   = ["Technical metric summary:"]
        for k, v in faith.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        if "logit_diff" in result:
            lines.append(f"  logit_diff (clean): {result['logit_diff']:.4f}")
        if "logit_diff_corrupted" in result:
            lines.append(f"  logit_diff (corrupted): {result['logit_diff_corrupted']:.4f}")
        if "n_heads" in result:
            lines.append(f"  circuit_size: {result['n_heads']} heads")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _suff_grade(suff: float) -> str:
        if suff >= _SUFF_EXCELLENT: return "Excellent"
        if suff >= _SUFF_GOOD:      return "Good"
        if suff >= _SUFF_MARGINAL:  return "Marginal"
        return "Poor"

    @staticmethod
    def _fmt_category(cat: str) -> str:
        return cat.replace("_", " ").title() if cat else "Unclassified"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def explain(
    result: Dict[str, Any],
    verbosity: str = "standard",
    include_article_refs: bool = True,
    **kwargs,
) -> str:
    """
    One-shot explain — equivalent to ``NaturalLanguageExplainer().explain(result)``.

    Example
    -------
    ::

        from glassbox import GlassboxV2
        from glassbox.explain import explain

        gb     = GlassboxV2(model)
        result = gb.analyze(prompt, correct=" Mary", incorrect=" John")
        print(explain(result))
    """
    return NaturalLanguageExplainer(
        verbosity=verbosity,
        include_article_refs=include_article_refs,
    ).explain(result, **kwargs)
