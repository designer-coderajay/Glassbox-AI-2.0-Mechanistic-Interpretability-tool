"""
glassbox/evidence_vault.py
===========================
Annex IV Evidence Vault

Builds a machine-readable, human-auditable evidence package that satisfies
Regulation (EU) 2024/1689 (EU AI Act) Annex IV documentation requirements for
high-risk AI systems.

The vault maps mechanistic-interpretability findings (circuit heads, SAE
features, faithfulness metrics, steering vector results) directly to the
Annex IV sections and underlying Articles they satisfy.  Everything is
rule-based — no LLM required.

EU AI Act Annex IV sections covered
------------------------------------
  § 1   General description of the AI system
  § 2   Detailed description of elements and development process
  § 3   Information on monitoring, functioning, and control
  § 4   Description of the risk management system (Article 9)
  § 5   Changes to the system through its lifecycle
  § 6   List of harmonised standards applied (or technical specs)
  § 7   Declaration of conformity certificate reference

Supporting Articles surfaced per finding
-----------------------------------------
  Article 9      Risk management system
  Article 10     Data and data governance
  Article 11     Technical documentation
  Article 13     Transparency and provision of information
  Article 15     Accuracy, robustness, and cybersecurity

Public API
----------
  AnnexIVEvidenceVault   — main class; build_vault(), to_json(), to_html()
  VaultEntry             — dataclass for individual evidence items
  build_annex_iv_vault() — convenience one-liner
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("glassbox-mech-interp")
except Exception:
    _VERSION = "unknown"
_REGULATION = "Regulation (EU) 2024/1689 — EU AI Act"

# ---------------------------------------------------------------------------
# Annex IV section catalogue
# ---------------------------------------------------------------------------
_ANNEX_IV_SECTIONS = {
    "§1": "General description of the AI system and its intended purpose",
    "§2": "Detailed description of system elements, training data, and development process",
    "§3": "Information on monitoring, functioning, and control mechanisms",
    "§4": "Description of the risk management system and measures per Article 9",
    "§5": "Changes to the system through its lifecycle",
    "§6": "List of harmonised standards or technical specifications applied",
    "§7": "Declaration of conformity (or reference to it)",
}

# ---------------------------------------------------------------------------
# Article reference table
# ---------------------------------------------------------------------------
_ARTICLES = {
    "Article 9":      "Risk management system — identification, analysis, estimation of risks",
    "Article 9(2)(b)":"Risk management — technical risk mitigation measures",
    "Article 9(5)":   "Testing of AI systems to identify mitigation measures",
    "Article 10":     "Data and data governance — training, validation, testing data",
    "Article 10(2)(f)":"Data — known biases and possible measures",
    "Article 10(5)":  "Special categories of data for bias monitoring",
    "Article 11":     "Technical documentation — before market placement",
    "Article 13":     "Transparency and provision of information to users",
    "Article 13(1)":  "Transparency enabling informed use of the AI system",
    "Article 15":     "Accuracy, robustness, and cybersecurity",
    "Article 15(1)":  "Appropriate level of accuracy and its metrics",
    "Article 72":     "Post-market monitoring — provider obligations",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VaultEntry:
    """
    A single evidence item in the Annex IV vault.

    Attributes
    ----------
    section        : Annex IV section reference (e.g. "§4")
    article_refs   : list of EU AI Act article references
    title          : short human-readable title
    description    : one-paragraph plain-English description
    evidence_type  : "faithfulness" | "circuit" | "bias" | "steering" |
                     "sae_feature" | "stability" | "data_governance" | "general"
    metric_name    : optional metric key
    metric_value   : optional scalar value
    threshold      : optional compliance threshold
    passed         : whether the evidence item passes its threshold
    raw            : raw data dict (machine-readable)
    timestamp_utc  : ISO timestamp
    """

    section: str
    article_refs: List[str]
    title: str
    description: str
    evidence_type: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    raw: Dict = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict:
        return {
            "section": self.section,
            "article_refs": self.article_refs,
            "title": self.title,
            "description": self.description,
            "evidence_type": self.evidence_type,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "passed": self.passed,
            "raw": self.raw,
            "timestamp_utc": self.timestamp_utc,
        }


# ---------------------------------------------------------------------------
# Main vault class
# ---------------------------------------------------------------------------

class AnnexIVEvidenceVault:
    """
    Builds and manages an Annex IV evidence package.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. "gpt2", "meta-llama/Llama-2-7b-hf").
    provider : str
        Organisation name for the conformity declaration.
    use_case : str
        Deployment use case description.
    deployment_ctx : str
        Deployment context string (e.g. "credit_scoring", "medical_diagnosis").
    version : str
        AI system version being documented.
    commit_sha : str
        Source code commit SHA for traceability.
    """

    def __init__(
        self,
        model_name: str = "unknown",
        provider: str = "Unknown",
        use_case: str = "CI/CD compliance audit",
        deployment_ctx: str = "other_high_risk",
        version: str = _VERSION,
        commit_sha: str = "unknown",
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.use_case = use_case
        self.deployment_ctx = deployment_ctx
        self.version = version
        self.commit_sha = commit_sha
        self.entries: List[VaultEntry] = []
        self._created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ------------------------------------------------------------------
    # Core builder: accepts all Glassbox analysis outputs
    # ------------------------------------------------------------------

    def build_vault(
        self,
        gb_result: Optional[Dict] = None,
        multiagent_report: Optional[Any] = None,
        steering_vectors: Optional[Dict] = None,
        steering_test_results: Optional[Dict[str, Dict]] = None,
        sae_features: Optional[List[Dict]] = None,
        stability_result: Optional[Dict] = None,
        custom_entries: Optional[List[VaultEntry]] = None,
    ) -> "AnnexIVEvidenceVault":
        """
        Populate the vault from all available Glassbox outputs.

        All parameters are optional — pass whatever you have.

        Parameters
        ----------
        gb_result : dict
            Output of GlassboxV2.analyze().
        multiagent_report : LiabilityReport or dict
            Output of MultiAgentAudit.audit_chain().
        steering_vectors : dict
            {concept_label: SteeringVector} from SteeringVectorExporter.
        steering_test_results : dict
            {concept_label: test_result_dict} from test_suppression().
        sae_features : list[dict]
            SAE feature activation dicts with keys: feature_id, activation,
            description, legal_risk_category, article_ref.
        stability_result : dict
            Output of GlassboxV2.stability_suite().
        custom_entries : list[VaultEntry]
            Any additional evidence items the caller wants to add.
        """
        self.entries.clear()

        # § 1 — General description (always present)
        self._add_general_description()

        # § 2 — Development process (from gb_result)
        if gb_result is not None:
            self._add_from_gb_result(gb_result)

        # § 3 — Monitoring / stability
        if stability_result is not None:
            self._add_stability_entries(stability_result)

        # § 4 — Risk management: steering vectors
        if steering_vectors is not None:
            self._add_steering_entries(steering_vectors, steering_test_results or {})

        # § 4 — Risk management: SAE features
        if sae_features is not None:
            self._add_sae_entries(sae_features)

        # § 4 — Risk management: multi-agent
        if multiagent_report is not None:
            self._add_multiagent_entries(multiagent_report)

        # § 6 — Standards alignment
        self._add_standards_entry()

        # § 7 — Conformity placeholder
        self._add_conformity_entry()

        # Custom additions
        if custom_entries:
            self.entries.extend(custom_entries)

        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return the full vault as a JSON-serialisable dict."""
        return {
            "glassbox_version": _VERSION,
            "regulation": _REGULATION,
            "model_name": self.model_name,
            "provider": self.provider,
            "use_case": self.use_case,
            "deployment_ctx": self.deployment_ctx,
            "version": self.version,
            "commit_sha": self.commit_sha,
            "created_at": self._created_at,
            "n_entries": len(self.entries),
            "sections_covered": sorted({e.section for e in self.entries}),
            "articles_covered": sorted({a for e in self.entries for a in e.article_refs}),
            "compliance_summary": self._compliance_summary(),
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the vault as a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str) -> None:
        """Save vault to a JSON file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(self.to_json())

    def to_html(self) -> str:
        """
        Render a self-contained HTML Evidence Vault report.
        Styled to be suitable for regulatory submission or attachment to
        a conformity declaration.
        """
        vault_dict = self.to_dict()
        summary = vault_dict["compliance_summary"]
        pct = summary.get("pass_rate", 0.0)
        overall_col = "#16a34a" if pct >= 0.80 else ("#d97706" if pct >= 0.50 else "#dc2626")
        overall_txt = "COMPLIANT" if pct >= 0.80 else ("MARGINAL" if pct >= 0.50 else "NON-COMPLIANT")

        # Build entry rows
        entry_rows = ""
        for e in self.entries:
            passed_badge = ""
            if e.passed is True:
                passed_badge = '<span class="badge pass">PASS</span>'
            elif e.passed is False:
                passed_badge = '<span class="badge fail">FAIL</span>'

            articles = ", ".join(e.article_refs) if e.article_refs else "—"
            metric_cell = (
                f"{e.metric_value:.4f}" if e.metric_value is not None else "—"
            )

            entry_rows += f"""
            <tr>
              <td><span class="sec-badge">{e.section}</span></td>
              <td>{e.title}</td>
              <td>{e.evidence_type}</td>
              <td style="font-size:11px">{articles}</td>
              <td>{metric_cell}</td>
              <td>{passed_badge}</td>
            </tr>"""

        # Build section breakdown cards
        section_cards = ""
        sections_seen: Dict[str, List[VaultEntry]] = {}
        for e in self.entries:
            sections_seen.setdefault(e.section, []).append(e)

        for sec, sec_entries in sorted(sections_seen.items()):
            sec_title = _ANNEX_IV_SECTIONS.get(sec, sec)
            items_html = "".join(
                f'<li><strong>{se.title}</strong> — {se.description}</li>'
                for se in sec_entries
            )
            section_cards += f"""
            <div class="section-card">
              <div class="sec-card-title">{sec} &mdash; {sec_title}</div>
              <ul>{items_html}</ul>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Annex IV Evidence Vault — {self.model_name}</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:system-ui,sans-serif;background:#f1f5f9;color:#1e293b;margin:0;padding:20px}}
  .container{{max-width:900px;margin:auto}}
  .header{{background:#0f172a;color:#fff;border-radius:10px;padding:24px 28px;margin-bottom:20px}}
  .header h1{{margin:0 0 4px;font-size:20px}}
  .header .sub{{color:#94a3b8;font-size:13px}}
  .overall-badge{{display:inline-block;background:{overall_col};color:#fff;padding:5px 16px;border-radius:6px;font-size:14px;font-weight:700;margin-top:10px}}
  .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}}
  .card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;text-align:center}}
  .card .val{{font-size:28px;font-weight:700;color:#0f172a}}
  .card .lbl{{font-size:12px;color:#64748b;margin-top:4px}}
  .table-wrap{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-bottom:20px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#f8fafc;color:#475569;font-weight:600;padding:8px 12px;text-align:left;border-bottom:1px solid #e2e8f0}}
  td{{padding:7px 12px;border-bottom:1px solid #f8fafc;vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700}}
  .badge.pass{{background:#dcfce7;color:#15803d}}
  .badge.fail{{background:#fee2e2;color:#b91c1c}}
  .sec-badge{{background:#dbeafe;color:#1d4ed8;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:700}}
  .section-cards{{margin-top:10px}}
  .section-card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin-bottom:12px}}
  .sec-card-title{{font-weight:700;font-size:13px;color:#0f172a;margin-bottom:8px}}
  .section-card ul{{margin:0;padding-left:18px;font-size:13px;color:#334155;line-height:1.7}}
  .footer{{font-size:11px;color:#94a3b8;text-align:center;margin-top:16px}}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Annex IV Evidence Vault</h1>
    <div class="sub">
      Model: <strong>{self.model_name}</strong> &nbsp;|&nbsp;
      Provider: <strong>{self.provider}</strong> &nbsp;|&nbsp;
      Commit: <code>{self.commit_sha[:8]}</code> &nbsp;|&nbsp;
      Generated: {self._created_at}
    </div>
    <div class="overall-badge">{overall_txt} &mdash; {pct:.0%} pass rate</div>
  </div>

  <div class="cards">
    <div class="card">
      <div class="val">{len(self.entries)}</div>
      <div class="lbl">Evidence entries</div>
    </div>
    <div class="card">
      <div class="val">{len(vault_dict['sections_covered'])}</div>
      <div class="lbl">Annex IV sections covered</div>
    </div>
    <div class="card">
      <div class="val">{summary.get('n_passed', 0)}/{summary.get('n_with_threshold', 0)}</div>
      <div class="lbl">Threshold tests passed</div>
    </div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Section</th><th>Title</th><th>Type</th>
          <th>Articles</th><th>Metric</th><th>Status</th>
        </tr>
      </thead>
      <tbody>{entry_rows}</tbody>
    </table>
  </div>

  <div class="section-cards">
    <h3 style="font-size:15px;margin-bottom:12px">Section Evidence Detail</h3>
    {section_cards}
  </div>

  <div class="footer">
    Generated by <strong>Glassbox AI</strong> v{_VERSION} &mdash;
    {_REGULATION} Annex IV &mdash;
    Use case: {self.use_case}
  </div>
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Private: entry builders
    # ------------------------------------------------------------------

    def _add_general_description(self) -> None:
        self.entries.append(VaultEntry(
            section="§1",
            article_refs=["Article 11", "Article 13"],
            title="General system description",
            description=(
                f"AI system '{self.model_name}' deployed by '{self.provider}' "
                f"for the use case: {self.use_case}. "
                f"Deployment context: {self.deployment_ctx}. "
                "This document constitutes Annex IV technical documentation "
                "per Article 11 of the EU AI Act."
            ),
            evidence_type="general",
            raw={
                "model_name": self.model_name,
                "provider": self.provider,
                "use_case": self.use_case,
                "deployment_ctx": self.deployment_ctx,
            },
        ))

    def _add_from_gb_result(self, result: Dict) -> None:
        faith = result.get("faithfulness", {})
        suff = faith.get("sufficiency", 0.0)
        comp = faith.get("comprehensiveness", 0.0)
        f1 = faith.get("f1", 0.0)
        n_heads = result.get("n_heads", 0)
        circuit = result.get("circuit", {})

        # §2 — sufficiency metric
        self.entries.append(VaultEntry(
            section="§2",
            article_refs=["Article 15(1)", "Article 11"],
            title="Circuit sufficiency (faithfulness metric)",
            description=(
                f"The minimum faithful circuit achieves a sufficiency score of "
                f"{suff:.1%}, meaning the isolated circuit retains {suff:.1%} of "
                "the full model's predictive capacity. Per Article 15(1), this "
                "quantifies the accuracy of the system's explanatory mechanism."
            ),
            evidence_type="faithfulness",
            metric_name="sufficiency",
            metric_value=suff,
            threshold=0.70,
            passed=suff >= 0.70,
            raw={"faithfulness": faith},
        ))

        # §2 — comprehensiveness
        self.entries.append(VaultEntry(
            section="§2",
            article_refs=["Article 15(1)"],
            title="Circuit comprehensiveness (faithfulness metric)",
            description=(
                f"Comprehensiveness score: {comp:.1%}. This measures how much "
                "predictive capacity is lost when the identified circuit is "
                "ablated, confirming circuit necessity."
            ),
            evidence_type="faithfulness",
            metric_name="comprehensiveness",
            metric_value=comp,
            threshold=0.60,
            passed=comp >= 0.60,
            raw={"faithfulness": faith},
        ))

        # §2 — circuit size
        self.entries.append(VaultEntry(
            section="§2",
            article_refs=["Article 13(1)", "Article 11"],
            title=f"Minimum faithful circuit ({n_heads} attention heads)",
            description=(
                f"The minimum faithful circuit consists of {n_heads} attention "
                "heads. A smaller circuit indicates higher interpretability and "
                "facilitates Annex IV §2 documentation of the system's internal "
                "reasoning process."
            ),
            evidence_type="circuit",
            metric_name="n_heads",
            metric_value=float(n_heads),
            raw={"n_heads": n_heads, "circuit": self._truncate_circuit(circuit)},
        ))

        # §4 — f1 score
        self.entries.append(VaultEntry(
            section="§4",
            article_refs=["Article 9", "Article 15(1)"],
            title="Faithfulness F1 score",
            description=(
                f"The harmonic mean of sufficiency and comprehensiveness is "
                f"{f1:.1%}. This combined score is used as the primary compliance "
                "gate in the CI/CD audit pipeline per Article 9 risk management."
            ),
            evidence_type="faithfulness",
            metric_name="f1",
            metric_value=f1,
            threshold=0.65,
            passed=f1 >= 0.65,
            raw={"faithfulness": faith},
        ))

        # §2 — top circuit heads
        if isinstance(circuit, dict) and circuit:
            top_heads = sorted(
                circuit.items(),
                key=lambda x: abs(float(x[1])),
                reverse=True,
            )[:10]
            head_descriptions = [
                f"L{k[0]}H{k[1]}: {float(v):.4f}" if isinstance(k, (list, tuple)) else f"{k}: {float(v):.4f}"
                for k, v in top_heads
            ]
            self.entries.append(VaultEntry(
                section="§2",
                article_refs=["Article 13(1)", "Article 11"],
                title="Top 10 circuit attention heads by importance",
                description=(
                    "The following attention heads contribute most to the model's "
                    "decision, identified via attribution patching: "
                    + "; ".join(head_descriptions) + ". "
                    "This list satisfies Article 11 requirements for documentation "
                    "of the system's relevant computational structures."
                ),
                evidence_type="circuit",
                raw={"top_heads": dict(zip([str(k) for k, _ in top_heads], [float(v) for _, v in top_heads]))},
            ))

    def _add_stability_entries(self, stability_result: Dict) -> None:
        for metric_key, metric_val in stability_result.items():
            if not isinstance(metric_val, (int, float)):
                continue
            self.entries.append(VaultEntry(
                section="§3",
                article_refs=["Article 15(1)", "Article 72"],
                title=f"Stability metric: {metric_key}",
                description=(
                    f"The system's {metric_key} stability score is {metric_val:.4f}. "
                    "Circuit stability across input perturbations is a key indicator "
                    "of robustness per Article 15(1) and supports post-market "
                    "monitoring obligations under Article 72."
                ),
                evidence_type="stability",
                metric_name=metric_key,
                metric_value=float(metric_val),
                threshold=0.70,
                passed=float(metric_val) >= 0.70,
                raw={metric_key: metric_val},
            ))

    def _add_steering_entries(
        self,
        steering_vectors: Dict,
        test_results: Dict[str, Dict],
    ) -> None:
        for concept, sv in steering_vectors.items():
            test = test_results.get(concept)
            sr = test.get("suppression_ratio", None) if test else None
            passed = test.get("passed_threshold", None) if test else None

            self.entries.append(VaultEntry(
                section="§4",
                article_refs=["Article 9(2)(b)", "Article 9(5)", "Article 15(1)"],
                title=f"Steering vector — {concept}",
                description=(
                    f"A steering vector for concept '{concept}' was extracted at "
                    f"layer {getattr(sv, 'layer', 'N/A')} using the "
                    f"{sv.source_info.get('extraction_method', 'N/A') if hasattr(sv, 'source_info') else 'N/A'} "
                    "method (Representation Engineering, Zou et al. 2023). "
                    f"This constitutes a documented risk mitigation measure per "
                    "Article 9(2)(b). "
                    + (
                        f"Suppression test: ratio {sr:.1%}, outcome "
                        f"{'EFFECTIVE' if passed else 'INEFFECTIVE'}."
                        if sr is not None else ""
                    )
                ),
                evidence_type="steering",
                metric_name="suppression_ratio" if sr is not None else None,
                metric_value=sr,
                threshold=0.10 if sr is not None else None,
                passed=passed,
                raw=sv.to_dict() if hasattr(sv, "to_dict") else {"concept": concept},
            ))

    def _add_sae_entries(self, sae_features: List[Dict]) -> None:
        """
        Map SAE (Sparse Autoencoder) feature activations to legal risk
        categories and Annex IV article references.
        """
        _RISK_TO_ARTICLES = {
            "gender_bias":       ["Article 10(2)(f)", "Article 10(5)"],
            "racial_bias":       ["Article 10(2)(f)", "Article 10(5)"],
            "age_bias":          ["Article 10(2)(f)"],
            "disability_bias":   ["Article 10(2)(f)", "Article 10(5)"],
            "religious_bias":    ["Article 10(2)(f)", "Article 10(5)"],
            "nationality_bias":  ["Article 10(2)(f)"],
            "socioeconomic_bias":["Article 10(2)(f)"],
            "toxicity":          ["Article 9", "Article 15(1)"],
            "misinformation":    ["Article 9", "Article 13"],
            "privacy":           ["Article 10", "Article 13(1)"],
            "security":          ["Article 15(1)"],
            "other":             ["Article 9"],
        }

        for feat in sae_features:
            fid = feat.get("feature_id", "unknown")
            act = feat.get("activation", 0.0)
            desc = feat.get("description", "")
            risk_cat = feat.get("legal_risk_category", "other")
            articles = _RISK_TO_ARTICLES.get(risk_cat, ["Article 9"])

            self.entries.append(VaultEntry(
                section="§4",
                article_refs=articles,
                title=f"SAE feature {fid} — {risk_cat}",
                description=(
                    f"Sparse autoencoder feature {fid} activates at magnitude "
                    f"{act:.4f} and is associated with the legal risk category "
                    f"'{risk_cat}'. Feature description: {desc}. "
                    "This feature activation constitutes evidence for the data "
                    "governance documentation required under Article 10(2)(f)."
                ),
                evidence_type="sae_feature",
                metric_name="activation",
                metric_value=float(act),
                raw=feat,
            ))

    def _add_multiagent_entries(self, report: Any) -> None:
        """
        Convert a MultiAgentAudit LiabilityReport (or plain dict) into
        vault entries.
        """
        # Support both LiabilityReport dataclass and raw dict
        if hasattr(report, "chain_risk_level"):
            chain_id = getattr(report, "chain_id", "unknown")
            risk_level = report.chain_risk_level
            most_liable = getattr(report, "most_liable_agent", "unknown")
            annex_text = getattr(report, "annex_iv_text", "")
        elif isinstance(report, dict):
            chain_id = report.get("chain_id", "unknown")
            risk_level = report.get("chain_risk_level", "UNKNOWN")
            most_liable = report.get("most_liable_agent", "unknown")
            annex_text = report.get("annex_iv_text", "")
        else:
            return

        risk_to_articles = {
            "LOW":      ["Article 9"],
            "MEDIUM":   ["Article 9", "Article 10(2)(f)"],
            "HIGH":     ["Article 9", "Article 10(2)(f)", "Article 13(1)"],
            "CRITICAL": ["Article 9", "Article 10(2)(f)", "Article 10(5)", "Article 13(1)"],
        }

        articles = risk_to_articles.get(risk_level, ["Article 9"])
        passed = risk_level in ("LOW", "MEDIUM")

        self.entries.append(VaultEntry(
            section="§4",
            article_refs=articles,
            title=f"Multi-agent chain audit — risk level {risk_level}",
            description=(
                f"Multi-agent causal handoff audit for chain '{chain_id}' "
                f"identifies chain-level risk as {risk_level}. "
                f"Most liable agent: {most_liable}. "
                "Bias contamination and semantic drift were measured across all "
                "agent handoffs. This evidence satisfies Article 9 requirements "
                "for system-level risk management in multi-agent deployments."
            ),
            evidence_type="bias",
            metric_name="chain_risk_level",
            metric_value={"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0, "CRITICAL": 4.0}.get(risk_level, 0.0),
            threshold=2.0,  # LOW or MEDIUM pass
            passed=passed,
            raw={"chain_id": chain_id, "chain_risk_level": risk_level, "most_liable_agent": most_liable},
        ))

        if annex_text:
            self.entries.append(VaultEntry(
                section="§4",
                article_refs=["Article 9", "Article 13(1)"],
                title="Multi-agent Annex IV risk narrative",
                description=annex_text[:500] + ("..." if len(annex_text) > 500 else ""),
                evidence_type="bias",
                raw={"annex_iv_text": annex_text},
            ))

    def _add_standards_entry(self) -> None:
        self.entries.append(VaultEntry(
            section="§6",
            article_refs=["Article 11"],
            title="Technical standards and methodologies applied",
            description=(
                "The following technical methodologies and references are applied: "
                "(1) Attribution patching — Conmy et al., 2023, NeurIPS; "
                "(2) Representation Engineering — Zou et al., 2023, arXiv:2310.01405; "
                "(3) Inference-Time Intervention — Li et al., 2023, arXiv:2306.03341; "
                "(4) Sparse Autoencoders for mechanistic interpretability — Elhage et al., 2022; "
                "(5) EU AI Act harmonised standard ISO/IEC 42001:2023 (AI Management Systems). "
                "No harmonised standard has been formally adopted for mechanistic interpretability; "
                "these references constitute applicable technical specifications."
            ),
            evidence_type="general",
            raw={
                "references": [
                    "Conmy et al. 2023 — Towards Automated Circuit Discovery for Mechanistic Interpretability",
                    "Zou et al. 2023 — Representation Engineering",
                    "Li et al. 2023 — Inference-Time Intervention",
                    "Elhage et al. 2022 — Toy Models of Superposition",
                    "ISO/IEC 42001:2023",
                ]
            },
        ))

    def _add_conformity_entry(self) -> None:
        self.entries.append(VaultEntry(
            section="§7",
            article_refs=["Article 11"],
            title="EU Declaration of Conformity (placeholder)",
            description=(
                f"Provider '{self.provider}' confirms that this technical "
                "documentation constitutes part of the EU Declaration of "
                f"Conformity for AI system '{self.model_name}' v{self.version} "
                f"(commit {self.commit_sha[:8]}), per Article 11 and Annex IV. "
                "The full signed declaration must be completed by an authorised "
                "representative before market placement."
            ),
            evidence_type="general",
            raw={
                "provider": self.provider,
                "model_name": self.model_name,
                "version": self.version,
                "commit_sha": self.commit_sha,
                "status": "PLACEHOLDER — requires signed declaration",
            },
        ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compliance_summary(self) -> Dict:
        with_threshold = [e for e in self.entries if e.passed is not None]
        n_passed = sum(1 for e in with_threshold if e.passed)
        n_failed = len(with_threshold) - n_passed
        pass_rate = n_passed / len(with_threshold) if with_threshold else 1.0

        failed_titles = [e.title for e in with_threshold if not e.passed]

        return {
            "n_entries": len(self.entries),
            "n_with_threshold": len(with_threshold),
            "n_passed": n_passed,
            "n_failed": n_failed,
            "pass_rate": round(pass_rate, 4),
            "overall_status": "COMPLIANT" if pass_rate >= 0.80 else (
                "MARGINAL" if pass_rate >= 0.50 else "NON-COMPLIANT"
            ),
            "failed_items": failed_titles,
        }

    @staticmethod
    def _truncate_circuit(circuit: Any, max_items: int = 20) -> Any:
        """Keep the dict to max_items entries for storage efficiency."""
        if isinstance(circuit, dict) and len(circuit) > max_items:
            items = sorted(
                circuit.items(),
                key=lambda x: abs(float(x[1])),
                reverse=True,
            )[:max_items]
            return {str(k): float(v) for k, v in items}
        return circuit


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_annex_iv_vault(
    gb_result: Optional[Dict] = None,
    model_name: str = "unknown",
    provider: str = "Unknown",
    use_case: str = "CI/CD compliance audit",
    deployment_ctx: str = "other_high_risk",
    commit_sha: str = "unknown",
    multiagent_report: Optional[Any] = None,
    steering_vectors: Optional[Dict] = None,
    steering_test_results: Optional[Dict[str, Dict]] = None,
    sae_features: Optional[List[Dict]] = None,
    stability_result: Optional[Dict] = None,
    output_json: Optional[str] = None,
    output_html: Optional[str] = None,
) -> AnnexIVEvidenceVault:
    """
    One-liner: build the full Annex IV evidence vault and optionally save
    JSON and HTML reports.

    Example
    -------
    >>> vault = build_annex_iv_vault(
    ...     gb_result=result,
    ...     model_name="gpt2",
    ...     provider="Acme Corp",
    ...     output_json="reports/annex-iv.json",
    ...     output_html="reports/annex-iv.html",
    ... )
    >>> print(vault.to_dict()["compliance_summary"])
    """
    vault = AnnexIVEvidenceVault(
        model_name=model_name,
        provider=provider,
        use_case=use_case,
        deployment_ctx=deployment_ctx,
        commit_sha=commit_sha,
    )
    vault.build_vault(
        gb_result=gb_result,
        multiagent_report=multiagent_report,
        steering_vectors=steering_vectors,
        steering_test_results=steering_test_results,
        sae_features=sae_features,
        stability_result=stability_result,
    )

    if output_json:
        vault.save_json(output_json)

    if output_html:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
        with open(output_html, "w") as fh:
            fh.write(vault.to_html())

    return vault
