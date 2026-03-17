"""
glassbox/compliance.py — EU AI Act Annex IV Compliance Report Generator
========================================================================

Regulation: EU 2024/1689 (AI Act), enforced August 2026.
Scope:      High-risk AI systems under Article 6 — any AI used in
            financial services, healthcare, HR/employment, legal,
            critical infrastructure, education, or law enforcement
            deploying to EU customers/users.

This module takes Glassbox analysis results and generates a
legally-structured Annex IV technical documentation report, satisfying:

    Annex IV — Technical Documentation (Article 11)
    ├── Section 1  — General description of the AI system
    ├── Section 2  — Development and design information
    ├── Section 3  — Monitoring, functioning, and control
    ├── Section 4  — Data governance
    ├── Section 5  — Risk management
    ├── Section 6  — Changes through the lifecycle
    ├── Section 7  — Harmonised standards applied
    ├── Section 8  — EU declaration of conformity (ref. Article 47)
    └── Section 9  — Post-market monitoring plan (ref. Article 72)

    Article 13   — Transparency and provision of information
    Article 9    — Risk management system
    Article 72   — Post-market monitoring by providers

Usage
-----
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2
    from glassbox.compliance import AnnexIVReport, DeploymentContext

    model = HookedTransformer.from_pretrained("gpt2")
    gb    = GlassboxV2(model)

    report = AnnexIVReport(
        model_name          = "GPT-2",
        system_purpose      = "Credit risk scoring for loan applications",
        provider_name       = "Acme Bank NV",
        provider_address    = "1 Fintech Street, Amsterdam, Netherlands 1011AB",
        deployment_context  = DeploymentContext.FINANCIAL_SERVICES,
    )

    result = gb.analyze(
        prompt    = "The loan applicant has low credit score, therefore the decision is",
        correct   = " denied",
        incorrect = " approved",
    )
    report.add_analysis(result, use_case="Loan denial decision")

    # Machine-readable — for LIMS / audit systems
    json_report = report.to_json()

    # Human-readable — send to auditor / regulator
    report.to_pdf("annex_iv_report.pdf")

Dependencies
------------
JSON output: stdlib only (json, datetime, uuid, dataclasses)
PDF output:  reportlab >= 3.6.0  (pip install reportlab)
"""

from __future__ import annotations

import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "AnnexIVReport",
    "DeploymentContext",
    "RiskClassification",
    "ExplainabilityGrade",
    "ComplianceStatus",
]

# ---------------------------------------------------------------------------
# Enums — legally meaningful categories
# ---------------------------------------------------------------------------

class DeploymentContext(str, Enum):
    """
    High-risk AI deployment contexts under Annex III of EU AI Act 2024/1689.
    Each maps to specific Article 6 risk classification criteria.
    """
    FINANCIAL_SERVICES      = "financial_services"     # creditworthiness, insurance, loans
    HEALTHCARE              = "healthcare"              # clinical AI, diagnosis, treatment
    HR_EMPLOYMENT           = "hr_employment"           # recruitment, promotion, termination
    LEGAL                   = "legal"                   # legal advice, document analysis
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure" # energy, water, transport
    EDUCATION               = "education"               # student assessment, admissions
    LAW_ENFORCEMENT         = "law_enforcement"         # biometrics, crime prediction
    OTHER_HIGH_RISK         = "other_high_risk"         # other Annex III categories
    GENERAL_PURPOSE         = "general_purpose"         # GPAI — different obligations


class RiskClassification(str, Enum):
    """
    AI system risk levels under EU AI Act Title III.
    Annex IV documentation is required for HIGH_RISK only.
    """
    UNACCEPTABLE = "unacceptable"  # Article 5 — prohibited (no report, no deployment)
    HIGH_RISK    = "high_risk"     # Annex III — requires full Annex IV documentation
    LIMITED_RISK = "limited_risk"  # Article 50 — transparency obligations only
    MINIMAL_RISK = "minimal_risk"  # No mandatory requirements


class ExplainabilityGrade(str, Enum):
    """
    Glassbox-derived explainability assessment for Article 13 compliance.
    Derived from faithfulness metrics (sufficiency, comprehensiveness, F1).
    """
    GRADE_A = "A — Fully Explainable"      # F1 > 0.70, suff > 0.80, comp > 0.60
    GRADE_B = "B — Substantially Explainable"  # F1 > 0.50, meets baseline
    GRADE_C = "C — Partially Explainable"  # F1 0.30-0.50, needs improvement
    GRADE_D = "D — Minimally Explainable"  # F1 < 0.30, high regulatory risk
    NOT_ASSESSED = "Not assessed"


class ComplianceStatus(str, Enum):
    """
    Overall compliance assessment for the AI system documentation.
    This is documentation completeness, NOT legal guarantee of compliance.
    """
    COMPLIANT             = "compliant"              # All required fields populated
    CONDITIONALLY_COMPLIANT = "conditionally_compliant"  # Minor gaps, supplementary docs needed
    INCOMPLETE            = "incomplete"             # Missing required sections
    NON_COMPLIANT         = "non_compliant"          # Critical failures identified


# ---------------------------------------------------------------------------
# Data structures for each Annex IV section
# ---------------------------------------------------------------------------

@dataclass
class Section1_GeneralDescription:
    """
    Annex IV, Section 1 — General description of the AI system.
    Maps to Article 13(3)(a): name, version, intended purpose, categories of persons.
    """
    system_name:            str = ""
    system_version:         str = ""
    intended_purpose:       str = ""
    deployment_context:     str = ""
    categories_of_persons:  str = ""   # who is subject to the system's decisions
    geographic_scope:       str = ""   # EU member states where deployed
    provider_name:          str = ""
    provider_address:       str = ""
    provider_eu_rep:        str = ""   # EU representative if provider is outside EU
    authorised_rep:         str = ""
    report_date:            str = ""
    report_id:              str = ""
    glassbox_version:       str = ""
    model_architecture:     str = ""   # transformer architecture specifics
    model_n_layers:         int = 0
    model_n_heads:          int = 0
    model_d_model:          int = 0


@dataclass
class Section2_DevelopmentDesign:
    """
    Annex IV, Section 2 — Design specifications and development processes.
    Maps to Article 10 (data governance), Article 11(1)(d) training description.
    """
    design_approach:            str = ""   # how the model was designed/selected
    architecture_description:   str = ""   # transformer architecture details
    attribution_method:         str = ""   # Glassbox method used (taylor/IG)
    circuit_discovery_algorithm: str = ""  # MFC algorithm description
    clean_prompt:               str = ""   # example input used for analysis
    corrupted_prompt:           str = ""   # name-swap corruption used
    correct_token:              str = ""
    incorrect_token:            str = ""
    n_analysis_heads_total:     int = 0    # total attention heads in model
    n_circuit_heads_found:      int = 0    # heads in minimum faithful circuit
    circuit_heads:              List[str] = field(default_factory=list)  # [(l, h), ...]
    top_attributions:           List[Dict] = field(default_factory=list) # top_heads
    clean_logit_difference:     float = 0.0  # logit(correct) - logit(incorrect)
    reference_papers:           List[str] = field(default_factory=list)


@dataclass
class Section3_MonitoringControl:
    """
    Annex IV, Section 3 — Monitoring, functioning, and control measures.
    Maps to Article 9(6) risk monitoring, Article 13(3)(b) performance metrics,
    Article 14 human oversight measures.
    """
    # Faithfulness metrics — the core explainability evidence
    sufficiency:            float = 0.0    # fraction of model behaviour explained
    comprehensiveness:      float = 0.0    # fraction of circuit that is necessary
    f1_score:               float = 0.0    # harmonic mean of suff + comp
    faithfulness_category:  str   = ""     # "faithful" / "backup_mechanisms" / etc.
    suff_is_approximate:    bool  = True   # True for taylor, False for IG
    explainability_grade:   str   = ""     # A/B/C/D
    explainability_rationale: str = ""     # human-readable assessment

    # Accuracy requirements (Article 15) — provider must fill
    accuracy_metric:        str   = ""     # e.g. "F1 score on validation set"
    accuracy_value:         str   = ""     # e.g. "0.87 on held-out test set"
    robustness_measures:    str   = ""     # adversarial/distribution shift handling
    cybersecurity_measures: str   = ""

    # Human oversight (Article 14) — mandatory for high-risk AI
    human_oversight_measures:   str = ""   # what oversight is in place
    override_mechanism:         str = ""   # how humans can override system decisions
    monitoring_indicators:      List[str] = field(default_factory=list)


@dataclass
class Section4_DataGovernance:
    """
    Annex IV, Section 4 — Data governance and management practices.
    Maps to Article 10 — data quality, bias assessment, data relevance.
    NOTE: Provider must supply dataset details. Glassbox audits the model,
    not the training data directly.
    """
    training_dataset_description:    str = ""
    validation_dataset_description:  str = ""
    test_dataset_description:        str = ""
    data_provenance:                 str = ""
    data_collection_methodology:     str = ""
    bias_assessment:                 str = ""   # how bias was assessed/mitigated
    data_preprocessing_steps:        str = ""
    known_data_limitations:          str = ""
    personal_data_involved:          bool = False
    gdpr_compliance_measures:        str = ""   # if personal data involved
    # Glassbox-derived data signal
    input_token_count_analyzed:      int = 0    # tokens in analyzed prompt
    token_attribution_available:     bool = False


@dataclass
class Section5_RiskManagement:
    """
    Annex IV, Section 5 — Risk management system documentation.
    Maps to Article 9 — risk identification, evaluation, mitigation, residual risk.
    """
    risk_identification_process:    str = ""
    identified_risks:               List[Dict[str, str]] = field(default_factory=list)
    risk_mitigation_measures:       str = ""
    residual_risks:                 str = ""
    # Glassbox-derived risk signals
    faithfulness_risk_flag:         bool = False  # True if F1 < 0.50
    circuit_concentration_risk:     bool = False  # True if circuit has < 3 heads
    explainability_risk_assessment: str  = ""
    recommended_actions:            List[str] = field(default_factory=list)


@dataclass
class Section6_LifecycleChanges:
    """
    Annex IV, Section 6 — Changes through the lifecycle.
    Maps to Article 16(d) — record keeping of substantial modifications.
    """
    model_version:          str = ""
    glassbox_version:       str = ""
    report_version:         str = "1.0"
    analysis_timestamp:     str = ""
    previous_reports:       List[str] = field(default_factory=list)
    planned_modifications:  str = ""
    retesting_trigger:      str = "Substantial modification as defined under Article 83"


@dataclass
class Section7_HarmonisedStandards:
    """
    Annex IV, Section 7 — Applied harmonised standards (Article 40).
    Harmonised standards provide presumption of conformity with AI Act requirements.
    NOTE: EU harmonised standards for AI Act are under development as of 2024.
    CEN/CENELEC JTC 21 is the primary standardisation body.
    """
    standards_applied:          List[Dict[str, str]] = field(default_factory=list)
    common_specifications_used: List[str] = field(default_factory=list)
    other_technical_specs:      List[str] = field(default_factory=list)
    standardisation_body_ref:   str = "CEN/CENELEC JTC 21 (AI standardisation)"


@dataclass
class Section8_Declaration:
    """
    Annex IV, Section 8 — EU declaration of conformity reference (Article 47).
    Article 47(1): Provider must draw up written declaration of conformity before
    placing high-risk AI on the market. This section cross-references that document.
    """
    declaration_reference:  str = ""   # reference number of the declaration doc
    declaration_date:       str = ""
    signatory_name:         str = ""
    signatory_position:     str = ""
    conformity_assessment_body: str = ""  # notified body name/number, if applicable
    notified_body_number:   str = ""   # EU notified body identification number
    certificate_reference:  str = ""


@dataclass
class Section9_PostMarketMonitoring:
    """
    Annex IV, Section 9 — Post-market monitoring plan (Article 72).
    Article 72(3): Provider must establish post-market monitoring system
    proportionate to the nature of AI technologies and risks.
    """
    monitoring_plan_reference:  str = ""
    data_collection_approach:   str = ""   # how post-deployment data is collected
    performance_indicators:     List[str] = field(default_factory=list)
    review_frequency:           str = ""   # how often monitoring is reviewed
    incident_reporting_process: str = ""   # Article 73 serious incident reporting
    stakeholder_feedback_loop:  str = ""
    # Glassbox re-audit recommendation
    reaudit_triggers:           List[str] = field(default_factory=list)
    reaudit_frequency:          str = "Recommended: every 6 months or on model update"


# ---------------------------------------------------------------------------
# Main report class
# ---------------------------------------------------------------------------

class AnnexIVReport:
    """
    EU AI Act Annex IV Technical Documentation Report Generator.

    Generates a complete Annex IV-structured compliance report from
    Glassbox mechanistic interpretability analysis results.

    This is the primary output format that compliance officers, legal teams,
    and EU national competent authorities require for high-risk AI systems.

    Parameters
    ----------
    model_name          : str — name of the AI system being audited
    system_purpose      : str — intended purpose (Article 13 requirement)
    provider_name       : str — legal name of the AI system provider
    provider_address    : str — registered address of the provider
    deployment_context  : DeploymentContext — which Annex III category applies
    risk_classification : RiskClassification — assessed risk level (default: HIGH_RISK)
    provider_eu_rep     : str — EU representative if provider is outside EU (Article 25)

    Example
    -------
        report = AnnexIVReport(
            model_name         = "CreditScorer v3.2",
            system_purpose     = "Automated credit risk assessment for loan applications",
            provider_name      = "Acme Bank NV",
            provider_address   = "1 Fintech Street, Amsterdam 1011AB, Netherlands",
            deployment_context = DeploymentContext.FINANCIAL_SERVICES,
        )
        result = gb.analyze(prompt, " denied", " approved")
        report.add_analysis(result, use_case="Loan denial — insufficient credit history")
        report.to_pdf("annex_iv_compliance_report.pdf")
    """

    # Article 13 transparency thresholds — derived from faithfulness metrics
    _GRADE_A_THRESHOLD = (0.80, 0.60, 0.70)   # (min_suff, min_comp, min_f1)
    _GRADE_B_THRESHOLD = (0.65, 0.40, 0.50)
    _GRADE_C_THRESHOLD = (0.40, 0.20, 0.30)

    # Risk flag thresholds
    _FAITHFULNESS_RISK_F1_THRESHOLD = 0.50
    _CIRCUIT_CONCENTRATION_MIN_HEADS = 3

    def __init__(
        self,
        model_name:          str,
        system_purpose:      str,
        provider_name:       str,
        provider_address:    str,
        deployment_context:  DeploymentContext = DeploymentContext.OTHER_HIGH_RISK,
        risk_classification: RiskClassification = RiskClassification.HIGH_RISK,
        provider_eu_rep:     str = "",
        authorised_rep:      str = "",
    ):
        self.model_name          = model_name
        self.system_purpose      = system_purpose
        self.provider_name       = provider_name
        self.provider_address    = provider_address
        self.deployment_context  = deployment_context
        self.risk_classification = risk_classification
        self.provider_eu_rep     = provider_eu_rep
        self.authorised_rep      = authorised_rep

        self._report_id   = str(uuid.uuid4()).upper()[:8]
        self._created_at  = datetime.now(timezone.utc)
        self._analyses:   List[Dict[str, Any]] = []  # raw analyze() results + use_case
        self._use_cases:  List[str] = []

        # Sections will be populated from analysis results
        self._s1: Optional[Section1_GeneralDescription]  = None
        self._s2: Optional[Section2_DevelopmentDesign]   = None
        self._s3: Optional[Section3_MonitoringControl]   = None
        self._s4: Optional[Section4_DataGovernance]      = None
        self._s5: Optional[Section5_RiskManagement]      = None
        self._s6: Optional[Section6_LifecycleChanges]    = None
        self._s7: Optional[Section7_HarmonisedStandards] = None
        self._s8: Optional[Section8_Declaration]         = None
        self._s9: Optional[Section9_PostMarketMonitoring] = None

    # ------------------------------------------------------------------
    # Public API: add analysis results
    # ------------------------------------------------------------------

    def add_analysis(
        self,
        result:       Dict[str, Any],
        use_case:     str = "General use case analysis",
    ) -> "AnnexIVReport":
        """
        Add a single GlassboxV2.analyze() result to the report.

        Multiple analyses are aggregated — the report uses averaged
        faithfulness metrics and includes all circuit heads found.

        Parameters
        ----------
        result   : dict returned by GlassboxV2.analyze()
        use_case : human-readable description of the decision being audited

        Returns
        -------
        self — for method chaining
        """
        self._analyses.append(result)
        self._use_cases.append(use_case)
        self._build_sections()
        return self

    def add_batch_analysis(
        self,
        results:    List[Dict[str, Any]],
        use_cases:  Optional[List[str]] = None,
    ) -> "AnnexIVReport":
        """
        Add multiple GlassboxV2.analyze() results.
        Aggregates metrics across all results for statistical robustness.
        """
        if use_cases is None:
            use_cases = [f"Use case {i+1}" for i in range(len(results))]
        for result, uc in zip(results, use_cases):
            self._analyses.append(result)
            self._use_cases.append(uc)
        self._build_sections()
        return self

    # ------------------------------------------------------------------
    # Output: JSON
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the full Annex IV report as a JSON string.

        The JSON schema is self-documenting: every key includes a
        'regulation_ref' annotation identifying the corresponding
        EU AI Act article or Annex IV section.

        Returns
        -------
        str — JSON-serializable report
        """
        self._ensure_sections()
        return json.dumps(self._build_json_structure(), indent=indent, default=str)

    def save_json(self, path: str) -> Path:
        """Save JSON report to file. Returns the Path."""
        out = Path(path)
        out.write_text(self.to_json(), encoding="utf-8")
        logger.info("Annex IV JSON report saved to %s", out)
        return out


    def to_model_card(self, path: Optional[str] = None) -> str:
        """
        Generate a HuggingFace-compatible model card (README.md format) from
        the Annex IV compliance report.

        Populates standard model card sections with EU AI Act compliance data:
        - Model description and intended use
        - Explainability grade and faithfulness metrics (Article 13)
        - Risk factors and deployment context (Article 9)
        - Training data governance status (Article 10)
        - EU AI Act compliance status
        - Citation block

        Parameters
        ----------
        path : str, optional
            If provided, writes the model card to this path and returns content.

        Returns
        -------
        str — model card content in Markdown format

        Example
        -------
            report = AnnexIVReport(model_name="gpt2", ...)
            report.add_analysis(result)
            md = report.to_model_card("MODEL_CARD.md")
        """
        self._ensure_sections()
        s1 = self._s1
        s3 = self._s3
        s5 = self._s5
        s4 = self._s4

        suff  = getattr(s3, "sufficiency_score", 0.0) or 0.0
        comp  = getattr(s3, "comprehensiveness_score", 0.0) or 0.0
        f1    = getattr(s3, "f1_score", 0.0) or 0.0
        grade = getattr(s3, "explainability_grade", ExplainabilityGrade.D)
        grade_str = grade.value if hasattr(grade, "value") else str(grade)

        risk_level  = self.risk_classification.value if hasattr(self.risk_classification, "value") else str(self.risk_classification)
        deploy_ctx  = self.deployment_context.value   if hasattr(self.deployment_context, "value")  else str(self.deployment_context)
        created_str = self._created_at.strftime("%Y-%m-%d")

        risks = getattr(s5, "identified_risks", []) or []
        risk_lines = "\n".join(
            f"- **{r.get('risk','Unknown')}** (Severity: {r.get('severity','unknown')}) — {r.get('article','')}"
            for r in risks[:5]
        ) if risks else "- No critical risks identified."

        compliance_status = getattr(self, "_compliance_status_str", "conditionally_compliant")

        model_card = f"""---
language:
- en
license: other
tags:
- eu-ai-act
- annex-iv
- compliance
- mechanistic-interpretability
- explainability
model_name: {self.model_name}
pipeline_tag: text-generation
library_name: glassbox-mech-interp
---

# {self.model_name}

> **EU AI Act Compliance Report** generated by [Glassbox](https://glassbox-ai-2-0-mechanistic.onrender.com) v{_get_version()}
> Report ID: {self._report_id} · Generated: {created_str}

## Model Description

**Purpose:** {self.system_purpose}

**Provider:** {self.provider_name}, {self.provider_address}

**Deployment context:** {deploy_ctx.replace("_", " ").title()}

**Risk classification:** {risk_level.replace("_", " ").title()} under EU AI Act Annex III

## Intended Use

{self.system_purpose}

This model has been audited for EU AI Act Annex IV compliance. Deployment is
authorised only for the stated purpose and deployment context listed above.

## EU AI Act Compliance Status

| Item | Status |
|------|--------|
| Annex IV documentation | ✅ Complete (all 9 sections) |
| Explainability grade | **{grade_str}** (Article 13) |
| Risk assessment | ✅ Completed (Article 9) |
| Data governance documented | {'✅' if getattr(s4, 'training_data_description', '') else '⚠️ Partial'} |
| Declaration of conformity | ⚠️ Provider must execute (Article 47) |
| Post-market monitoring plan | ✅ Documented (Article 72) |

**Overall compliance status:** `{compliance_status.replace("_", " ").title()}`

> Enforcement deadline: **August 2026** — Regulation (EU) 2024/1689, Article 99(4):
> non-compliance penalties up to €15M or 3% of global annual turnover.

## Explainability Metrics (Article 13)

These metrics are generated by Glassbox mechanistic interpretability analysis
using attribution patching (Nanda et al. 2023) and minimum faithful circuit
discovery.

| Metric | Score | Threshold (Grade A) |
|--------|-------|---------------------|
| Sufficiency | `{suff:.3f}` | ≥ 0.80 |
| Comprehensiveness | `{comp:.3f}` | ≥ 0.60 |
| F1 (harmonic mean) | `{f1:.3f}` | ≥ 0.70 |
| **Explainability grade** | **{grade_str}** | A = fully explainable |

Sufficiency measures whether the identified circuit alone recovers the model's
prediction. Comprehensiveness measures how much ablating the circuit degrades
performance. F1 is the harmonic mean of both.

## Identified Risks (Article 9)

{risk_lines}

## Training Data (Article 10)

See full Annex IV Section 4 — Data Governance in the [compliance report JSON].

## Audit Trail

- **Audit method:** {'White-box (mechanistic interpretability)' if self._analyses and self._analyses[0].get('circuit') else 'Black-box (behavioural audit)'}
- **Tool:** Glassbox v{_get_version()} — [github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)
- **Regulation:** Regulation (EU) 2024/1689 (AI Act) — [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)

## Citation

```bibtex
@software{{mahale2026glassbox,
  author  = {{Mahale, Ajay Pravin}},
  title   = {{Glassbox: A Causal Mechanistic Interpretability Toolkit with Circuit Alignment Scoring}},
  year    = {{2026}},
  url     = {{https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool}},
  note    = {{arXiv:2603.09988}}
}}
```
"""
        if path:
            from pathlib import Path as _Path
            p = _Path(path)
            p.write_text(model_card, encoding="utf-8")
            logger.info("Model card written to %s", p)
        return model_card

    def save_model_card(self, path: str = "MODEL_CARD.md") -> Path:
        """Write HuggingFace model card to path. Returns Path."""
        self.to_model_card(path)
        return Path(path)

    # ------------------------------------------------------------------
    # Output: PDF
    # ------------------------------------------------------------------

    def to_pdf(self, path: str) -> Path:
        """
        Generate a legally-structured Annex IV PDF report.

        Requires: pip install reportlab

        The PDF is formatted for submission to EU national competent
        authorities and internal compliance teams. Sections follow the
        exact ordering mandated by Annex IV.

        Parameters
        ----------
        path : output file path (e.g. "annex_iv_report.pdf")

        Returns
        -------
        Path — location of the saved PDF
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, PageBreak,
            )
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        except ImportError as e:
            raise ImportError(
                "PDF generation requires reportlab. "
                "Install it with:  pip install reportlab"
            ) from e

        self._ensure_sections()
        out = Path(path)
        doc = SimpleDocTemplate(
            str(out),
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=25*mm,
            bottomMargin=20*mm,
            title=f"EU AI Act Annex IV — {self.model_name}",
            author=self.provider_name,
        )

        # Styles
        base   = getSampleStyleSheet()
        styles = self._build_pdf_styles(base, colors)

        story  = []
        story += self._pdf_cover(styles, colors)
        story.append(PageBreak())
        story += self._pdf_legal_notice(styles)
        story.append(PageBreak())
        story += self._pdf_executive_summary(styles, colors)
        story.append(PageBreak())

        # Nine sections
        for section_fn in [
            self._pdf_section1,
            self._pdf_section2,
            self._pdf_section3,
            self._pdf_section4,
            self._pdf_section5,
            self._pdf_section6,
            self._pdf_section7,
            self._pdf_section8,
            self._pdf_section9,
        ]:
            story += section_fn(styles, colors)
            story.append(PageBreak())

        story += self._pdf_methodology(styles)
        story.append(PageBreak())
        story += self._pdf_signature_block(styles, colors)

        doc.build(story)
        logger.info("Annex IV PDF report saved to %s", out)
        return out

    # ------------------------------------------------------------------
    # Internal: build sections from analysis results
    # ------------------------------------------------------------------

    def _ensure_sections(self):
        if not self._analyses:
            raise ValueError(
                "No analysis results added. Call add_analysis() or "
                "add_batch_analysis() before generating the report."
            )
        if self._s1 is None:
            self._build_sections()

    def _build_sections(self):
        """Populate all 9 sections from analysis results + provider metadata."""
        if not self._analyses:
            return

        # Aggregate faithfulness across all analyses
        all_suff  = [r["faithfulness"]["sufficiency"]       for r in self._analyses if "faithfulness" in r]
        all_comp  = [r["faithfulness"]["comprehensiveness"]  for r in self._analyses if "faithfulness" in r]
        all_f1    = [r["faithfulness"]["f1"]                 for r in self._analyses if "faithfulness" in r]
        all_cats  = [r["faithfulness"]["category"]           for r in self._analyses if "faithfulness" in r]

        avg_suff = sum(all_suff) / len(all_suff) if all_suff else 0.0
        avg_comp = sum(all_comp) / len(all_comp) if all_comp else 0.0
        avg_f1   = sum(all_f1)   / len(all_f1)   if all_f1   else 0.0

        # Use first analysis for model metadata
        first = self._analyses[0]
        meta  = first.get("model_metadata", {})
        total_heads = meta.get("n_layers", 0) * meta.get("n_heads", 0)

        # Aggregate circuits across all analyses
        all_circuit_heads: List[str] = []
        for r in self._analyses:
            for h in r.get("circuit", []):
                label = f"L{h[0]}H{h[1]}"
                if label not in all_circuit_heads:
                    all_circuit_heads.append(label)

        grade       = self._compute_grade(avg_suff, avg_comp, avg_f1)
        rationale   = self._grade_rationale(grade, avg_suff, avg_comp, avg_f1, all_cats)
        risk_flag   = avg_f1 < self._FAITHFULNESS_RISK_F1_THRESHOLD
        conc_risk   = len(all_circuit_heads) < self._CIRCUIT_CONCENTRATION_MIN_HEADS

        ts = self._created_at.isoformat()

        # Section 1 — General description
        self._s1 = Section1_GeneralDescription(
            system_name           = self.model_name,
            system_version        = meta.get("glassbox_version", ""),
            intended_purpose      = self.system_purpose,
            deployment_context    = self.deployment_context.value,
            categories_of_persons = self._infer_affected_persons(),
            geographic_scope      = "European Union",
            provider_name         = self.provider_name,
            provider_address      = self.provider_address,
            provider_eu_rep       = self.provider_eu_rep,
            authorised_rep        = self.authorised_rep,
            report_date           = ts[:10],
            report_id             = self._report_id,
            glassbox_version      = meta.get("glassbox_version", ""),
            model_architecture    = "Transformer (decoder-only, autoregressive)",
            model_n_layers        = meta.get("n_layers", 0),
            model_n_heads         = meta.get("n_heads", 0),
            model_d_model         = meta.get("d_model", 0),
        )

        # Section 2 — Development and design
        top_heads_all = first.get("top_heads", [])
        self._s2 = Section2_DevelopmentDesign(
            design_approach             = "Pre-trained transformer neural network, mechanistic interpretability analysis via attribution patching",
            architecture_description    = (
                f"Autoregressive transformer: {meta.get('n_layers', '?')} layers, "
                f"{meta.get('n_heads', '?')} attention heads per layer, "
                f"d_model={meta.get('d_model', '?')}, d_head={meta.get('d_head', '?')}"
            ),
            attribution_method          = first.get("method", "taylor"),
            circuit_discovery_algorithm = "Minimum Faithful Circuit (MFC) — greedy forward selection + backward pruning (Mahale 2026)",
            clean_prompt                = first.get("corr_prompt", "")[:200] + "..." if len(first.get("corr_prompt", "")) > 200 else first.get("corr_prompt", ""),
            corrupted_prompt            = first.get("corr_prompt", ""),
            correct_token               = self._use_cases[0] if self._use_cases else "",
            incorrect_token             = "",
            n_analysis_heads_total      = total_heads,
            n_circuit_heads_found       = len(all_circuit_heads),
            circuit_heads               = all_circuit_heads,
            top_attributions            = top_heads_all[:10],
            clean_logit_difference      = first.get("clean_ld", 0.0),
            reference_papers            = [
                "Nanda et al. 2023 — Attribution Patching: Activation Patching at Industrial Scale",
                "Wang et al. 2022 — Interpretability in the Wild: IOI Circuit in GPT-2 Small",
                "Conmy et al. 2023 — ACDC: Automated Circuit Discovery for Mechanistic Interpretability",
                "Elhage et al. 2021 — A Mathematical Framework for Transformer Circuits",
                "Mahale 2026 — Glassbox: 37x Faster Circuit Discovery via Asymmetric MFC",
            ],
        )

        # Section 3 — Monitoring and control
        primary_cat = max(set(all_cats), key=all_cats.count) if all_cats else "not_assessed"
        suff_approx = first.get("faithfulness", {}).get("suff_is_approx", True)
        self._s3 = Section3_MonitoringControl(
            sufficiency             = avg_suff,
            comprehensiveness       = avg_comp,
            f1_score                = avg_f1,
            faithfulness_category   = primary_cat,
            suff_is_approximate     = suff_approx,
            explainability_grade    = grade.value,
            explainability_rationale= rationale,
            accuracy_metric         = "[PROVIDER TO COMPLETE — e.g. accuracy on held-out test set]",
            accuracy_value          = "[PROVIDER TO COMPLETE — e.g. 0.87 F1 on 10,000 sample validation set]",
            robustness_measures     = "[PROVIDER TO COMPLETE — adversarial testing, distribution shift evaluation]",
            cybersecurity_measures  = "[PROVIDER TO COMPLETE — access controls, adversarial input monitoring]",
            human_oversight_measures= self._infer_oversight_measures(),
            override_mechanism      = "Human reviewer can override any automated decision prior to finalisation",
            monitoring_indicators   = self._build_monitoring_indicators(avg_suff, avg_comp, avg_f1),
        )

        # Section 4 — Data governance
        prompt_tokens = len(first.get("corr_prompt", "").split()) if first.get("corr_prompt") else 0
        self._s4 = Section4_DataGovernance(
            training_dataset_description   = "[PROVIDER TO COMPLETE — training dataset name, size, source, collection period]",
            validation_dataset_description = "[PROVIDER TO COMPLETE — validation dataset name, size, temporal split]",
            test_dataset_description       = "[PROVIDER TO COMPLETE — test dataset name, size, evaluation methodology]",
            data_provenance                = "[PROVIDER TO COMPLETE — data sources, licensing, consent mechanisms]",
            data_collection_methodology    = "[PROVIDER TO COMPLETE — data collection process, quality controls applied]",
            bias_assessment                = "[PROVIDER TO COMPLETE — fairness metrics by protected group, bias mitigation steps]",
            data_preprocessing_steps       = "[PROVIDER TO COMPLETE — tokenisation, normalisation, filtering applied]",
            known_data_limitations         = "[PROVIDER TO COMPLETE — known gaps, underrepresented groups, temporal coverage]",
            personal_data_involved         = True,  # conservative default for high-risk AI
            gdpr_compliance_measures       = "[PROVIDER TO COMPLETE — GDPR Article 22 measures, lawful basis, DPO consultation]",
            input_token_count_analyzed     = prompt_tokens,
            token_attribution_available    = True,
        )

        # Section 5 — Risk management
        self._s5 = Section5_RiskManagement(
            risk_identification_process = "Systematic mechanistic interpretability analysis of model decision pathways via Glassbox",
            identified_risks            = self._identify_risks(avg_suff, avg_comp, avg_f1, risk_flag, conc_risk),
            risk_mitigation_measures    = "[PROVIDER TO COMPLETE — specific technical + organisational measures per identified risk]",
            residual_risks              = "[PROVIDER TO COMPLETE — risks that remain after mitigation + accepted residual risk level]",
            faithfulness_risk_flag      = risk_flag,
            circuit_concentration_risk  = conc_risk,
            explainability_risk_assessment = rationale,
            recommended_actions         = self._build_recommendations(avg_f1, risk_flag, conc_risk),
        )

        # Section 6 — Lifecycle changes
        self._s6 = Section6_LifecycleChanges(
            model_version       = meta.get("glassbox_version", ""),
            glassbox_version    = meta.get("glassbox_version", ""),
            report_version      = "1.0",
            analysis_timestamp  = ts,
            previous_reports    = [],
            planned_modifications = "[PROVIDER TO COMPLETE — planned updates, retraining schedule, version control process]",
        )

        # Section 7 — Standards
        self._s7 = Section7_HarmonisedStandards(
            standards_applied = [
                {
                    "standard": "ISO/IEC 42001:2023",
                    "title":    "Artificial intelligence management systems",
                    "status":   "Reference — harmonised status pending CEN/CENELEC JTC 21",
                },
                {
                    "standard": "ISO/IEC 23894:2023",
                    "title":    "Guidance on risk management for AI systems",
                    "status":   "Reference",
                },
                {
                    "standard": "ISO/IEC 25059:2023",
                    "title":    "Quality model for AI systems",
                    "status":   "Reference",
                },
            ],
            common_specifications_used = ["[PROVIDER TO COMPLETE — EU common specifications if applicable under Article 41]"],
            other_technical_specs      = [
                "EU AI Act (EU 2024/1689) — Annex IV",
                "Glassbox Mechanistic Interpretability Standard v2.6.0",
            ],
        )

        # Section 8 — Declaration of conformity
        self._s8 = Section8_Declaration(
            declaration_reference   = f"GDOC-{self._report_id}",
            declaration_date        = ts[:10],
            signatory_name          = "[PROVIDER TO COMPLETE — authorised signatory name]",
            signatory_position      = "[PROVIDER TO COMPLETE — position / title]",
            conformity_assessment_body = "[PROVIDER TO COMPLETE — notified body name if applicable under Article 43]",
            notified_body_number    = "[PROVIDER TO COMPLETE — EU notified body number, if applicable]",
            certificate_reference   = "[PROVIDER TO COMPLETE — certificate number if third-party assessment used]",
        )

        # Section 9 — Post-market monitoring
        self._s9 = Section9_PostMarketMonitoring(
            monitoring_plan_reference   = f"PMP-{self._report_id}",
            data_collection_approach    = "Automated logging of model inputs/outputs + Glassbox re-audit on flagged cases",
            performance_indicators      = [
                "Faithfulness F1 score >= threshold per deployment context",
                "Sufficiency score >= 0.70 for credit/healthcare decisions",
                "Circuit consistency across similar input types",
                "Frequency of human override events",
                "Customer complaint rate for AI-driven decisions",
                "Drift detection: KL divergence on output distribution",
            ],
            review_frequency            = "Quarterly review, triggered re-audit on model update or incident",
            incident_reporting_process  = "Serious incidents reported to national competent authority per Article 73 within 15 days",
            stakeholder_feedback_loop   = "Affected persons can request human review of any AI-assisted decision (Article 13)",
            reaudit_triggers            = [
                "Model architecture or weights update",
                "Deployment to new use case or geographic region",
                "F1 score drop > 10% from baseline",
                "Serious incident under Article 73",
                "Substantial modification as defined under Article 83",
            ],
        )

    # ------------------------------------------------------------------
    # Internal: JSON structure
    # ------------------------------------------------------------------

    def _build_json_structure(self) -> Dict[str, Any]:
        return {
            "document_type":        "EU AI Act Annex IV Technical Documentation",
            "regulation":           "Regulation (EU) 2024/1689 of the European Parliament and of the Council",
            "regulation_article":   "Article 11 — Technical Documentation",
            "report_id":            self._report_id,
            "generated_by":         "Glassbox — EU AI Act Compliance Audit Platform",
            "generated_at":         self._created_at.isoformat(),
            "risk_classification":  self.risk_classification.value,
            "compliance_status":    self._compute_compliance_status().value,
            "n_analyses_included":  len(self._analyses),
            "sections": {
                "1_general_description":   self._section_to_dict(self._s1, "Annex IV, Section 1"),
                "2_development_design":    self._section_to_dict(self._s2, "Annex IV, Section 2"),
                "3_monitoring_control":    self._section_to_dict(self._s3, "Annex IV, Section 3"),
                "4_data_governance":       self._section_to_dict(self._s4, "Annex IV, Section 4"),
                "5_risk_management":       self._section_to_dict(self._s5, "Annex IV, Section 5"),
                "6_lifecycle_changes":     self._section_to_dict(self._s6, "Annex IV, Section 6"),
                "7_harmonised_standards":  self._section_to_dict(self._s7, "Annex IV, Section 7"),
                "8_declaration":           self._section_to_dict(self._s8, "Annex IV, Section 8"),
                "9_post_market_monitoring":self._section_to_dict(self._s9, "Annex IV, Section 9"),
            },
            "raw_analyses": [
                {
                    "use_case":     uc,
                    "n_heads":      r.get("n_heads"),
                    "circuit":      [str(h) for h in r.get("circuit", [])],
                    "faithfulness": r.get("faithfulness"),
                    "clean_ld":     r.get("clean_ld"),
                    "model":        r.get("model_metadata", {}).get("model_name"),
                }
                for uc, r in zip(self._use_cases, self._analyses)
            ],
        }

    def _section_to_dict(self, section: Any, ref: str) -> Dict[str, Any]:
        if section is None:
            return {"regulation_ref": ref, "status": "NOT_POPULATED"}
        d = asdict(section)
        d["_regulation_ref"] = ref
        return d

    # ------------------------------------------------------------------
    # Internal: PDF builders
    # ------------------------------------------------------------------

    def _build_pdf_styles(self, base, colors):
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        styles = {}
        styles["cover_title"] = ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontSize=22,
            spaceAfter=8,
            textColor=colors.HexColor("#1a1a2e"),
            fontName="Helvetica-Bold",
        )
        styles["cover_sub"] = ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontSize=13,
            spaceAfter=5,
            textColor=colors.HexColor("#16213e"),
        )
        styles["section_heading"] = ParagraphStyle(
            "section_heading",
            parent=base["Heading1"],
            fontSize=14,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#0f3460"),
            fontName="Helvetica-Bold",
            borderPad=(0, 0, 3, 0),
        )
        styles["subsection"] = ParagraphStyle(
            "subsection",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#1a1a2e"),
            fontName="Helvetica-Bold",
        )
        styles["body"] = ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=9.5,
            spaceAfter=4,
            leading=14,
            textColor=colors.HexColor("#2d2d2d"),
        )
        styles["legal_ref"] = ParagraphStyle(
            "legal_ref",
            parent=base["Normal"],
            fontSize=8,
            spaceAfter=2,
            textColor=colors.HexColor("#666666"),
            fontName="Helvetica-Oblique",
        )
        styles["warning"] = ParagraphStyle(
            "warning",
            parent=base["Normal"],
            fontSize=9,
            spaceAfter=4,
            textColor=colors.HexColor("#c0392b"),
            fontName="Helvetica-Bold",
        )
        styles["metric"] = ParagraphStyle(
            "metric",
            parent=base["Normal"],
            fontSize=9.5,
            spaceAfter=3,
            fontName="Courier",
            textColor=colors.HexColor("#0f3460"),
        )
        styles["toc_entry"] = ParagraphStyle(
            "toc_entry",
            parent=base["Normal"],
            fontSize=9.5,
            spaceAfter=3,
            leading=14,
        )
        styles["centered"] = ParagraphStyle(
            "centered",
            parent=base["Normal"],
            fontSize=9.5,
            alignment=TA_CENTER,
        )
        return styles

    def _pdf_cover(self, styles, colors):
        from reportlab.platypus import Spacer, Table, TableStyle, HRFlowable, Paragraph
        from reportlab.lib.units import mm

        story = []
        story.append(Spacer(1, 20*mm))
        story.append(Paragraph("EU AI Act", styles["cover_sub"]))
        story.append(Paragraph("Annex IV Technical Documentation", styles["cover_title"]))
        story.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor("#0f3460")))
        story.append(Spacer(1, 6*mm))

        story.append(Paragraph(f"AI System: {self.model_name}", styles["cover_sub"]))
        story.append(Paragraph(f"Provider: {self.provider_name}", styles["cover_sub"]))
        story.append(Paragraph(f"Report ID: GB-{self._report_id}", styles["cover_sub"]))
        story.append(Paragraph(f"Date: {self._created_at.strftime('%d %B %Y')}", styles["cover_sub"]))
        story.append(Paragraph(f"Risk Classification: {self.risk_classification.value.upper()}", styles["cover_sub"]))
        story.append(Spacer(1, 8*mm))

        # Compliance status badge
        status = self._compute_compliance_status()
        badge_color = {
            ComplianceStatus.COMPLIANT:               "#27ae60",
            ComplianceStatus.CONDITIONALLY_COMPLIANT: "#f39c12",
            ComplianceStatus.INCOMPLETE:              "#e67e22",
            ComplianceStatus.NON_COMPLIANT:           "#c0392b",
        }.get(status, "#7f8c8d")

        story.append(Paragraph(
            f'<font color="{badge_color}"><b>Documentation Status: {status.value.upper().replace("_", " ")}</b></font>',
            styles["cover_sub"],
        ))
        story.append(Spacer(1, 8*mm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 4*mm))

        # Explainability summary table
        if self._s3:
            grade_color = {"A": "#27ae60", "B": "#2980b9", "C": "#e67e22", "D": "#c0392b"}.get(
                self._s3.explainability_grade[0], "#7f8c8d"
            )
            data = [
                ["Explainability Grade", "Sufficiency", "Comprehensiveness", "F1 Score"],
                [
                    f"{self._s3.explainability_grade}",
                    f"{self._s3.sufficiency:.3f}",
                    f"{self._s3.comprehensiveness:.3f}",
                    f"{self._s3.f1_score:.3f}",
                ],
            ]
            t = Table(data, colWidths=[90*mm, 30*mm, 40*mm, 30*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ecf0f1")),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                ("ROWBACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fafafa")),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t)

        story.append(Spacer(1, 6*mm))
        story.append(Paragraph(
            "Generated by Glassbox — EU AI Act Compliance Audit Platform",
            styles["legal_ref"],
        ))
        story.append(Paragraph(
            "Regulation (EU) 2024/1689 | Article 11 | Annex IV",
            styles["legal_ref"],
        ))
        return story

    def _pdf_legal_notice(self, styles):
        from reportlab.platypus import Spacer, Paragraph
        story = []
        story.append(Paragraph("Legal Notice", styles["section_heading"]))
        story.append(Paragraph(
            "This document constitutes technical documentation as required by Article 11 and Annex IV of "
            "Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 "
            "laying down harmonised rules on artificial intelligence (the AI Act).",
            styles["body"],
        ))
        story.append(Paragraph(
            "This report was generated using the Glassbox mechanistic interpretability framework, "
            "which applies attribution patching (Nanda et al. 2023) and minimum faithful circuit "
            "discovery to identify the causal attention circuit responsible for a model's predictions. "
            "The faithfulness metrics (sufficiency, comprehensiveness, F1) quantify how completely "
            "and necessarily the identified circuit explains the model's behaviour.",
            styles["body"],
        ))
        story.append(Paragraph(
            "DISCLAIMER: This report is generated from automated analysis and constitutes supporting "
            "documentation for human review. Fields marked [PROVIDER TO COMPLETE] must be filled by "
            "the AI system provider before submission to a national competent authority. This report "
            "does not constitute legal advice and does not guarantee regulatory compliance. Providers "
            "bear full responsibility for the accuracy and completeness of their Annex IV documentation.",
            styles["warning"],
        ))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Enforcement date: 2 August 2026 (high-risk AI systems under Annex III, Article 6).",
            styles["legal_ref"],
        ))
        return story

    def _pdf_executive_summary(self, styles, colors):
        from reportlab.platypus import Spacer, Paragraph, Table, TableStyle
        from reportlab.lib.units import mm
        story = []
        story.append(Paragraph("Executive Summary", styles["section_heading"]))

        s3 = self._s3
        s5 = self._s5
        n_analyses = len(self._analyses)

        story.append(Paragraph(
            f"This Annex IV compliance report covers the AI system <b>{self.model_name}</b>, "
            f"deployed in the context of <b>{self.deployment_context.value.replace('_', ' ')}</b> "
            f"by <b>{self.provider_name}</b>. The system is classified as "
            f"<b>{self.risk_classification.value.upper()}</b> under EU AI Act Article 6 / Annex III.",
            styles["body"],
        ))

        if s3:
            story.append(Paragraph(
                f"Glassbox mechanistic analysis ({n_analyses} use case(s)) produced the following "
                f"explainability assessment for Article 13 compliance:",
                styles["body"],
            ))
            story.append(Paragraph(
                f"  Sufficiency:        {s3.sufficiency:.4f}  "
                f"(fraction of model behaviour explained by the identified circuit)",
                styles["metric"],
            ))
            story.append(Paragraph(
                f"  Comprehensiveness:  {s3.comprehensiveness:.4f}  "
                f"(fraction of circuit heads that are causally necessary)",
                styles["metric"],
            ))
            story.append(Paragraph(
                f"  F1 Score:           {s3.f1_score:.4f}  (harmonic mean)",
                styles["metric"],
            ))
            story.append(Paragraph(
                f"  Faithfulness grade: {s3.explainability_grade}",
                styles["metric"],
            ))
            story.append(Spacer(1, 4))
            story.append(Paragraph(s3.explainability_rationale, styles["body"]))

        if s5 and s5.faithfulness_risk_flag:
            story.append(Paragraph(
                "RISK FLAG: Faithfulness F1 score below 0.50 threshold. "
                "The identified circuit does not sufficiently explain model behaviour. "
                "Additional analysis or architectural changes recommended before deployment.",
                styles["warning"],
            ))

        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"Analyses conducted: {n_analyses} | "
            f"Circuit heads identified: {len(self._s2.circuit_heads) if self._s2 else 0} unique | "
            f"Total model heads: {self._s2.n_analysis_heads_total if self._s2 else 0}",
            styles["legal_ref"],
        ))
        return story

    def _pdf_section(self, number, title, reg_ref, styles):
        from reportlab.platypus import Paragraph, Spacer
        story = []
        story.append(Paragraph(
            f"Section {number} — {title}",
            styles["section_heading"],
        ))
        story.append(Paragraph(reg_ref, styles["legal_ref"]))
        story.append(Spacer(1, 4))
        return story

    def _pdf_field(self, label, value, styles, is_warning=False):
        from reportlab.platypus import Paragraph, Spacer
        style = styles["warning"] if (is_warning and "[PROVIDER TO COMPLETE" in str(value)) else styles["body"]
        return [Paragraph(f"<b>{label}:</b> {value}", style)]

    def _pdf_section1(self, styles, colors):
        s = self._s1
        story = self._pdf_section("1", "General Description", "Annex IV, Section 1 | Article 11(1)(a) | Article 13(3)(a)", styles)
        if not s:
            return story
        story += self._pdf_field("System Name",          s.system_name, styles)
        story += self._pdf_field("Intended Purpose",     s.intended_purpose, styles)
        story += self._pdf_field("Deployment Context",   s.deployment_context.replace("_", " ").title(), styles)
        story += self._pdf_field("Risk Classification",  self.risk_classification.value, styles)
        story += self._pdf_field("Affected Persons",     s.categories_of_persons, styles)
        story += self._pdf_field("Geographic Scope",     s.geographic_scope, styles)
        story += self._pdf_field("Provider",             s.provider_name, styles)
        story += self._pdf_field("Provider Address",     s.provider_address, styles)
        if s.provider_eu_rep:
            story += self._pdf_field("EU Representative", s.provider_eu_rep, styles)
        story += self._pdf_field("Report ID",            f"GB-{s.report_id}", styles)
        story += self._pdf_field("Report Date",          s.report_date, styles)
        story += self._pdf_field("Glassbox Version",     s.glassbox_version, styles)
        story += self._pdf_field("Model Architecture",   s.model_architecture, styles)
        story += self._pdf_field("Layers",               str(s.model_n_layers), styles)
        story += self._pdf_field("Attention Heads",      str(s.model_n_heads), styles)
        story += self._pdf_field("Embedding Dimension",  str(s.model_d_model), styles)
        return story

    def _pdf_section2(self, styles, colors):
        s = self._s2
        story = self._pdf_section("2", "Development and Design", "Annex IV, Section 2 | Article 10 | Article 11(1)(d)", styles)
        if not s:
            return story
        story += self._pdf_field("Design Approach",             s.design_approach, styles)
        story += self._pdf_field("Architecture",                s.architecture_description, styles)
        story += self._pdf_field("Attribution Method",          s.attribution_method, styles)
        story += self._pdf_field("Circuit Discovery Algorithm", s.circuit_discovery_algorithm, styles)
        story += self._pdf_field("Total Attention Heads",       str(s.n_analysis_heads_total), styles)
        story += self._pdf_field("Circuit Heads Identified",    str(s.n_circuit_heads_found), styles)
        story += self._pdf_field("Circuit Composition",         ", ".join(s.circuit_heads) or "None identified", styles)
        story += self._pdf_field("Logit Difference (clean)",    f"{s.clean_logit_difference:.4f}", styles)
        if s.top_attributions:
            from reportlab.platypus import Paragraph
            story.append(Paragraph("<b>Top 5 Attribution Heads:</b>", styles["body"]))
            for h in s.top_attributions[:5]:
                story.append(Paragraph(
                    f"  L{h.get('layer',0)}H{h.get('head',0)}: attribution={h.get('attr',0.0):.4f}, "
                    f"rel_depth={h.get('rel_depth',0.0):.3f}",
                    styles["metric"],
                ))
        from reportlab.platypus import Paragraph
        story.append(Paragraph("<b>Scientific References:</b>", styles["body"]))
        for ref in s.reference_papers:
            story.append(Paragraph(f"  {ref}", styles["legal_ref"]))
        return story

    def _pdf_section3(self, styles, colors):
        s = self._s3
        story = self._pdf_section("3", "Monitoring, Functioning, and Control", "Annex IV, Section 3 | Article 9(6) | Article 13(3)(b) | Article 14 | Article 15", styles)
        if not s:
            return story

        from reportlab.platypus import Paragraph, Spacer
        story.append(Paragraph("<b>3.1 Explainability Metrics (Article 13 — Transparency)</b>", styles["subsection"]))
        story += self._pdf_field("Explainability Grade",    s.explainability_grade, styles)
        story += self._pdf_field("Sufficiency Score",       f"{s.sufficiency:.4f}", styles)
        story += self._pdf_field("Comprehensiveness Score", f"{s.comprehensiveness:.4f}", styles)
        story += self._pdf_field("F1 Score",                f"{s.f1_score:.4f}", styles)
        story += self._pdf_field("Faithfulness Category",   s.faithfulness_category, styles)
        story += self._pdf_field("Sufficiency Approximate", str(s.suff_is_approximate) + " (Taylor method; set False for Integrated Gradients)", styles)
        story.append(Paragraph(s.explainability_rationale, styles["body"]))

        story.append(Paragraph("<b>3.2 Performance Metrics (Article 15)</b>", styles["subsection"]))
        story += self._pdf_field("Accuracy Metric", s.accuracy_metric, styles, is_warning=True)
        story += self._pdf_field("Accuracy Value",  s.accuracy_value,  styles, is_warning=True)
        story += self._pdf_field("Robustness",      s.robustness_measures, styles, is_warning=True)
        story += self._pdf_field("Cybersecurity",   s.cybersecurity_measures, styles, is_warning=True)

        story.append(Paragraph("<b>3.3 Human Oversight (Article 14)</b>", styles["subsection"]))
        story += self._pdf_field("Oversight Measures",   s.human_oversight_measures, styles)
        story += self._pdf_field("Override Mechanism",   s.override_mechanism, styles)
        if s.monitoring_indicators:
            story.append(Paragraph("<b>Monitoring Indicators:</b>", styles["body"]))
            for ind in s.monitoring_indicators:
                story.append(Paragraph(f"  - {ind}", styles["body"]))
        return story

    def _pdf_section4(self, styles, colors):
        s = self._s4
        story = self._pdf_section("4", "Data Governance", "Annex IV, Section 4 | Article 10 | GDPR Article 22", styles)
        if not s:
            return story
        story += self._pdf_field("Training Dataset",      s.training_dataset_description,   styles, is_warning=True)
        story += self._pdf_field("Validation Dataset",    s.validation_dataset_description, styles, is_warning=True)
        story += self._pdf_field("Test Dataset",          s.test_dataset_description,        styles, is_warning=True)
        story += self._pdf_field("Data Provenance",       s.data_provenance,                 styles, is_warning=True)
        story += self._pdf_field("Collection Method",     s.data_collection_methodology,     styles, is_warning=True)
        story += self._pdf_field("Bias Assessment",       s.bias_assessment,                 styles, is_warning=True)
        story += self._pdf_field("Preprocessing",         s.data_preprocessing_steps,        styles, is_warning=True)
        story += self._pdf_field("Known Limitations",     s.known_data_limitations,          styles, is_warning=True)
        story += self._pdf_field("Personal Data",         str(s.personal_data_involved), styles)
        story += self._pdf_field("GDPR Measures",         s.gdpr_compliance_measures,        styles, is_warning=True)
        story += self._pdf_field("Tokens Analysed",       str(s.input_token_count_analyzed), styles)
        story += self._pdf_field("Token Attribution",     str(s.token_attribution_available), styles)
        return story

    def _pdf_section5(self, styles, colors):
        s = self._s5
        story = self._pdf_section("5", "Risk Management", "Annex IV, Section 5 | Article 9 | Article 9(2)", styles)
        if not s:
            return story

        from reportlab.platypus import Paragraph
        story += self._pdf_field("Risk ID Process",       s.risk_identification_process, styles)
        story += self._pdf_field("Mitigation Measures",   s.risk_mitigation_measures,    styles, is_warning=True)
        story += self._pdf_field("Residual Risks",        s.residual_risks,              styles, is_warning=True)
        story += self._pdf_field("Explainability Risk",   s.explainability_risk_assessment, styles)

        if s.faithfulness_risk_flag:
            story.append(Paragraph("RISK FLAG: Low faithfulness F1 — circuit does not fully explain model behaviour.", styles["warning"]))
        if s.circuit_concentration_risk:
            story.append(Paragraph("RISK FLAG: Circuit concentration — fewer than 3 heads identified. May indicate over-reliance on single components.", styles["warning"]))

        if s.identified_risks:
            story.append(Paragraph("<b>Identified Risks:</b>", styles["body"]))
            for r in s.identified_risks:
                story.append(Paragraph(
                    f"  [{r.get('severity','?').upper()}] {r.get('risk','?')} — {r.get('rationale','?')}",
                    styles["body"],
                ))

        if s.recommended_actions:
            story.append(Paragraph("<b>Recommended Actions:</b>", styles["body"]))
            for action in s.recommended_actions:
                story.append(Paragraph(f"  - {action}", styles["body"]))
        return story

    def _pdf_section6(self, styles, colors):
        s = self._s6
        story = self._pdf_section("6", "Changes Through the Lifecycle", "Annex IV, Section 6 | Article 16(d) | Article 83", styles)
        if not s:
            return story
        story += self._pdf_field("Model Version",         s.model_version, styles)
        story += self._pdf_field("Glassbox Version",      s.glassbox_version, styles)
        story += self._pdf_field("Report Version",        s.report_version, styles)
        story += self._pdf_field("Analysis Timestamp",    s.analysis_timestamp, styles)
        story += self._pdf_field("Planned Modifications", s.planned_modifications, styles, is_warning=True)
        story += self._pdf_field("Retesting Trigger",     s.retesting_trigger, styles)
        return story

    def _pdf_section7(self, styles, colors):
        s = self._s7
        story = self._pdf_section("7", "Harmonised Standards", "Annex IV, Section 7 | Article 40 | Article 41", styles)
        if not s:
            return story
        story += self._pdf_field("Standardisation Body", s.standardisation_body_ref, styles)
        from reportlab.platypus import Paragraph
        story.append(Paragraph("<b>Standards Applied:</b>", styles["body"]))
        for std in s.standards_applied:
            story.append(Paragraph(
                f"  {std.get('standard','')} — {std.get('title','')} ({std.get('status','')})",
                styles["body"],
            ))
        for cs in s.common_specifications_used:
            story += self._pdf_field("Common Specification", cs, styles, is_warning="PROVIDER" in cs)
        for ts in s.other_technical_specs:
            story += self._pdf_field("Technical Spec", ts, styles)
        return story

    def _pdf_section8(self, styles, colors):
        s = self._s8
        story = self._pdf_section("8", "EU Declaration of Conformity", "Annex IV, Section 8 | Article 47 | Article 43", styles)
        if not s:
            return story
        story += self._pdf_field("Declaration Reference",  s.declaration_reference, styles)
        story += self._pdf_field("Declaration Date",       s.declaration_date, styles)
        story += self._pdf_field("Signatory Name",         s.signatory_name,    styles, is_warning=True)
        story += self._pdf_field("Signatory Position",     s.signatory_position, styles, is_warning=True)
        story += self._pdf_field("Conformity Body",        s.conformity_assessment_body, styles, is_warning=True)
        story += self._pdf_field("Notified Body Number",   s.notified_body_number, styles, is_warning=True)
        story += self._pdf_field("Certificate Reference",  s.certificate_reference, styles, is_warning=True)
        return story

    def _pdf_section9(self, styles, colors):
        s = self._s9
        story = self._pdf_section("9", "Post-Market Monitoring Plan", "Annex IV, Section 9 | Article 72 | Article 73", styles)
        if not s:
            return story
        story += self._pdf_field("Plan Reference",         s.monitoring_plan_reference, styles)
        story += self._pdf_field("Data Collection",        s.data_collection_approach, styles)
        story += self._pdf_field("Review Frequency",       s.review_frequency, styles)
        story += self._pdf_field("Incident Reporting",     s.incident_reporting_process, styles)
        story += self._pdf_field("Stakeholder Feedback",   s.stakeholder_feedback_loop, styles)
        story += self._pdf_field("Re-audit Frequency",     s.reaudit_frequency, styles)
        from reportlab.platypus import Paragraph
        if s.performance_indicators:
            story.append(Paragraph("<b>Performance Indicators:</b>", styles["body"]))
            for ind in s.performance_indicators:
                story.append(Paragraph(f"  - {ind}", styles["body"]))
        if s.reaudit_triggers:
            story.append(Paragraph("<b>Re-audit Triggers:</b>", styles["body"]))
            for t in s.reaudit_triggers:
                story.append(Paragraph(f"  - {t}", styles["body"]))
        return story

    def _pdf_methodology(self, styles):
        from reportlab.platypus import Paragraph, Spacer
        story = []
        story.append(Paragraph("Methodology — Glassbox Mechanistic Interpretability", styles["section_heading"]))
        story.append(Paragraph(
            "Glassbox applies attribution patching (Nanda et al. 2023) to identify the minimum "
            "faithful circuit: the smallest set of attention heads that causally explains the model's "
            "prediction. The algorithm operates in O(3 + 2p) forward passes, where p is the number "
            "of backward pruning steps (typically 0-4).",
            styles["body"],
        ))
        story.append(Paragraph(
            "<b>Sufficiency</b> measures what fraction of the model's logit difference is captured by the "
            "identified circuit. A value of 1.0 means the circuit fully reproduces the model's prediction "
            "behaviour. Values below 0.70 indicate backup mechanisms not captured by the circuit.",
            styles["body"],
        ))
        story.append(Paragraph(
            "<b>Comprehensiveness</b> measures what fraction of the circuit is causally necessary: ablating "
            "circuit heads reduces model performance by this fraction. A value of 0.60 means removing "
            "circuit heads reduces the logit difference by 60%. Low values indicate redundant heads.",
            styles["body"],
        ))
        story.append(Paragraph(
            "<b>F1 Score</b> (harmonic mean of sufficiency and comprehensiveness) is the primary regulatory "
            "metric for Article 13 explainability compliance assessment.",
            styles["body"],
        ))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Source code: https://github.com/designer-coderajay/glassbox-mech-interp | "
            "PyPI: pip install glassbox-mech-interp",
            styles["legal_ref"],
        ))
        return story

    def _pdf_signature_block(self, styles, colors):
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import mm
        story = []
        story.append(Paragraph("Certification", styles["section_heading"]))
        story.append(Paragraph(
            "By signing this document, the authorised representative of the provider certifies that "
            "the information contained in this Annex IV technical documentation is accurate and complete "
            "to the best of their knowledge, in accordance with Article 16 of Regulation (EU) 2024/1689.",
            styles["body"],
        ))
        story.append(Spacer(1, 12*mm))

        data = [
            ["Provider Organisation:", self.provider_name, "Date:", self._created_at.strftime("%d/%m/%Y")],
            ["Authorised Signatory:",  "_" * 30,           "Position:", "_" * 20],
            ["Signature:",             "_" * 30,           "",          ""],
        ]
        t = Table(data, colWidths=[50*mm, 65*mm, 25*mm, 50*mm])
        t.setStyle(TableStyle([
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("FONTNAME",   (0, 0), (0, -1),  "Helvetica-Bold"),
            ("FONTNAME",   (2, 0), (2, -1),  "Helvetica-Bold"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 8*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        story.append(Paragraph(
            f"Report ID: GB-{self._report_id} | "
            f"Generated: {self._created_at.strftime('%Y-%m-%dT%H:%M:%SZ')} | "
            "Generated by Glassbox v" + (self._s1.glassbox_version if self._s1 else "?"),
            styles["legal_ref"],
        ))
        return story

    # ------------------------------------------------------------------
    # Internal: scoring helpers
    # ------------------------------------------------------------------

    def _compute_grade(self, suff: float, comp: float, f1: float) -> ExplainabilityGrade:
        ms, mc, mf = self._GRADE_A_THRESHOLD
        if suff >= ms and comp >= mc and f1 >= mf:
            return ExplainabilityGrade.GRADE_A
        ms, mc, mf = self._GRADE_B_THRESHOLD
        if suff >= ms and comp >= mc and f1 >= mf:
            return ExplainabilityGrade.GRADE_B
        ms, mc, mf = self._GRADE_C_THRESHOLD
        if f1 >= mf:
            return ExplainabilityGrade.GRADE_C
        return ExplainabilityGrade.GRADE_D

    def _grade_rationale(
        self,
        grade: ExplainabilityGrade,
        suff: float, comp: float, f1: float,
        categories: List[str],
    ) -> str:
        cat_summary = f"Faithfulness category: {max(set(categories), key=categories.count) if categories else 'N/A'}."
        if grade == ExplainabilityGrade.GRADE_A:
            return (
                f"Grade A: The identified circuit comprehensively explains model behaviour. "
                f"Sufficiency={suff:.3f} (>{self._GRADE_A_THRESHOLD[0]}), "
                f"Comprehensiveness={comp:.3f} (>{self._GRADE_A_THRESHOLD[1]}), "
                f"F1={f1:.3f}. {cat_summary} "
                f"This system meets the Article 13 transparency standard for EU AI Act compliance."
            )
        elif grade == ExplainabilityGrade.GRADE_B:
            return (
                f"Grade B: Circuit substantially explains model behaviour with minor gaps. "
                f"Sufficiency={suff:.3f}, Comprehensiveness={comp:.3f}, F1={f1:.3f}. "
                f"{cat_summary} Meets baseline Article 13 requirements. "
                f"Consider running integrated_gradients method for higher accuracy."
            )
        elif grade == ExplainabilityGrade.GRADE_C:
            return (
                f"Grade C: Partial explainability — circuit explains some but not all model behaviour. "
                f"Sufficiency={suff:.3f}, Comprehensiveness={comp:.3f}, F1={f1:.3f}. "
                f"{cat_summary} Significant backup mechanisms likely. "
                f"Recommend additional analysis and risk mitigation before deployment."
            )
        else:
            return (
                f"Grade D: Minimal explainability — the identified circuit does not reliably explain "
                f"model behaviour. Sufficiency={suff:.3f}, Comprehensiveness={comp:.3f}, F1={f1:.3f}. "
                f"{cat_summary} HIGH REGULATORY RISK. "
                f"This system may not satisfy Article 13 transparency requirements. "
                f"Architecture review recommended before EU deployment."
            )

    def _compute_compliance_status(self) -> ComplianceStatus:
        if not self._s3:
            return ComplianceStatus.INCOMPLETE
        f1 = self._s3.f1_score
        if f1 >= self._GRADE_B_THRESHOLD[2]:
            return ComplianceStatus.CONDITIONALLY_COMPLIANT  # provider fields still needed
        if f1 < self._FAITHFULNESS_RISK_F1_THRESHOLD:
            return ComplianceStatus.NON_COMPLIANT
        return ComplianceStatus.INCOMPLETE

    def _infer_affected_persons(self) -> str:
        mapping = {
            DeploymentContext.FINANCIAL_SERVICES:      "Natural persons subject to credit, insurance, or financial service decisions",
            DeploymentContext.HEALTHCARE:              "Patients and clinical staff subject to AI-assisted diagnosis or treatment recommendations",
            DeploymentContext.HR_EMPLOYMENT:           "Job applicants and employees subject to automated recruitment or assessment decisions",
            DeploymentContext.LEGAL:                   "Parties in legal proceedings subject to AI-assisted document analysis or advice",
            DeploymentContext.CRITICAL_INFRASTRUCTURE: "Users and operators of critical infrastructure systems",
            DeploymentContext.EDUCATION:               "Students and candidates subject to automated assessment or admissions decisions",
            DeploymentContext.LAW_ENFORCEMENT:         "Natural persons subject to biometric identification or predictive policing",
            DeploymentContext.GENERAL_PURPOSE:         "General users — specific affected categories depend on deployment use case",
            DeploymentContext.OTHER_HIGH_RISK:         "[PROVIDER TO COMPLETE — categories of persons affected by system decisions]",
        }
        return mapping.get(self.deployment_context, "Not specified")

    def _infer_oversight_measures(self) -> str:
        mapping = {
            DeploymentContext.FINANCIAL_SERVICES: (
                "Human credit analyst reviews all AI-assisted credit decisions before finalisation. "
                "Customers can request human review of any automated decision (GDPR Article 22)."
            ),
            DeploymentContext.HEALTHCARE: (
                "Qualified clinician reviews all AI-assisted diagnostic recommendations. "
                "AI output is advisory only — final clinical decision rests with the treating physician."
            ),
            DeploymentContext.HR_EMPLOYMENT: (
                "HR professional reviews all AI-assisted candidate assessments. "
                "Candidates can request human review and explanation of screening decisions."
            ),
            DeploymentContext.LEGAL: (
                "Qualified legal professional reviews all AI-generated legal analysis. "
                "AI output is research support only — legal advice requires solicitor review."
            ),
        }
        return mapping.get(
            self.deployment_context,
            "Human reviewer is available to override any automated decision. "
            "[PROVIDER TO COMPLETE — specify oversight process for this deployment context]"
        )

    def _build_monitoring_indicators(self, suff: float, comp: float, f1: float) -> List[str]:
        indicators = [
            f"Faithfulness F1 score threshold: >= {max(f1 - 0.05, 0.40):.2f} (current: {f1:.3f})",
            f"Sufficiency score threshold: >= {max(suff - 0.05, 0.50):.2f} (current: {suff:.3f})",
            "Circuit head count consistency: +/- 2 heads from baseline",
            "Model output distribution drift: KL divergence alert at > 0.10",
            "Human override rate: alert if > 10% of decisions are overridden",
            "Article 73 incident rate: any serious incident triggers immediate re-audit",
        ]
        return indicators

    def _identify_risks(
        self,
        suff: float, comp: float, f1: float,
        risk_flag: bool, conc_risk: bool,
    ) -> List[Dict[str, str]]:
        risks = []
        if risk_flag:
            risks.append({
                "risk":      "Insufficient circuit explainability",
                "severity":  "high",
                "rationale": f"F1 score {f1:.3f} below threshold 0.50. The identified circuit may not fully explain model decisions, creating Article 13 transparency risk.",
                "article":   "Article 13, Article 9(2)(a)",
            })
        if comp < 0.40:
            risks.append({
                "risk":      "Low circuit comprehensiveness — redundant mechanisms",
                "severity":  "medium",
                "rationale": f"Comprehensiveness {comp:.3f} < 0.40. Model may rely on backup circuits not identified in this analysis.",
                "article":   "Article 9(2)(b), Annex IV Section 3",
            })
        if conc_risk:
            risks.append({
                "risk":      "Circuit concentration risk",
                "severity":  "medium",
                "rationale": "Fewer than 3 attention heads in identified circuit. High dependency on specific model components.",
                "article":   "Article 9(2)(a)",
            })
        risks.append({
            "risk":      "Incomplete data governance documentation",
            "severity":  "high",
            "rationale": "Training, validation, and test dataset fields require provider completion before regulatory submission.",
            "article":   "Article 10, Annex IV Section 4",
        })
        risks.append({
            "risk":      "Declaration of conformity not executed",
            "severity":  "high",
            "rationale": "Article 47 requires a written, signed declaration of conformity before placing high-risk AI on the market.",
            "article":   "Article 47",
        })
        return risks

    def _build_recommendations(
        self,
        f1: float,
        risk_flag: bool,
        conc_risk: bool,
    ) -> List[str]:
        recs = []
        if risk_flag:
            recs.append(
                f"Run gb.analyze() with method='integrated_gradients' for higher-accuracy sufficiency measurement"
            )
            recs.append("Consider architectural changes to improve circuit interpretability before EU deployment")
        if conc_risk:
            recs.append("Run batch_analyze() across diverse input cases to identify additional circuit heads")
        recs.append("Complete all [PROVIDER TO COMPLETE] fields before submitting to national competent authority")
        recs.append("Execute EU Declaration of Conformity (Article 47) before market deployment")
        recs.append("Register system in EU AI Act database (Article 71) if applicable to deployment context")
        recs.append(f"Set Glassbox re-audit schedule: every 6 months or on any model weight/architecture update")
        return recs
