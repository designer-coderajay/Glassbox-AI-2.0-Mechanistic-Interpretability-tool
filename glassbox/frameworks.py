# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.frameworks
===================
V5 multi-framework report packs (ROADMAP_V5_FOUNDATIONS.md Part 4 / Phase C #12).

The same evidence base (faithfulness, evidence tier, Annex IV sections, drift
monitoring) maps to more than the EU AI Act. This module cross-walks Glassbox
evidence to two other widely-used frameworks so one audit produces several
report packs:

  * NIST AI RMF — four core functions: GOVERN, MAP, MEASURE, MANAGE.
  * ISO/IEC 42001:2023 — Annex A, 38 controls across 9 objectives (A.2-A.10).

Honest scope: this maps at the **function / objective theme** level, which is
deterministic and testable. Exact NIST subcategory IDs and ISO/IEC 42001 Annex A
*control* IDs must be confirmed against the published standards — the mapping is a
documentation aid, not certification or legal advice. (The other half of #12, an
SSM/Mamba adapter, is torch and implements the AuditableModel protocol like any
backend.)

Sources for the framework structure:
  - NIST AI RMF core functions (airc.nist.gov/airmf-resources/airmf/)
  - ISO/IEC 42001:2023 Annex A (iso.org std 81230)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

__all__ = ["NIST_AI_RMF", "ISO_42001_OBJECTIVES", "framework_pack"]

NIST_AI_RMF: List[str] = ["GOVERN", "MAP", "MEASURE", "MANAGE"]

ISO_42001_OBJECTIVES: Dict[str, str] = {
    "A.2": "AI policy",
    "A.3": "Internal organization",
    "A.4": "Resources for AI systems",
    "A.5": "Assessing impacts of AI systems",
    "A.6": "AI system life cycle",
    "A.7": "Data for AI systems",
    "A.8": "Information for interested parties",
    "A.9": "Use of AI systems",
    "A.10": "Third-party relationships",
}

# Which Glassbox evidence supports which framework element (theme-level cross-walk).
_CROSSWALK: Dict[str, Dict[str, Any]] = {
    "faithfulness": {
        "nist": ["MEASURE"], "iso": ["A.5", "A.6"],
        "desc": "measured circuit faithfulness (sufficiency/comprehensiveness/F1) — validity of the explanation",
    },
    "evidence_tier": {
        "nist": ["GOVERN", "MEASURE"], "iso": ["A.2", "A.6"],
        "desc": "evidence-tier label + honest degradation policy (never silent, never fabricate)",
    },
    "annex_iv": {
        "nist": ["MAP"], "iso": ["A.6", "A.8"],
        "desc": "Annex IV technical documentation generated from the model",
    },
    "risk_management": {
        "nist": ["MAP", "MANAGE"], "iso": ["A.5"],
        "desc": "risk identification, mitigation, residual risk (EU AI Act Article 9)",
    },
    "drift_monitoring": {
        "nist": ["MANAGE"], "iso": ["A.6"],
        "desc": "post-market drift detection (CUSUM on fingerprints; EU AI Act Article 72)",
    },
    "confidence_gap": {
        "nist": ["MEASURE"], "iso": ["A.6"],
        "desc": "confidence is not faithfulness (r=0.009) — validity caveat for auditors",
    },
    "data_governance": {
        "nist": ["MAP"], "iso": ["A.7"],
        "desc": "training/analysis data provenance and handling",
    },
}

_DISCLAIMER = (
    "Maps Glassbox evidence to NIST AI RMF functions and ISO/IEC 42001 objectives "
    "at the theme level. Exact NIST subcategory IDs and ISO/IEC 42001 Annex A "
    "control IDs must be confirmed against the published standards. This is a "
    "documentation aid, not certification or legal advice."
)


def framework_pack(evidence_present: Iterable[str]) -> Dict[str, Any]:
    """Cross-walk the Glassbox evidence on hand to NIST AI RMF and ISO/IEC 42001.

    Args:
        evidence_present: Iterable of evidence category names (subset of the
            crosswalk keys: faithfulness, evidence_tier, annex_iv, risk_management,
            drift_monitoring, confidence_gap, data_governance). Unknown names are
            ignored.

    Returns:
        ``{nist_ai_rmf: {function: [{evidence, desc}]}, iso_42001: {obj: {objective,
        evidence:[...]}}, evidence_used, disclaimer}``.
    """
    keys = [k for k in evidence_present if k in _CROSSWALK]
    nist: Dict[str, List[Dict[str, str]]] = {fn: [] for fn in NIST_AI_RMF}
    iso: Dict[str, Dict[str, Any]] = {}

    for k in keys:
        entry = _CROSSWALK[k]
        for fn in entry["nist"]:
            nist[fn].append({"evidence": k, "desc": entry["desc"]})
        for obj in entry["iso"]:
            iso.setdefault(obj, {"objective": ISO_42001_OBJECTIVES[obj], "evidence": []})
            iso[obj]["evidence"].append(k)

    return {
        "nist_ai_rmf": {fn: items for fn, items in nist.items() if items},
        "iso_42001": iso,
        "evidence_used": keys,
        "disclaimer": _DISCLAIMER,
    }
