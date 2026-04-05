# SPDX-License-Identifier: BUSL-1.1
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
# Licensed under the Business Source License 1.1 (see LICENSE-COMMERCIAL).
# Tamper-evident audit chain is patent-pending — see PATENTS.md.
# Free for non-commercial and internal production use.
# Commercial redistribution / SaaS use requires a separate license.
# Contact: mahale.ajay01@gmail.com
"""
glassbox.audit_log
==================
Append-only audit trail for Glassbox compliance documentation sessions.

Every audit run (white-box or black-box) can be recorded here with a
tamper-evident hash chain so governance teams have a complete, verifiable
history of which models were audited, by whom, when, and what the outcome was.

Exportable as JSON or CSV to support technical documentation obligations and
internal risk registers.

LEGAL NOTICE — DOCUMENTATION AID ONLY
---------------------------------------
This module generates and stores audit trail records as a documentation aid.
It is provided strictly for informational and record-keeping support purposes.

  - Audit records are NOT official regulatory submissions. Whether records
    satisfy record-keeping obligations under EU AI Act Article 12 depends
    on the deployer's full compliance programme and applicable implementing
    acts — consult qualified legal counsel.
  - "Tamper-evident" hash chains detect unintended modification within the
    local file; they do not constitute a certified audit system under any
    regulatory standard (e.g., ISO 27001, SOC 2).
  - Exportable records are documentation starting points and must be reviewed
    and supplemented before submission to a competent authority.

Regulatory References (informational only)
------------------------------------------
Regulation (EU) 2024/1689 — EU AI Act:
  Article 12       — Logging and record-keeping for high-risk AI systems
  Annex IV, Sec. 6 — Lifecycle changes and version control documentation
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["AuditRecord", "AuditLog"]


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """
    A single immutable audit record.

    Attributes
    ----------
    record_id        : Unique ID for this record (UUID4 hex).
    timestamp_utc    : Unix timestamp (UTC) when the audit was run.
    model_name       : Name / identifier of the audited model.
    analysis_mode    : "white_box" | "black_box" | "demo".
    prompt           : The decision prompt used for analysis.
    correct_token    : Expected correct output token (white-box only).
    incorrect_token  : Distractor token (white-box only).
    provider_name    : Provider name from Annex IV metadata.
    deployment_context : Deployment context (financial_services, healthcare…).
    explainability_grade : A | B | C | D.
    compliance_status : conditionally_compliant | incomplete | non_compliant.
    faithfulness_f1  : Faithfulness F1 score (0.0–1.0).
    faithfulness_sufficiency : Sufficiency metric.
    faithfulness_comprehensiveness : Comprehensiveness metric.
    n_circuit_heads  : Number of circuit components identified.
    report_id        : Glassbox report ID (links to PDF/JSON artifact).
    auditor          : Free-text name/email of the person who triggered the audit.
    notes            : Optional free-text notes.
    prev_hash        : SHA-256 of the previous record (chain integrity).
    record_hash      : SHA-256 of this record's canonical fields.
    """
    record_id:                    str
    timestamp_utc:                float
    model_name:                   str
    analysis_mode:                str
    prompt:                       str
    correct_token:                str
    incorrect_token:              str
    provider_name:                str
    deployment_context:           str
    explainability_grade:         str
    compliance_status:            str
    faithfulness_f1:              float
    faithfulness_sufficiency:     float
    faithfulness_comprehensiveness: float
    n_circuit_heads:              int
    report_id:                    str
    auditor:                      str  = ""
    notes:                        str  = ""
    prev_hash:                    str  = ""
    record_hash:                  str  = field(default="", init=False)

    def __post_init__(self) -> None:
        self.record_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA-256 of the canonical fields (excluding record_hash itself)."""
        canonical = {
            "record_id":    self.record_id,
            "timestamp_utc": self.timestamp_utc,
            "model_name":   self.model_name,
            "analysis_mode": self.analysis_mode,
            "compliance_status": self.compliance_status,
            "faithfulness_f1": self.faithfulness_f1,
            "report_id":    self.report_id,
            "prev_hash":    self.prev_hash,
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def verify(self) -> bool:
        """Return True if the stored hash matches a freshly computed hash."""
        return self.record_hash == self._compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# AuditLog
# ──────────────────────────────────────────────────────────────────────────────

class AuditLog:
    """
    Append-only audit log for Glassbox compliance analyses.

    Persists records to a newline-delimited JSON file (one record per line).
    Each record contains a SHA-256 of the previous record's hash, forming a
    tamper-evident chain. Integrity can be verified with ``verify_chain()``.

    Parameters
    ----------
    path : Path to the .jsonl log file.  Created on first append if absent.

    Examples
    --------
    >>> log = AuditLog("glassbox_audit.jsonl")
    >>> log.append_from_result(result_dict, auditor="ajay@example.com")
    >>> summary = log.summary()
    >>> log.export_csv("audit_export.csv")
    """

    _CSV_FIELDS = [
        "record_id", "timestamp_utc", "model_name", "analysis_mode",
        "provider_name", "deployment_context", "explainability_grade",
        "compliance_status", "faithfulness_f1", "faithfulness_sufficiency",
        "faithfulness_comprehensiveness", "n_circuit_heads", "report_id",
        "auditor", "notes", "record_hash", "prev_hash",
    ]

    def __init__(self, path: Union[str, Path] = "glassbox_audit.jsonl") -> None:
        self.path = Path(path)
        self._records: List[AuditRecord] = []
        if self.path.exists():
            self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load existing records from disk."""
        self._records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # re-instantiate without triggering __post_init__ hash recompute
                    rec = AuditRecord.__new__(AuditRecord)
                    rec.__dict__.update(d)
                    self._records.append(rec)
                except Exception as exc:
                    logger.warning("Skipping malformed log line: %s", exc)

    def _append_to_disk(self, record: AuditRecord) -> None:
        """Append a single record to the JSONL file."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")

    # ── public API ────────────────────────────────────────────────────────────

    def append(
        self,
        model_name:            str,
        analysis_mode:         str,
        prompt:                str,
        correct_token:         str          = "",
        incorrect_token:       str          = "",
        provider_name:         str          = "",
        deployment_context:    str          = "other_high_risk",
        explainability_grade:  str          = "D",
        compliance_status:     str          = "non_compliant",
        faithfulness_f1:       float        = 0.0,
        faithfulness_sufficiency: float     = 0.0,
        faithfulness_comprehensiveness: float = 0.0,
        n_circuit_heads:       int          = 0,
        report_id:             str          = "",
        auditor:               str          = "",
        notes:                 str          = "",
    ) -> AuditRecord:
        """
        Append a new record to the audit log.

        Returns the created AuditRecord.
        """
        prev_hash = self._records[-1].record_hash if self._records else ""
        rec = AuditRecord(
            record_id=uuid.uuid4().hex[:12].upper(),
            timestamp_utc=time.time(),
            model_name=model_name,
            analysis_mode=analysis_mode,
            prompt=prompt[:500],  # truncate long prompts
            correct_token=correct_token,
            incorrect_token=incorrect_token,
            provider_name=provider_name,
            deployment_context=deployment_context,
            explainability_grade=explainability_grade,
            compliance_status=compliance_status,
            faithfulness_f1=round(float(faithfulness_f1), 6),
            faithfulness_sufficiency=round(float(faithfulness_sufficiency), 6),
            faithfulness_comprehensiveness=round(float(faithfulness_comprehensiveness), 6),
            n_circuit_heads=int(n_circuit_heads),
            report_id=report_id,
            auditor=auditor,
            notes=notes,
            prev_hash=prev_hash,
        )
        self._records.append(rec)
        self._append_to_disk(rec)
        logger.info(
            "Audit logged: %s | grade=%s | f1=%.3f | id=%s",
            model_name, explainability_grade, faithfulness_f1, rec.record_id,
        )
        return rec

    def append_from_result(
        self,
        result: Dict[str, Any],
        auditor: str = "",
        notes: str = "",
    ) -> AuditRecord:
        """
        Append a record from a Glassbox ``analyze()`` or API response dict.

        Compatible with the output of:
          - ``GlassboxV2.analyze()``
          - ``POST /v1/audit/analyze`` API response
          - ``POST /v1/audit/black-box`` API response

        Parameters
        ----------
        result  : Dict returned by analyze() or the REST API.
        auditor : Who triggered the audit (free text, e.g. email or username).
        notes   : Optional notes for the audit record.
        """
        faith  = result.get("faithfulness") or {}
        meta   = result.get("metadata") or {}           # GlassboxV2.analyze() output
        report = result.get("full_report") or {}
        s1 = (report.get("sections") or {}).get("1_general_description") or {}
        s2 = (report.get("sections") or {}).get("2_development_design") or {}
        s3 = (report.get("sections") or {}).get("3_monitoring_control") or {}

        # ── Grade ─────────────────────────────────────────────────────────────
        # Priority: explicit key → section 3 → derive from F1 score.
        # Raw GlassboxV2.analyze() output never contains "explainability_grade",
        # so we compute it from the faithfulness F1 rather than defaulting to D.
        grade_raw = result.get("explainability_grade") or s3.get("explainability_grade")
        if grade_raw:
            grade = grade_raw[0].upper()
        else:
            f1 = float(faith.get("f1") or s3.get("f1_score") or 0.0)
            if   f1 >= 0.80: grade = "A"
            elif f1 >= 0.65: grade = "B"
            elif f1 >= 0.50: grade = "C"
            else:            grade = "D"

        # ── model_name ────────────────────────────────────────────────────────
        # GlassboxV2.analyze() returns metadata.model_name; REST API returns
        # model_name at top level; black-box returns target_model.
        model_name = (
            result.get("model_name")
            or result.get("target_model")
            or meta.get("model_name")
            or s1.get("model_name")
            or "unknown"
        )

        # ── analysis_mode ─────────────────────────────────────────────────────
        # GlassboxV2 uses metadata.method ("taylor"|"ig"|"eap"); API uses
        # analysis_mode ("white_box"|"black_box").
        raw_mode = result.get("analysis_mode") or meta.get("method") or ""
        if "black" in raw_mode:
            analysis_mode = "black_box"
        elif raw_mode in ("white_box", "taylor", "ig", "eap", "integrated_gradients"):
            analysis_mode = "white_box"
        else:
            analysis_mode = raw_mode or "white_box"

        # ── circuit heads ─────────────────────────────────────────────────────
        # GlassboxV2 returns result["circuit"] as a list of (layer, head) tuples.
        n_circuit = (
            result.get("n_circuit_components")
            or len(result.get("circuit") or [])
            or len(s2.get("circuit_heads") or [])
        )

        # ── compliance_status ─────────────────────────────────────────────────
        # Derive from grade when not explicitly provided.
        raw_status = result.get("compliance_status") or report.get("compliance_status") or ""
        if not raw_status:
            raw_status = (
                "conditionally_compliant" if grade in ("A", "B")
                else "non_compliant"
            )

        return self.append(
            model_name=model_name,
            analysis_mode=analysis_mode,
            prompt=result.get("prompt") or result.get("decision_prompt") or "",
            correct_token=result.get("correct_token") or "",
            incorrect_token=result.get("incorrect_token") or "",
            provider_name=s1.get("provider_name") or report.get("provider_name") or "",
            deployment_context=s1.get("deployment_context") or result.get("deployment_context") or "other_high_risk",
            explainability_grade=grade,
            compliance_status=raw_status,
            faithfulness_f1=float(faith.get("f1") or s3.get("f1_score") or 0.0),
            faithfulness_sufficiency=float(faith.get("sufficiency") or s3.get("sufficiency") or 0.0),
            faithfulness_comprehensiveness=float(faith.get("comprehensiveness") or s3.get("comprehensiveness") or 0.0),
            n_circuit_heads=n_circuit,
            report_id=result.get("report_id") or report.get("report_id") or "",
            auditor=auditor,
            notes=notes,
        )

    # ── queries ───────────────────────────────────────────────────────────────

    def records(self) -> List[AuditRecord]:
        """Return all records in chronological order (oldest first)."""
        return list(self._records)

    def latest(self, n: int = 10) -> List[AuditRecord]:
        """Return the N most recent records."""
        return self._records[-n:]

    def by_model(self, model_name: str) -> List[AuditRecord]:
        """Return all records for a specific model name."""
        return [r for r in self._records if r.model_name == model_name]

    def by_grade(self, grade: str) -> List[AuditRecord]:
        """Return all records with a specific explainability grade (A/B/C/D)."""
        return [r for r in self._records if r.explainability_grade.upper() == grade.upper()]

    def non_compliant(self) -> List[AuditRecord]:
        """Return all records where compliance_status == 'non_compliant'."""
        return [r for r in self._records if r.compliance_status == "non_compliant"]

    # ── analytics ─────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary dict suitable for a compliance dashboard.

        Keys
        ----
        total_audits          : int
        grade_distribution    : {"A":n, "B":n, "C":n, "D":n}
        compliance_rate       : float (fraction NOT non_compliant)
        avg_f1                : float
        models_audited        : list of unique model names
        non_compliant_count   : int
        latest_audit_utc      : float | None
        chain_valid           : bool (hash chain integrity)
        """
        if not self._records:
            return {
                "total_audits": 0,
                "grade_distribution": {"A": 0, "B": 0, "C": 0, "D": 0},
                "compliance_rate": 0.0,
                "avg_f1": 0.0,
                "models_audited": [],
                "non_compliant_count": 0,
                "latest_audit_utc": None,
                "chain_valid": True,
            }

        grades = {"A": 0, "B": 0, "C": 0, "D": 0}
        for r in self._records:
            g = r.explainability_grade.upper()
            if g in grades:
                grades[g] += 1

        nc = len(self.non_compliant())
        avg_f1 = sum(r.faithfulness_f1 for r in self._records) / len(self._records)

        return {
            "total_audits": len(self._records),
            "grade_distribution": grades,
            "compliance_rate": round(1 - nc / len(self._records), 4),
            "avg_f1": round(avg_f1, 4),
            "models_audited": sorted({r.model_name for r in self._records}),
            "non_compliant_count": nc,
            "latest_audit_utc": self._records[-1].timestamp_utc,
            "chain_valid": self.verify_chain(),
        }

    # ── integrity ─────────────────────────────────────────────────────────────

    def verify_chain(self) -> bool:
        """
        Verify the hash chain integrity of the entire log.

        Returns True if every record's hash is valid and each prev_hash
        correctly references the previous record. Returns False if any
        tampering is detected.
        """
        for i, rec in enumerate(self._records):
            if not rec.verify():
                logger.warning("Hash mismatch at record %d (id=%s)", i, rec.record_id)
                return False
            expected_prev = self._records[i - 1].record_hash if i > 0 else ""
            if rec.prev_hash != expected_prev:
                logger.warning("Chain break at record %d (id=%s)", i, rec.record_id)
                return False
        return True

    # ── export ────────────────────────────────────────────────────────────────

    def export_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Export all records as a pretty-printed JSON array.

        Parameters
        ----------
        path : If provided, write to this file. Otherwise returns the JSON string.
        """
        data = {
            "glassbox_audit_log": {
                "version": "1.0",
                "total_records": len(self._records),
                "chain_valid": self.verify_chain(),
                "records": [r.to_dict() for r in self._records],
            }
        }
        blob = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(blob, encoding="utf-8")
            logger.info("Exported %d records to %s", len(self._records), path)
        return blob

    def export_csv(self, path: Union[str, Path]) -> None:
        """
        Export all records as CSV, suitable for import into Excel/GRC tools.

        Parameters
        ----------
        path : Output CSV file path.
        """
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in self._records:
                writer.writerow(r.to_dict())
        logger.info("Exported %d records to %s (CSV)", len(self._records), path)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"AuditLog(path={self.path!r}, records={len(self._records)})"
