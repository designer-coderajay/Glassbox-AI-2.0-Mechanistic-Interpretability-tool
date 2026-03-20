"""
glassbox.risk_register
======================
Persistent risk register that tracks compliance documentation risks identified
across multiple AnnexIV report drafts and audit sessions.

Stores risks in a JSON file. Each entry records which model produced the risk,
which EU AI Act article it maps to, severity, status, and when it was last seen.
Supports deduplication, status tracking (open/mitigated/accepted), and
trend reporting so compliance officers can see whether risk is improving.

LEGAL NOTICE — DOCUMENTATION AID ONLY
---------------------------------------
This module is a software tool for managing risk documentation drafts.
It does not constitute legal advice, certify regulatory compliance, or
establish any professional advisory relationship. Risk entries and severity
classifications are internal documentation aids only — they are not
official regulatory determinations. Whether identified items constitute
actual legal risk under Regulation (EU) 2024/1689 or any other applicable
law is a matter for qualified legal counsel.

Regulatory References (informational only)
------------------------------------------
Regulation (EU) 2024/1689 — EU AI Act:
  Article 9              — Risk management system (obligation for high-risk AI)
  Annex IV, Section 5    — Risk management documentation requirements
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

__all__ = ["RiskEntry", "RiskRegister"]

RiskStatus   = Literal["open", "mitigated", "accepted", "escalated"]
RiskSeverity = Literal["critical", "high", "medium", "low", "info"]

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


class RiskEntry:
    """
    A single risk record in the register.

    Parameters
    ----------
    risk_id         : unique identifier (auto-generated if not supplied)
    description     : human-readable description of the risk
    model_name      : model that triggered this risk
    article         : EU AI Act article reference (e.g. 'Article 13')
    severity        : critical | high | medium | low | info
    status          : open | mitigated | accepted | escalated
    first_seen      : ISO timestamp when first detected
    last_seen       : ISO timestamp of most recent detection
    occurrences     : how many audits have surfaced this risk
    notes           : optional free-text notes (mitigation steps, decisions)
    """

    def __init__(
        self,
        description:  str,
        model_name:   str                    = "unknown",
        article:      str                    = "Annex IV, Section 5",
        severity:     RiskSeverity           = "medium",
        status:       RiskStatus             = "open",
        risk_id:      Optional[str]          = None,
        first_seen:   Optional[str]          = None,
        last_seen:    Optional[str]          = None,
        occurrences:  int                    = 1,
        notes:        Optional[str]          = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.risk_id     = risk_id    or str(uuid.uuid4())[:12]
        self.description = description
        self.model_name  = model_name
        self.article     = article
        self.severity    = severity
        self.status      = status
        self.first_seen  = first_seen or now
        self.last_seen   = last_seen  or now
        self.occurrences = occurrences
        self.notes       = notes or ""

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_id":     self.risk_id,
            "description": self.description,
            "model_name":  self.model_name,
            "article":     self.article,
            "severity":    self.severity,
            "status":      self.status,
            "first_seen":  self.first_seen,
            "last_seen":   self.last_seen,
            "occurrences": self.occurrences,
            "notes":       self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskEntry":
        return cls(
            description = d["description"],
            model_name  = d.get("model_name", "unknown"),
            article     = d.get("article", "Annex IV, Section 5"),
            severity    = d.get("severity", "medium"),
            status      = d.get("status", "open"),
            risk_id     = d.get("risk_id"),
            first_seen  = d.get("first_seen"),
            last_seen   = d.get("last_seen"),
            occurrences = d.get("occurrences", 1),
            notes       = d.get("notes", ""),
        )

    def __repr__(self) -> str:
        return (f"RiskEntry(id={self.risk_id!r}, severity={self.severity!r}, "
                f"status={self.status!r}, model={self.model_name!r})")


class RiskRegister:
    """
    Persistent risk register — tracks compliance risks across audit sessions.

    Usage
    -----
    ::

        from glassbox import RiskRegister

        rr = RiskRegister("risks.json")

        # Add risks from an AnnexIVReport
        rr.ingest_annex_report(annex, model_name="gpt2")

        # Manually add a risk
        rr.add("F1 < 0.50 for 3 consecutive audits — Article 13 breach risk",
               model_name="gpt2", severity="high", article="Article 13")

        # Query
        open_risks   = rr.open_risks()
        critical     = rr.by_severity("critical")
        trend        = rr.trend_summary()

        # Update status
        rr.set_status(risk_id, "mitigated", notes="Retrained with more data")

        # Persist
        rr.save()

    Parameters
    ----------
    path : Path or str — JSON file to persist the register.
           Created automatically if it doesn't exist.
    """

    def __init__(self, path: str = "glassbox_risks.json") -> None:
        self._path: Path          = Path(path)
        self._risks: Dict[str, RiskEntry] = {}
        if self._path.exists():
            self._load()

    # ── Core operations ────────────────────────────────────────────────────────

    def add(
        self,
        description:  str,
        model_name:   str           = "unknown",
        article:      str           = "Annex IV, Section 5",
        severity:     RiskSeverity  = "medium",
        notes:        str           = "",
        deduplicate:  bool          = True,
    ) -> RiskEntry:
        """
        Add a risk to the register.

        If ``deduplicate=True`` (default) and a risk with the same description
        and model already exists, the existing entry's occurrence count and
        last_seen timestamp are updated instead of creating a duplicate.

        Returns the RiskEntry (new or updated).
        """
        if deduplicate:
            for entry in self._risks.values():
                if (entry.description.strip().lower() == description.strip().lower()
                        and entry.model_name == model_name):
                    entry.occurrences += 1
                    entry.last_seen    = datetime.now(timezone.utc).isoformat()
                    logger.debug("RiskRegister: updated occurrence count for %s", entry.risk_id)
                    self.save()
                    return entry

        entry = RiskEntry(
            description = description,
            model_name  = model_name,
            article     = article,
            severity    = severity,
            notes       = notes,
        )
        self._risks[entry.risk_id] = entry
        logger.info("RiskRegister: new risk added %s — %s", entry.risk_id, description[:60])
        self.save()
        return entry

    def ingest_annex_report(
        self,
        annex: Any,
        model_name: Optional[str] = None,
    ) -> List[RiskEntry]:
        """
        Extract risk flags from an AnnexIVReport and add them to the register.

        Works with the to_json() output so no private attribute access is needed.

        Parameters
        ----------
        annex      : AnnexIVReport instance (must have .to_json() method)
        model_name : override model name; falls back to the value in the report

        Returns list of RiskEntry objects added or updated.
        """
        import json as _json

        rj  = _json.loads(annex.to_json())
        s5  = rj.get("sections", {}).get("5_risk_management", {})
        s3  = rj.get("sections", {}).get("3_monitoring_control", {})
        s1  = rj.get("sections", {}).get("1_general_description", {})

        resolved_model = (
            model_name
            or s1.get("ai_system_name")
            or rj.get("model_name")
            or "unknown"
        )

        added: List[RiskEntry] = []

        # Faithfulness risk flag
        if s5.get("faithfulness_risk_flag"):
            f1 = float(s3.get("f1_score") or 0)
            entry = self.add(
                description = f"Faithfulness F1={f1:.2f} below 0.50 threshold — Article 13 transparency risk",
                model_name  = resolved_model,
                article     = "Article 13",
                severity    = "high" if f1 < 0.30 else "medium",
            )
            added.append(entry)

        # Identified risks from Section 5
        for r in (s5.get("identified_risks") or []):
            if isinstance(r, dict):
                desc     = r.get("risk") or r.get("description") or str(r)
                art      = r.get("article") or "Annex IV, Section 5"
                sev_raw  = r.get("severity") or "medium"
                sev      = sev_raw if sev_raw in _SEVERITY_ORDER else "medium"
            else:
                desc = str(r)
                art  = "Annex IV, Section 5"
                sev  = "medium"

            if desc:
                entry = self.add(
                    description = desc,
                    model_name  = resolved_model,
                    article     = art,
                    severity    = sev,
                )
                added.append(entry)

        return added

    # ── Query ─────────────────────────────────────────────────────────────────

    def all_risks(self) -> List[RiskEntry]:
        """Return all risks sorted by severity then first_seen."""
        return sorted(
            self._risks.values(),
            key=lambda r: (_SEVERITY_ORDER.get(r.severity, 99), r.first_seen),
        )

    def open_risks(self) -> List[RiskEntry]:
        """Return all risks with status='open'."""
        return [r for r in self.all_risks() if r.status == "open"]

    def by_severity(self, severity: RiskSeverity) -> List[RiskEntry]:
        """Return all risks matching the given severity."""
        return [r for r in self.all_risks() if r.severity == severity]

    def by_model(self, model_name: str) -> List[RiskEntry]:
        """Return all risks for a given model."""
        return [r for r in self.all_risks() if r.model_name == model_name]

    def by_status(self, status: RiskStatus) -> List[RiskEntry]:
        """Return all risks with the given status."""
        return [r for r in self.all_risks() if r.status == status]

    def get(self, risk_id: str) -> Optional[RiskEntry]:
        """Return a specific risk by ID."""
        return self._risks.get(risk_id)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def set_status(
        self,
        risk_id: str,
        status:  RiskStatus,
        notes:   str = "",
    ) -> RiskEntry:
        """
        Update the status of a risk entry.

        Parameters
        ----------
        risk_id : the risk_id to update
        status  : new status (open | mitigated | accepted | escalated)
        notes   : optional note to append (e.g. 'Retrained with more data')

        Raises ValueError if risk_id not found.
        """
        entry = self._risks.get(risk_id)
        if entry is None:
            raise ValueError(f"RiskRegister: risk_id {risk_id!r} not found")
        entry.status = status
        if notes:
            ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            entry.notes = f"{entry.notes}\n[{ts}] {notes}".strip()
        self.save()
        logger.info("RiskRegister: %s → %s", risk_id, status)
        return entry

    def remove(self, risk_id: str) -> None:
        """
        Remove a risk entry entirely.
        Prefer set_status('mitigated') for audit trail continuity.
        """
        if risk_id in self._risks:
            del self._risks[risk_id]
            self.save()

    # ── Analytics ─────────────────────────────────────────────────────────────

    def trend_summary(self) -> Dict[str, Any]:
        """
        Return a summary dict suitable for dashboards and compliance reports.

        Keys
        ----
        total, open, mitigated, accepted, escalated,
        by_severity (dict), by_model (dict),
        compliance_health (str: green | amber | red)
        """
        all_r = self.all_risks()
        by_sev: Dict[str, int] = {s: 0 for s in _SEVERITY_ORDER}
        by_mod: Dict[str, int] = {}

        for r in all_r:
            by_sev[r.severity] = by_sev.get(r.severity, 0) + 1
            by_mod[r.model_name] = by_mod.get(r.model_name, 0) + 1

        open_r     = [r for r in all_r if r.status == "open"]
        critical_o = sum(1 for r in open_r if r.severity == "critical")
        high_o     = sum(1 for r in open_r if r.severity == "high")

        if critical_o > 0:
            health = "red"
        elif high_o > 0:
            health = "amber"
        else:
            health = "green"

        return {
            "total":             len(all_r),
            "open":              len(open_r),
            "mitigated":         sum(1 for r in all_r if r.status == "mitigated"),
            "accepted":          sum(1 for r in all_r if r.status == "accepted"),
            "escalated":         sum(1 for r in all_r if r.status == "escalated"),
            "by_severity":       by_sev,
            "by_model":          by_mod,
            "compliance_health": health,
        }

    def to_markdown(self) -> str:
        """
        Render the risk register as a markdown table.
        Useful for embedding in Annex IV reports or PR comments.
        """
        rows = self.all_risks()
        if not rows:
            return "_No risks recorded._\n"

        lines = [
            "## Risk Register",
            "",
            "| ID | Severity | Status | Model | Article | Description |",
            "|----|----------|--------|-------|---------|-------------|",
        ]
        for r in rows:
            desc = r.description[:80] + ("…" if len(r.description) > 80 else "")
            lines.append(
                f"| `{r.risk_id}` | **{r.severity}** | {r.status} "
                f"| {r.model_name} | {r.article} | {desc} |"
            )

        summary = self.trend_summary()
        health_icon = {"green": "✅", "amber": "⚠️", "red": "❌"}.get(
            summary["compliance_health"], "❓"
        )

        lines += [
            "",
            f"**Compliance health:** {health_icon} {summary['compliance_health'].upper()}  ",
            f"**Total:** {summary['total']} &nbsp;|&nbsp; "
            f"**Open:** {summary['open']} &nbsp;|&nbsp; "
            f"**Mitigated:** {summary['mitigated']}",
        ]
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        """Serialize the full register to JSON."""
        return json.dumps(
            {
                "schema_version": "1.0",
                "regulation": "EU AI Act (EU) 2024/1689",
                "article_ref": "Article 9 — Risk management system",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": self.trend_summary(),
                "risks": [r.to_dict() for r in self.all_risks()],
            },
            indent=2,
            default=str,
        )

    def save(self) -> None:
        """Persist the register to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self.to_json(), encoding="utf-8")

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for r_dict in data.get("risks", []):
                entry = RiskEntry.from_dict(r_dict)
                self._risks[entry.risk_id] = entry
            logger.info(
                "RiskRegister: loaded %d risks from %s", len(self._risks), self._path
            )
        except Exception as exc:
            logger.warning("RiskRegister: could not load %s — %s", self._path, exc)

    def __len__(self) -> int:
        return len(self._risks)

    def __repr__(self) -> str:
        s = self.trend_summary()
        return (f"RiskRegister(path={str(self._path)!r}, "
                f"total={s['total']}, open={s['open']}, "
                f"health={s['compliance_health']!r})")
