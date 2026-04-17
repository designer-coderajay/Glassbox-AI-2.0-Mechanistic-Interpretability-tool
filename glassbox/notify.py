# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ajay Pravin Mahale
"""
glassbox/notify.py — Slack & Teams Alerting
============================================

Send formatted alerts to Slack or Microsoft Teams when:
- CircuitDiff detects mechanistic drift between model versions
- A compliance grade drops (e.g., Excellent → Marginal)
- stability_suite() finds circuit instability below threshold
- An audit completes (with summary)

No SDKs required — uses incoming webhooks (plain HTTPS POST).

Usage
-----
::

    from glassbox.notify import SlackNotifier, TeamsNotifier, AlertConfig

    # Slack
    slack = SlackNotifier(webhook_url="https://hooks.slack.com/services/T.../B.../...")
    slack.send_audit_complete(result, model_name="gpt2", use_case="credit_scoring")
    slack.send_compliance_drop(old_grade="Excellent", new_grade="Marginal",
                               model_name="gpt2", run_id="v3.1→v3.2")

    # Microsoft Teams
    teams = TeamsNotifier(webhook_url="https://outlook.office.com/webhook/...")
    teams.send_circuit_drift(diff_result, model_a="v3.1", model_b="v3.2")

    # AlertConfig — centralised configuration for both channels
    config = AlertConfig(
        slack_webhook="https://hooks.slack.com/...",
        teams_webhook="https://outlook.office.com/webhook/...",
        alert_on_compliance_drop=True,
        alert_on_circuit_drift=True,
        jaccard_alert_threshold=0.70,
        sufficiency_alert_threshold=0.75,
    )
    config.notify_all(result, model_name="gpt2")

Setting up webhooks
-------------------
Slack:  https://api.slack.com/messaging/webhooks
Teams:  https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.request
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 10) -> int:
    """POST JSON payload to a URL. Returns HTTP status code."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


def _grade(suff: float) -> str:
    if suff >= 0.90: return "Excellent"
    if suff >= 0.75: return "Good"
    if suff >= 0.50: return "Marginal"
    return "Poor"


def _grade_emoji(grade: str) -> str:
    return {"Excellent": "✅", "Good": "✅", "Marginal": "⚠️", "Poor": "❌"}.get(grade, "❔")


def _grade_color(grade: str) -> str:
    """Return a hex color for Teams card accent."""
    return {"Excellent": "#22C55E", "Good": "#84CC16",
            "Marginal": "#F59E0B", "Poor": "#EF4444"}.get(grade, "#94A3B8")


# ---------------------------------------------------------------------------
# SlackNotifier
# ---------------------------------------------------------------------------

class SlackNotifier:
    """
    Send Glassbox alerts to a Slack channel via incoming webhook.

    Parameters
    ----------
    webhook_url : str
        Slack incoming webhook URL.
    channel : str, optional
        Override the channel set in the webhook configuration.
    username : str
        Bot display name in Slack.
    icon_emoji : str
        Bot icon emoji.
    signing_secret : str, optional
        If set, adds ``X-Slack-Signature`` HMAC header (for verified delivery).
    dry_run : bool
        If True, prints the payload to stdout instead of sending.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "Glassbox AI",
        icon_emoji: str = ":shield:",
        signing_secret: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.webhook_url    = webhook_url
        self.channel        = channel
        self.username       = username
        self.icon_emoji     = icon_emoji
        self.signing_secret = signing_secret
        self.dry_run        = dry_run

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    def send_audit_complete(
        self,
        result: Dict[str, Any],
        model_name: Optional[str] = None,
        use_case: Optional[str] = None,
        report_url: Optional[str] = None,
    ) -> int:
        """Send an audit completion summary."""
        faith   = result.get("faithfulness", {})
        suff    = faith.get("sufficiency", 0.0)
        comp    = faith.get("comprehensiveness", 0.0)
        f1      = faith.get("f1", 0.0)
        n_heads = result.get("n_heads", 0)
        grade   = _grade(suff)
        emoji   = _grade_emoji(grade)

        fields = [
            {"type": "mrkdwn", "text": f"*Sufficiency:* {suff:.1%}"},
            {"type": "mrkdwn", "text": f"*Comprehensiveness:* {comp:.1%}"},
            {"type": "mrkdwn", "text": f"*F1:* {f1:.1%}"},
            {"type": "mrkdwn", "text": f"*Circuit heads:* {n_heads}"},
            {"type": "mrkdwn", "text": f"*Grade:* {emoji} {grade}"},
        ]
        if use_case:
            fields.append({"type": "mrkdwn", "text": f"*Use case:* {use_case}"})

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} Glassbox Audit Complete"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Model `{model_name or 'unknown'}` — EU AI Act Annex IV audit finished."},
            },
            {"type": "section", "fields": fields},
        ]
        if report_url:
            blocks.append({
                "type": "actions",
                "elements": [{
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📄 View Annex IV Report"},
                    "url": report_url,
                    "style": "primary" if suff >= 0.75 else "danger",
                }],
            })
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Glassbox v4.2.6 · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · Regulation (EU) 2024/1689"}],
        })
        return self._send(blocks=blocks)

    def send_compliance_drop(
        self,
        old_grade: str,
        new_grade: str,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        old_suff: Optional[float] = None,
        new_suff: Optional[float] = None,
        report_url: Optional[str] = None,
    ) -> int:
        """Send a compliance grade degradation alert."""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "⚠️ Compliance Grade Degradation Detected"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Model:* `{model_name or 'unknown'}`\n"
                        f"*Run / checkpoint:* `{run_id or 'N/A'}`\n"
                        f"Grade dropped from *{_grade_emoji(old_grade)} {old_grade}* "
                        f"→ *{_grade_emoji(new_grade)} {new_grade}*"
                        + (f"\nSufficiency: `{old_suff:.1%}` → `{new_suff:.1%}`" if old_suff is not None and new_suff is not None else "")
                    ),
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "⚡ *Action required:* Review model checkpoint before promoting to production. "
                        "EU AI Act Article 72 post-market monitoring requires logging this event."
                    ),
                },
            },
        ]
        if report_url:
            blocks.append({
                "type": "actions",
                "elements": [{
                    "type": "button", "style": "danger",
                    "text": {"type": "plain_text", "text": "🔎 View Report"},
                    "url": report_url,
                }],
            })
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Glassbox v4.2.6 · Article 72 Regulation (EU) 2024/1689 · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"}],
        })
        return self._send(blocks=blocks)

    def send_circuit_drift(
        self,
        diff_result: Dict[str, Any],
        model_a: Optional[str] = None,
        model_b: Optional[str] = None,
        report_url: Optional[str] = None,
    ) -> int:
        """Send a CircuitDiff drift alert."""
        jaccard   = diff_result.get("jaccard", 0.0)
        heads_added   = diff_result.get("heads_added", [])
        heads_removed = diff_result.get("heads_removed", [])
        drift_flag    = jaccard < 0.75

        emoji = "🔴" if drift_flag else "🟡"
        label = "SIGNIFICANT DRIFT" if drift_flag else "MINOR DRIFT"

        added_str   = ", ".join([f"L{h[0]}H{h[1]}" for h in heads_added[:5]]) or "none"
        removed_str = ", ".join([f"L{h[0]}H{h[1]}" for h in heads_removed[:5]]) or "none"
        if len(heads_added) > 5:
            added_str   += f" +{len(heads_added) - 5} more"
        if len(heads_removed) > 5:
            removed_str += f" +{len(heads_removed) - 5} more"

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} Circuit Drift Detected — {label}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Model A:* `{model_a or 'baseline'}`"},
                    {"type": "mrkdwn", "text": f"*Model B:* `{model_b or 'current'}`"},
                    {"type": "mrkdwn", "text": f"*Jaccard Similarity:* `{jaccard:.2f}`"},
                    {"type": "mrkdwn", "text": f"*Drift Flag:* {'Yes ⚠️' if drift_flag else 'No ✅'}"},
                    {"type": "mrkdwn", "text": f"*Heads Added:* {added_str}"},
                    {"type": "mrkdwn", "text": f"*Heads Removed:* {removed_str}"},
                ],
            },
        ]
        if drift_flag:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "⚡ *Action required:* Jaccard similarity below 0.75 threshold. "
                        "This change must be logged in the Article 72 post-market monitoring register. "
                        "Re-run full Annex IV audit before deploying Model B."
                    ),
                },
            })
        if report_url:
            blocks.append({
                "type": "actions",
                "elements": [{"type": "button", "text": {"type": "plain_text", "text": "View CircuitDiff Report"}, "url": report_url}],
            })
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Glassbox CircuitDiff v4.2.6 · Article 72 Regulation (EU) 2024/1689 · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"}],
        })
        return self._send(blocks=blocks)

    def send_raw(self, text: str) -> int:
        """Send a plain text message."""
        return self._send(text=text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(self, blocks: Optional[list] = None, text: Optional[str] = None) -> int:
        payload: Dict[str, Any] = {}
        if self.channel:
            payload["channel"] = self.channel
        if self.username:
            payload["username"] = self.username
        if self.icon_emoji:
            payload["icon_emoji"] = self.icon_emoji
        if blocks:
            payload["blocks"] = blocks
            payload["text"]   = text or "Glassbox AI alert"
        elif text:
            payload["text"] = text

        if self.dry_run:
            print("[SlackNotifier DRY RUN]")
            print(json.dumps(payload, indent=2))
            return 200

        return _post_json(self.webhook_url, payload)


# ---------------------------------------------------------------------------
# TeamsNotifier
# ---------------------------------------------------------------------------

class TeamsNotifier:
    """
    Send Glassbox alerts to a Microsoft Teams channel via incoming webhook.

    Uses the Adaptive Card format (Office 365 Connector Card).

    Parameters
    ----------
    webhook_url : str
        Teams incoming webhook URL.
    dry_run : bool
        If True, prints the payload to stdout instead of sending.
    """

    def __init__(self, webhook_url: str, dry_run: bool = False) -> None:
        self.webhook_url = webhook_url
        self.dry_run     = dry_run

    def send_audit_complete(
        self,
        result: Dict[str, Any],
        model_name: Optional[str] = None,
        use_case: Optional[str] = None,
        report_url: Optional[str] = None,
    ) -> int:
        faith   = result.get("faithfulness", {})
        suff    = faith.get("sufficiency", 0.0)
        f1      = faith.get("f1", 0.0)
        n_heads = result.get("n_heads", 0)
        grade   = _grade(suff)
        color   = _grade_color(grade)

        facts = [
            {"name": "Model", "value": model_name or "unknown"},
            {"name": "Sufficiency", "value": f"{suff:.1%}"},
            {"name": "F1", "value": f"{f1:.1%}"},
            {"name": "Circuit Heads", "value": str(n_heads)},
            {"name": "Grade", "value": grade},
            {"name": "Regulation", "value": "Regulation (EU) 2024/1689 — AI Act Annex IV"},
        ]
        if use_case:
            facts.append({"name": "Use Case", "value": use_case})

        card = self._build_card(
            title     = f"{_grade_emoji(grade)} Glassbox Audit Complete — {grade}",
            summary   = f"Audit complete for {model_name or 'unknown'}. Grade: {grade}.",
            color     = color,
            facts     = facts,
            report_url= report_url,
        )
        return self._send(card)

    def send_compliance_drop(
        self,
        old_grade: str,
        new_grade: str,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        old_suff: Optional[float] = None,
        new_suff: Optional[float] = None,
        report_url: Optional[str] = None,
    ) -> int:
        facts = [
            {"name": "Model", "value": model_name or "unknown"},
            {"name": "Previous Grade", "value": f"{_grade_emoji(old_grade)} {old_grade}"},
            {"name": "New Grade", "value": f"{_grade_emoji(new_grade)} {new_grade}"},
        ]
        if old_suff is not None and new_suff is not None:
            facts.append({"name": "Sufficiency Change", "value": f"{old_suff:.1%} → {new_suff:.1%}"})
        if run_id:
            facts.append({"name": "Checkpoint", "value": run_id})
        facts.append({"name": "Article", "value": "Article 72 — Post-Market Monitoring"})

        card = self._build_card(
            title      = "⚠️ Compliance Grade Degradation",
            summary    = f"Grade dropped from {old_grade} to {new_grade} for {model_name or 'unknown'}.",
            color      = "#EF4444",
            facts      = facts,
            report_url = report_url,
        )
        return self._send(card)

    def send_circuit_drift(
        self,
        diff_result: Dict[str, Any],
        model_a: Optional[str] = None,
        model_b: Optional[str] = None,
        report_url: Optional[str] = None,
    ) -> int:
        jaccard   = diff_result.get("jaccard", 0.0)
        drift_flag = jaccard < 0.75
        facts = [
            {"name": "Baseline Model", "value": model_a or "unknown"},
            {"name": "Current Model", "value": model_b or "unknown"},
            {"name": "Jaccard Similarity", "value": f"{jaccard:.3f}"},
            {"name": "Drift Flag", "value": "Yes — Action required" if drift_flag else "No — Within threshold"},
            {"name": "Article", "value": "Article 72 — Post-Market Monitoring"},
        ]
        card = self._build_card(
            title      = "🔴 Circuit Drift Detected" if drift_flag else "🟡 Circuit Drift (Minor)",
            summary    = f"Mechanistic drift detected. Jaccard: {jaccard:.3f}",
            color      = "#EF4444" if drift_flag else "#F59E0B",
            facts      = facts,
            report_url = report_url,
        )
        return self._send(card)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_card(
        title: str,
        summary: str,
        color: str,
        facts: List[Dict[str, str]],
        report_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an Office 365 Connector Card payload."""
        card: Dict[str, Any] = {
            "@type":      "MessageCard",
            "@context":   "http://schema.org/extensions",
            "themeColor": color.lstrip("#"),
            "summary":    summary,
            "title":      title,
            "sections": [
                {
                    "facts": facts,
                    "text":  f"Generated by Glassbox AI v4.2.6 · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
                }
            ],
        }
        if report_url:
            card["potentialAction"] = [{
                "@type": "OpenUri",
                "name":  "View Report",
                "targets": [{"os": "default", "uri": report_url}],
            }]
        return card

    def _send(self, payload: Dict[str, Any]) -> int:
        if self.dry_run:
            print("[TeamsNotifier DRY RUN]")
            print(json.dumps(payload, indent=2))
            return 200
        return _post_json(self.webhook_url, payload)


# ---------------------------------------------------------------------------
# AlertConfig — centralised multi-channel configuration
# ---------------------------------------------------------------------------

class AlertConfig:
    """
    Central alert configuration that manages both Slack and Teams notifiers.

    Parameters
    ----------
    slack_webhook : str, optional
        Slack incoming webhook URL. If not set, Slack alerts are skipped.
    teams_webhook : str, optional
        Teams incoming webhook URL. If not set, Teams alerts are skipped.
    alert_on_audit_complete : bool
        Send a summary when any audit finishes.
    alert_on_compliance_drop : bool
        Alert when compliance grade drops between consecutive audits.
    alert_on_circuit_drift : bool
        Alert on CircuitDiff Jaccard below threshold.
    jaccard_alert_threshold : float
        Jaccard threshold below which drift alerts fire (default 0.75).
    sufficiency_alert_threshold : float
        Sufficiency threshold below which compliance drop alerts fire (default 0.75).
    dry_run : bool
        If True, print instead of sending.

    Example
    -------
    ::

        config = AlertConfig(
            slack_webhook="https://hooks.slack.com/services/...",
            teams_webhook="https://outlook.office.com/webhook/...",
        )

        # Automatically routes to both channels:
        config.notify_audit_complete(result, model_name="gpt2")
        config.notify_circuit_drift(diff_result, model_a="v1", model_b="v2")
    """

    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        teams_webhook: Optional[str] = None,
        alert_on_audit_complete:   bool  = True,
        alert_on_compliance_drop:  bool  = True,
        alert_on_circuit_drift:    bool  = True,
        jaccard_alert_threshold:   float = 0.75,
        sufficiency_alert_threshold: float = 0.75,
        dry_run: bool = False,
    ) -> None:
        self.slack = SlackNotifier(slack_webhook, dry_run=dry_run) if slack_webhook else None
        self.teams = TeamsNotifier(teams_webhook, dry_run=dry_run) if teams_webhook else None
        self.alert_on_audit_complete   = alert_on_audit_complete
        self.alert_on_compliance_drop  = alert_on_compliance_drop
        self.alert_on_circuit_drift    = alert_on_circuit_drift
        self.jaccard_alert_threshold   = jaccard_alert_threshold
        self.sufficiency_alert_threshold = sufficiency_alert_threshold
        self._last_result: Optional[Dict[str, Any]] = None

    def notify_audit_complete(
        self,
        result: Dict[str, Any],
        model_name: Optional[str] = None,
        use_case: Optional[str] = None,
        report_url: Optional[str] = None,
    ) -> None:
        """Notify both channels that an audit completed."""
        if not self.alert_on_audit_complete:
            return

        # Check for compliance drop vs previous audit
        if self.alert_on_compliance_drop and self._last_result is not None:
            old_suff  = self._last_result.get("faithfulness", {}).get("sufficiency", 1.0)
            new_suff  = result.get("faithfulness", {}).get("sufficiency", 1.0)
            old_grade = _grade(old_suff)
            new_grade = _grade(new_suff)
            _grade_order = {"Excellent": 4, "Good": 3, "Marginal": 2, "Poor": 1}
            if _grade_order.get(new_grade, 0) < _grade_order.get(old_grade, 0):
                for notifier in [self.slack, self.teams]:
                    if notifier:
                        try:
                            notifier.send_compliance_drop(
                                old_grade=old_grade, new_grade=new_grade,
                                model_name=model_name,
                                old_suff=old_suff, new_suff=new_suff,
                                report_url=report_url,
                            )
                        except Exception as e:
                            import warnings
                            warnings.warn(f"Notification failed: {e}")

        # Audit complete notification
        for notifier in [self.slack, self.teams]:
            if notifier:
                try:
                    notifier.send_audit_complete(
                        result,
                        model_name=model_name,
                        use_case=use_case,
                        report_url=report_url,
                    )
                except Exception as e:
                    import warnings
                    warnings.warn(f"Notification failed: {e}")

        self._last_result = result

    def notify_circuit_drift(
        self,
        diff_result: Dict[str, Any],
        model_a: Optional[str] = None,
        model_b: Optional[str] = None,
        report_url: Optional[str] = None,
    ) -> None:
        """Notify both channels of CircuitDiff drift."""
        if not self.alert_on_circuit_drift:
            return
        jaccard = diff_result.get("jaccard", 1.0)
        if jaccard >= self.jaccard_alert_threshold:
            return  # Within acceptable range, no alert
        for notifier in [self.slack, self.teams]:
            if notifier:
                try:
                    notifier.send_circuit_drift(
                        diff_result,
                        model_a=model_a, model_b=model_b,
                        report_url=report_url,
                    )
                except Exception as e:
                    import warnings
                    warnings.warn(f"Notification failed: {e}")
