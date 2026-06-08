# SPDX-License-Identifier: MIT
"""
tests/test_notify.py — coverage for Slack / Teams alerting.

notify.py is pure-Python (urllib + json, no torch). These tests capture the
outgoing payload instead of hitting the network, exercising every send method,
both channels, the AlertConfig router, and the dry-run + HTTP paths.
"""

import json

import pytest

from glassbox import notify
from glassbox.notify import AlertConfig, SlackNotifier, TeamsNotifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def capture(monkeypatch):
    """Replace _post_json with a recorder that captures (url, payload)."""
    sent = []

    def _fake(url, payload, timeout=10):
        sent.append((url, payload))
        return 200

    monkeypatch.setattr(notify, "_post_json", _fake)
    return sent


@pytest.fixture
def result():
    return {
        "faithfulness": {"sufficiency": 0.92, "comprehensiveness": 0.80, "f1": 0.85},
        "n_heads": 5,
    }


@pytest.fixture
def diff_drift():
    return {"jaccard": 0.50, "heads_added": [(0, 1), (2, 3)], "heads_removed": [(4, 5)]}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    @pytest.mark.parametrize("suff,grade", [
        (0.95, "Excellent"), (0.80, "Good"), (0.60, "Marginal"), (0.30, "Poor"),
    ])
    def test_grade(self, suff, grade):
        assert notify._grade(suff) == grade

    def test_grade_emoji_known_and_default(self):
        assert notify._grade_emoji("Excellent") == "✅"
        assert notify._grade_emoji("Nonsense") == "❔"

    def test_grade_color_known_and_default(self):
        assert notify._grade_color("Poor") == "#EF4444"
        assert notify._grade_color("Nonsense") == "#94A3B8"


# ---------------------------------------------------------------------------
# SlackNotifier
# ---------------------------------------------------------------------------

class TestSlackNotifier:
    def test_audit_complete_payload(self, capture, result):
        s = SlackNotifier("https://hooks.slack.com/x", channel="#c")
        rc = s.send_audit_complete(result, model_name="gpt2", use_case="credit",
                                   report_url="https://r")
        assert rc == 200
        url, payload = capture[-1]
        assert "blocks" in payload and payload["channel"] == "#c"
        # header block present
        assert any(b.get("type") == "header" for b in payload["blocks"])

    def test_audit_complete_without_optional(self, capture, result):
        SlackNotifier("https://x").send_audit_complete(result)
        _, payload = capture[-1]
        assert "blocks" in payload

    def test_compliance_drop_payload(self, capture):
        SlackNotifier("https://x").send_compliance_drop(
            "Excellent", "Marginal", model_name="gpt2", run_id="v1->v2",
            old_suff=0.92, new_suff=0.6, report_url="https://r")
        _, payload = capture[-1]
        blob = json.dumps(payload)
        assert "Degradation" in blob and "Marginal" in blob

    def test_compliance_drop_no_suff(self, capture):
        SlackNotifier("https://x").send_compliance_drop("Good", "Poor")
        assert capture  # sent something

    def test_circuit_drift_significant(self, capture, diff_drift):
        SlackNotifier("https://x").send_circuit_drift(diff_drift, "a", "b",
                                                      report_url="https://r")
        _, payload = capture[-1]
        assert "SIGNIFICANT DRIFT" in json.dumps(payload)

    def test_circuit_drift_minor(self, capture):
        SlackNotifier("https://x").send_circuit_drift({"jaccard": 0.9})
        _, payload = capture[-1]
        assert "MINOR DRIFT" in json.dumps(payload)

    def test_circuit_drift_many_heads_truncates(self, capture):
        diff = {"jaccard": 0.3,
                "heads_added": [(i, i) for i in range(8)],
                "heads_removed": [(i, i) for i in range(7)]}
        SlackNotifier("https://x").send_circuit_drift(diff)
        _, payload = capture[-1]
        assert "more" in json.dumps(payload)

    def test_send_raw(self, capture):
        SlackNotifier("https://x").send_raw("hello")
        _, payload = capture[-1]
        assert payload["text"] == "hello"

    def test_dry_run_prints_and_returns_200(self, capsys, result):
        rc = SlackNotifier("https://x", dry_run=True).send_audit_complete(result)
        assert rc == 200
        out = capsys.readouterr().out
        assert "DRY RUN" in out


# ---------------------------------------------------------------------------
# TeamsNotifier
# ---------------------------------------------------------------------------

class TestTeamsNotifier:
    def test_audit_complete_card(self, capture, result):
        rc = TeamsNotifier("https://outlook.office.com/x").send_audit_complete(
            result, model_name="gpt2", use_case="credit", report_url="https://r")
        assert rc == 200
        _, card = capture[-1]
        assert card["@type"] == "MessageCard"
        assert card["potentialAction"][0]["@type"] == "OpenUri"

    def test_compliance_drop_card(self, capture):
        TeamsNotifier("https://x").send_compliance_drop(
            "Excellent", "Poor", model_name="gpt2", run_id="ckpt",
            old_suff=0.9, new_suff=0.3)
        _, card = capture[-1]
        assert card["themeColor"] == "EF4444"

    def test_circuit_drift_card_drift(self, capture, diff_drift):
        TeamsNotifier("https://x").send_circuit_drift(diff_drift, "a", "b")
        _, card = capture[-1]
        assert "Circuit Drift Detected" in card["title"]

    def test_circuit_drift_card_minor(self, capture):
        TeamsNotifier("https://x").send_circuit_drift({"jaccard": 0.95})
        _, card = capture[-1]
        assert "Minor" in card["title"]

    def test_dry_run(self, capsys, result):
        rc = TeamsNotifier("https://x", dry_run=True).send_audit_complete(result)
        assert rc == 200
        assert "DRY RUN" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# AlertConfig router
# ---------------------------------------------------------------------------

class TestAlertConfig:
    def test_no_webhooks_no_notifiers(self):
        cfg = AlertConfig()
        assert cfg.slack is None and cfg.teams is None

    def test_both_webhooks_create_notifiers(self):
        cfg = AlertConfig(slack_webhook="https://s", teams_webhook="https://t")
        assert isinstance(cfg.slack, SlackNotifier)
        assert isinstance(cfg.teams, TeamsNotifier)

    def test_audit_complete_routes_to_both(self, capture, result):
        cfg = AlertConfig(slack_webhook="https://s", teams_webhook="https://t")
        cfg.notify_audit_complete(result, model_name="gpt2")
        assert len(capture) == 2  # slack + teams

    def test_audit_complete_disabled(self, capture, result):
        cfg = AlertConfig(slack_webhook="https://s", alert_on_audit_complete=False)
        cfg.notify_audit_complete(result)
        assert capture == []

    def test_compliance_drop_detection(self, capture):
        cfg = AlertConfig(slack_webhook="https://s")
        high = {"faithfulness": {"sufficiency": 0.95}, "n_heads": 3}
        low = {"faithfulness": {"sufficiency": 0.40}, "n_heads": 3}
        cfg.notify_audit_complete(high)          # first: just audit complete
        capture.clear()
        cfg.notify_audit_complete(low)           # second: drop + audit complete
        blob = json.dumps([p for _, p in capture])
        assert "Degradation" in blob

    def test_circuit_drift_within_threshold_skips(self, capture):
        cfg = AlertConfig(slack_webhook="https://s", jaccard_alert_threshold=0.75)
        cfg.notify_circuit_drift({"jaccard": 0.9})
        assert capture == []

    def test_circuit_drift_below_threshold_fires(self, capture, diff_drift):
        cfg = AlertConfig(slack_webhook="https://s", teams_webhook="https://t",
                          jaccard_alert_threshold=0.75)
        cfg.notify_circuit_drift(diff_drift, model_a="a", model_b="b")
        assert len(capture) == 2

    def test_circuit_drift_disabled(self, capture, diff_drift):
        cfg = AlertConfig(slack_webhook="https://s", alert_on_circuit_drift=False)
        cfg.notify_circuit_drift(diff_drift)
        assert capture == []

    def test_notify_failure_is_warned_not_raised(self, monkeypatch, result):
        def _boom(url, payload, timeout=10):
            raise RuntimeError("network down")
        monkeypatch.setattr(notify, "_post_json", _boom)
        cfg = AlertConfig(slack_webhook="https://s")
        with pytest.warns(UserWarning):
            cfg.notify_audit_complete(result)  # must not raise


# ---------------------------------------------------------------------------
# _post_json (HTTP path) via mocked urlopen
# ---------------------------------------------------------------------------

class TestPostJson:
    def test_post_json_returns_status(self, monkeypatch):
        class _Resp:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr(notify.urllib.request, "urlopen", lambda req, timeout=10: _Resp())
        rc = notify._post_json("https://x", {"a": 1})
        assert rc == 200
