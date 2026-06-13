"""Tests for glassbox/causal_abstraction.py — interchange accuracy + certificate."""
import pytest

from glassbox.causal_abstraction import certify_abstraction, interchange_accuracy


def test_interchange_accuracy_perfect_and_half():
    assert interchange_accuracy([("a", "a"), ("b", "b")]) == 1.0
    assert interchange_accuracy([("a", "a"), ("b", "c")]) == 0.5


def test_interchange_accuracy_empty_raises():
    with pytest.raises(ValueError):
        interchange_accuracy([])


def test_certify_pass():
    trials = [("approve", "approve")] * 9 + [("approve", "deny")]  # 0.9
    cert = certify_abstraction(trials, threshold=0.80)
    assert cert["interchange_accuracy"] == pytest.approx(0.9)
    assert cert["certified"] is True
    assert cert["tier_eligible"].startswith("A")
    assert cert["n_trials"] == 10


def test_certify_fail():
    trials = [("approve", "approve")] * 5 + [("approve", "deny")] * 5  # 0.5
    cert = certify_abstraction(trials, threshold=0.80)
    assert cert["certified"] is False
    assert cert["tier_eligible"] == "below A"
    assert "do not claim" in cert["note"]
