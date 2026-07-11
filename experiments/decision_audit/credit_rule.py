#!/usr/bin/env python3
"""
Synthetic credit decision rule — the GROUND TRUTH for the faithfulness experiment.

WHY a known, deterministic rule (Phase 1.1 of docs/PLAN_DECISION_AUDIT.md):
    Because *we* define the rule, we can later verify that the circuit Glassbox
    discovers actually implements THIS rule (Phase 1.7) — not merely that the
    model is accurate. That is faithfulness validated against ground truth, which
    is stronger than any black-box dataset where the "true" circuit is unknown.
    It is also 100% legally clean: we own the data outright.

THE RULE (transparent, deterministic, single source of truth):

    APPROVE  iff  (credit_score >= 640) AND (dti < 0.40) AND (num_defaults == 0)
    else     DENY

  - Rule-relevant features : credit_score, dti, num_defaults
  - Distractor features     : annual_income, employment_years, age, loan_amount
    Distractors are included on purpose so Phase 1.7 can check that the discovered
    circuit responds to the RELEVANT features and ignores the irrelevant ones.

No labels are noisy: the label is exactly the rule applied to the features, so the
model has a learnable ground truth and the circuit has a known target to match.
"""

SCORE_MIN = 640
DTI_MAX = 0.40
DEFAULTS_MAX = 0

RELEVANT_FEATURES = ("credit_score", "dti", "num_defaults")
DISTRACTOR_FEATURES = ("annual_income", "employment_years", "age", "loan_amount")
ALL_FEATURES = RELEVANT_FEATURES + DISTRACTOR_FEATURES


def decide(credit_score: float, dti: float, num_defaults: int, **_ignored) -> str:
    """Return 'approved' or 'denied' per the documented rule. Deterministic.

    Extra keyword args (the distractor features) are accepted and ignored, so a
    full feature dict can be splatted in: decide(**row).
    """
    approve = (
        credit_score >= SCORE_MIN
        and dti < DTI_MAX
        and num_defaults <= DEFAULTS_MAX
    )
    return "approved" if approve else "denied"


if __name__ == "__main__":
    # tiny self-check
    assert decide(720, 0.30, 0) == "approved"
    assert decide(639, 0.30, 0) == "denied"   # score fails
    assert decide(720, 0.40, 0) == "denied"   # dti fails (strict <)
    assert decide(720, 0.30, 1) == "denied"   # default fails
    print("credit_rule self-check passed")
