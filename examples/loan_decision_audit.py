#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
examples/loan_decision_audit.py
===============================
The complete real-world workflow, end to end, in one file:

    high-risk decision  ->  causal circuit  ->  faithfulness + grade  ->  Annex IV report

This answers the question "how does anyone actually USE this?". A user points
Glassbox at their model, frames one regulated decision as a contrast between two
outcomes, calls analyze(), and gets both the mechanistic evidence and the EU AI
Act Annex IV paperwork — as files they can hand to a regulator.

It runs on gpt2 so it reproduces on a laptop (already in .venv-torch).

────────────────────────────────────────────────────────────────────────────
HONEST NOTE — read this before trusting any number below.

gpt2-small is NOT a loan-underwriting model. It has no idea how to underwrite a
loan, so the *decision* it makes here is meaningless. What IS real and meaningful
is the PIPELINE: a real model, real activations/gradients, a real causal circuit,
real faithfulness metrics, and a real Annex IV report file. On a customer's own
fine-tuned underwriting model, these exact three lines produce a meaningful audit
— only the model on line 1 changes. This file demonstrates the plumbing, not a
credit decision.

Also note the hard scope: this works because gpt2 is an OPEN-WEIGHT model we can
load and read gradients from. The same is true for a self-hosted Llama / Mistral
/ Phi. It does NOT work through a closed API (GPT-4 / Claude) — you cannot get
activations or gradients out of an API, so those only get the weaker black-box
tier, not this circuit-level audit.
────────────────────────────────────────────────────────────────────────────

Run:
    python examples/loan_decision_audit.py
Outputs (git-ignored):
    reports/loan-annex-iv.json   machine-readable evidence vault
    reports/loan-annex-iv.html   human-readable Annex IV report (open in a browser)
"""
from __future__ import annotations

import os
import sys

# Convenience for running straight from a repo checkout
# (`python examples/loan_decision_audit.py`): put the repo root — which contains
# the `glassbox/` package — on the import path. End users who
# `pip install glassbox-mech-interp` don't need this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _grade(f1: float) -> str:
    """Same F1->grade thresholds the dashboard/report use."""
    if f1 >= 0.80:
        return "A"
    if f1 >= 0.65:
        return "B"
    if f1 >= 0.50:
        return "C"
    return "D"


def main() -> int:
    from transformer_lens import HookedTransformer

    from glassbox import GlassboxV2, build_annex_iv_vault

    # 1. THE MODEL UNDER AUDIT.
    #    Swap "gpt2" for any TransformerLens-loadable open-weight model — this is
    #    where the customer's own self-hosted model goes.
    print("Loading model under audit ...")
    model = HookedTransformer.from_pretrained("gpt2")
    gb = GlassboxV2(model)

    # 2. FRAME THE REGULATED DECISION AS A CONTRAST.
    #    This is the key modelling step: a high-risk decision must be expressed
    #    as a choice between outcomes (approve vs deny), because the audit
    #    measures which internal components drive the model toward one over the
    #    other. Free-form chat has no contrast and cannot be audited this way.
    prompt = (
        "Applicant: 34 years old, annual income EUR 82000, employed 6 years, "
        "credit score 740, existing debt EUR 5000, requested loan EUR 15000. "
        "Decision: the loan application is"
    )
    outcome_a = " approved"
    outcome_b = " denied"

    # 3. ONE CALL — causal circuit + faithfulness + grade.
    print("Running circuit + faithfulness audit ...")
    result = gb.analyze(prompt, outcome_a, outcome_b)

    f = result.get("faithfulness", {})
    f1 = float(f.get("f1", 0.0))
    circuit = result.get("circuit", [])

    print("\n" + "=" * 66)
    print("GLASSBOX — high-risk decision audit (worked example, gpt2 demo)")
    print("=" * 66)
    print(f"Decision contrast : '{outcome_a.strip()}'  vs  '{outcome_b.strip()}'")
    print(f"Circuit size      : {len(circuit)} attention head(s)")
    print(f"Top circuit heads : {circuit[:8]}")
    print(f"Sufficiency       : {f.get('sufficiency')}")
    print(f"Comprehensiveness : {f.get('comprehensiveness')}")
    print(f"F1 (faithfulness) : {round(f1, 3)}")
    print(f"Grade             : {_grade(f1)}")
    print(f"Category          : {f.get('category', 'n/a')}")

    # 4. EMIT THE ACTUAL EU AI ACT ANNEX IV REPORT (JSON + HTML).
    os.makedirs("reports", exist_ok=True)
    vault = build_annex_iv_vault(
        gb_result=result,
        model_name="gpt2 (DEMO — replace with your underwriting model)",
        provider="Example Bank AG",
        use_case="Consumer credit / loan approval (EU AI Act Annex III, high-risk)",
        deployment_ctx="credit_scoring",
        output_json="reports/loan-annex-iv.json",
        output_html="reports/loan-annex-iv.html",
    )
    summary = vault.to_dict().get("compliance_summary", "(see report)")

    print("-" * 66)
    print("Annex IV report written:")
    print("  reports/loan-annex-iv.json   (machine-readable)")
    print("  reports/loan-annex-iv.html   (open in a browser)")
    print(f"Compliance summary : {summary}")
    print("=" * 66)
    print(
        "\nReminder: gpt2 can't underwrite — the numbers above are pipeline proof,\n"
        "not a credit judgment. Point line 1 at a real model for a real audit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
