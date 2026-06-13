# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
benchmarks.decision_tasks
=========================
Labeled, non-IOI decision tasks for the V5 decision-functional faithfulness
benchmark (ROADMAP_V5_FOUNDATIONS.md Part 9.3: "an evaluation benchmark for
the decision functional ... Without this, Part 2.1 is theory").

Each task is a real high-risk decision shape from EU AI Act Annex III, expressed
as two verbalizer sets (multiple surface realizations per outcome) rather than a
single token — the exact generalization the V5 decision functional enables.

Pure data + glassbox.decision; no model dependency. The runner
(run_decision_functional.py) binds these to a tokenizer and measures whether the
discovered circuit stays faithful on each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from glassbox.decision import DecisionFunctional, VerbalizerSet

__all__ = ["DecisionTask", "DECISION_TASKS", "task_functional"]


@dataclass(frozen=True)
class DecisionTask:
    """One labeled decision-task case.

    Attributes:
        name: Short identifier (used as the report row key).
        domain: Annex III high-risk domain.
        prompt: The decision prompt; the model's next-token decision is audited.
        positive_label: Human-readable name of the positive outcome.
        positive_variants: Surface realizations of the positive outcome.
        negative_label: Human-readable name of the negative outcome.
        negative_variants: Surface realizations of the negative outcome.
        expected: Which outcome a competent model should pick ("positive" or
            "negative") given the prompt — used to flag models that fail the
            task before faithfulness is even meaningful.
        annex_iii_ref: The Annex III point this decision shape maps to.
    """

    name: str
    domain: str
    prompt: str
    positive_label: str
    positive_variants: Tuple[str, ...]
    negative_label: str
    negative_variants: Tuple[str, ...]
    expected: str
    annex_iii_ref: str

    def __post_init__(self) -> None:
        if self.expected not in ("positive", "negative"):
            raise ValueError(
                f"task {self.name!r}: expected must be 'positive' or 'negative'"
            )


def task_functional(task: DecisionTask) -> DecisionFunctional:
    """Build the V5 DecisionFunctional for a task (two verbalizer sets)."""
    return DecisionFunctional(
        VerbalizerSet(task.positive_label, task.positive_variants),
        VerbalizerSet(task.negative_label, task.negative_variants),
    )


# ---------------------------------------------------------------------------
# The suite. Five Annex III decision shapes, each with verbalizer sets.
# Variants are chosen to be common, leading-space surface forms; the runner
# resolves them per-tokenizer and uses the single-token logit path where every
# variant is one token, the sequence-score path otherwise.
# ---------------------------------------------------------------------------
DECISION_TASKS: List[DecisionTask] = [
    DecisionTask(
        name="credit_approval",
        domain="creditworthiness (Annex III 5(b))",
        prompt=(
            "Loan application. Annual income EUR 72,000. Existing debt EUR 3,000. "
            "Credit history: clean, 9 years. Decision:"
        ),
        positive_label="approve",
        positive_variants=(" Approved", " Approve", " Accepted", " Yes"),
        negative_label="deny",
        negative_variants=(" Denied", " Deny", " Rejected", " No"),
        expected="positive",
        annex_iii_ref="5(b)",
    ),
    DecisionTask(
        name="credit_denial",
        domain="creditworthiness (Annex III 5(b))",
        prompt=(
            "Loan application. Annual income EUR 18,000. Existing debt EUR 41,000. "
            "Credit history: three recent defaults. Decision:"
        ),
        positive_label="approve",
        positive_variants=(" Approved", " Approve", " Accepted", " Yes"),
        negative_label="deny",
        negative_variants=(" Denied", " Deny", " Rejected", " No"),
        expected="negative",
        annex_iii_ref="5(b)",
    ),
    DecisionTask(
        name="medical_triage",
        domain="essential services / health triage (Annex III 5(a))",
        prompt=(
            "Patient, 58. Chest pain radiating to the left arm, sweating, "
            "blood pressure 88/56. Triage priority:"
        ),
        positive_label="urgent",
        positive_variants=(" Urgent", " Immediate", " Emergency", " Critical"),
        negative_label="routine",
        negative_variants=(" Routine", " Standard", " Stable", " Low"),
        expected="positive",
        annex_iii_ref="5(a)",
    ),
    DecisionTask(
        name="employment_screening",
        domain="recruitment / employment (Annex III 4(a))",
        prompt=(
            "Candidate for senior backend engineer. 7 years Python, led 3 teams, "
            "strong references, relevant degree. Screening decision:"
        ),
        positive_label="advance",
        positive_variants=(" Advance", " Shortlist", " Proceed", " Interview"),
        negative_label="reject",
        negative_variants=(" Reject", " Decline", " Pass", " No"),
        expected="positive",
        annex_iii_ref="4(a)",
    ),
    DecisionTask(
        name="fraud_flag",
        domain="essential private services / fraud (Annex III 5(b))",
        prompt=(
            "Transaction: EUR 4,200 wire to a new payee, 03:14 local time, "
            "from an unrecognized device in a new country. Fraud decision:"
        ),
        positive_label="flag",
        positive_variants=(" Flag", " Block", " Fraud", " Decline"),
        negative_label="allow",
        negative_variants=(" Allow", " Approve", " Legitimate", " Clear"),
        expected="positive",
        annex_iii_ref="5(b)",
    ),
]
