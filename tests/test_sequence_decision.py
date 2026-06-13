"""Tests for glassbox/sequence_decision.py — teacher-forced multi-token decision value.

Pure logic, verified against a constant-logits stub so every number is
hand-computable. The model glue (model_scorer) is torch-only and validated on a
real model, not here.
"""
import math

import pytest

from glassbox.decision import DecisionFunctional, VerbalizerSet
from glassbox.sequence_decision import sequence_decision_value, teacher_forced_logprob

# Constant vocab logits (size 6); same row at every position.
# logsumexp(V) = 2 + log(4*e^-2 + e^-1 + 1) = 2.646654
_V = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
_LP5 = 2.0 - 2.646654   # -0.646654
_LP4 = 1.0 - 2.646654   # -1.646654
_LP0 = 0.0 - 2.646654   # -2.646654


def _const_logits(token_ids):
    return [list(_V) for _ in token_ids]


def test_single_token_logprob():
    assert teacher_forced_logprob([1, 2], [5], _const_logits) == pytest.approx(_LP5, abs=1e-4)


def test_multi_token_logprob_sums_over_span():
    assert teacher_forced_logprob([1, 2], [5, 4], _const_logits) == pytest.approx(_LP5 + _LP4, abs=1e-4)


def test_empty_inputs_guarded():
    with pytest.raises(ValueError):
        teacher_forced_logprob([], [5], _const_logits)
    with pytest.raises(ValueError):
        teacher_forced_logprob([1], [], _const_logits)


def test_sequence_decision_value_singletons():
    fn = DecisionFunctional(VerbalizerSet("yes", (" yes",)), VerbalizerSet("no", (" no",)))
    enc = {" yes": [5], " no": [0]}
    D = sequence_decision_value(fn, lambda s: enc[s], [1, 2], _const_logits)
    # single-element logsumexp == value: D = LP5 - LP0 = 2.0
    assert D == pytest.approx(_LP5 - _LP0, abs=1e-4)
    assert D == pytest.approx(2.0, abs=1e-4)


def test_sequence_decision_value_pools_multi_variant():
    fn = DecisionFunctional(
        VerbalizerSet("yes", (" yes", " yeah")), VerbalizerSet("no", (" no",))
    )
    enc = {" yes": [5], " yeah": [4], " no": [0]}
    D = sequence_decision_value(fn, lambda s: enc[s], [1, 2], _const_logits)
    lse_pos = math.log(math.exp(_LP5) + math.exp(_LP4))
    assert D == pytest.approx(lse_pos - _LP0, abs=1e-4)


def test_sequence_decision_value_multi_token_variant():
    # a variant that is itself multiple tokens
    fn = DecisionFunctional(
        VerbalizerSet("approve", (" Approved",)), VerbalizerSet("deny", (" Denied",))
    )
    enc = {" Approved": [5, 4], " Denied": [0]}
    D = sequence_decision_value(fn, lambda s: enc[s], [1, 2], _const_logits)
    # positive variant spans two tokens: LP5 + LP4
    assert D == pytest.approx((_LP5 + _LP4) - _LP0, abs=1e-4)
