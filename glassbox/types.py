"""
glassbox/types.py — Public type aliases and constants.

All public type aliases are re-exported from glassbox.__init__ via __all__
so downstream code can write:

    from glassbox.types import HeadTuple, CircuitList

These are thin wrappers around Python builtins and exist purely to document
intent — Glassbox's function signatures become self-describing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union

__all__ = [
    # Head / circuit types
    "HeadTuple",
    "CircuitList",
    "AttributionDict",
    "PromptTuple",
    # Result dict types
    "AnalyzeResult",
    "FaithfulnessResult",
    # Constants
    "VALID_HEAD_TYPES",
    "FAITHFULNESS_CATEGORIES",
    "ATTRIBUTION_METHODS",
]

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

#: A single attention head identified by (layer_index, head_index).
HeadTuple = Tuple[int, int]

#: Ordered list of heads forming a circuit, highest-attribution first.
CircuitList = List[HeadTuple]

#: Attribution scores keyed by HeadTuple.
AttributionDict = Dict[HeadTuple, float]

#: A single analysis input: (prompt, correct_token, incorrect_token).
PromptTuple = Tuple[str, str, str]

# ---------------------------------------------------------------------------
# Rich result types (used for documentation; results are plain dicts at runtime)
# ---------------------------------------------------------------------------

#: Full output of GlassboxV2.analyze().
AnalyzeResult = Dict[str, object]

#: The 'faithfulness' sub-dict inside AnalyzeResult.
FaithfulnessResult = Dict[str, Union[float, str, bool]]

# ---------------------------------------------------------------------------
# Validated constants — tests and type-checkers import these to stay in sync
# with the production code.
# ---------------------------------------------------------------------------

#: Set of head-type strings produced by attention_patterns().
VALID_HEAD_TYPES: Set[str] = {
    "induction_candidate",
    "previous_token",
    "focused",
    "uniform",
    "self_attn",
    "mixed",
}

#: Faithfulness category strings produced by analyze().
FAITHFULNESS_CATEGORIES: Set[str] = {
    "faithful",
    "backup_mechanisms",
    "incomplete",
    "weak",
    "moderate",
}

#: Attribution methods accepted by analyze() and minimum_faithful_circuit().
ATTRIBUTION_METHODS: Set[str] = {
    "taylor",
    "integrated_gradients",
}
