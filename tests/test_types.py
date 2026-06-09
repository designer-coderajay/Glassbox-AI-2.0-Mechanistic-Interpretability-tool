"""Tests for glassbox/types.py — public type aliases and validated constants."""
from glassbox.types import (
    ATTRIBUTION_METHODS,
    FAITHFULNESS_CATEGORIES,
    VALID_HEAD_TYPES,
    __all__,
)
import glassbox.types as gbt


def test_all_exports_resolve():
    # Every name promised in __all__ must actually exist on the module.
    for name in __all__:
        assert hasattr(gbt, name), f"{name} listed in __all__ but missing"


def test_valid_head_types_membership():
    expected = {
        "induction_candidate", "previous_token", "focused",
        "uniform", "self_attn", "mixed",
    }
    assert VALID_HEAD_TYPES == expected
    assert "focused" in VALID_HEAD_TYPES
    assert isinstance(VALID_HEAD_TYPES, set)


def test_faithfulness_categories():
    assert FAITHFULNESS_CATEGORIES == {
        "faithful", "backup_mechanisms", "incomplete", "weak", "moderate",
    }
    assert "faithful" in FAITHFULNESS_CATEGORIES


def test_attribution_methods():
    assert ATTRIBUTION_METHODS == {"taylor", "integrated_gradients"}
    assert "taylor" in ATTRIBUTION_METHODS
    assert "nonexistent_method" not in ATTRIBUTION_METHODS


def test_type_aliases_are_importable():
    # The aliases are documentation wrappers; just confirm they resolve.
    from glassbox.types import (
        AnalyzeResult, AttributionDict, CircuitList,
        FaithfulnessResult, HeadTuple, PromptTuple,
    )
    for alias in (HeadTuple, CircuitList, AttributionDict, PromptTuple,
                  AnalyzeResult, FaithfulnessResult):
        assert alias is not None
