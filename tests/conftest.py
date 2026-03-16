"""
conftest.py — shared pytest fixtures and configuration for Glassbox tests.

Fixtures are session-scoped to avoid reloading GPT-2 (~117M params) for
every test class. The model is loaded once and reused across all tests.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Session-scoped model + engine — loaded ONCE for the entire test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hf_model():
    """HookedTransformer GPT-2 small, loaded once per pytest session."""
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def gb(hf_model):
    """GlassboxV2 engine wrapping GPT-2 small."""
    from glassbox import GlassboxV2
    return GlassboxV2(hf_model)


@pytest.fixture(scope="session")
def composition_analyzer(hf_model):
    """HeadCompositionAnalyzer for GPT-2 small."""
    from glassbox import HeadCompositionAnalyzer
    return HeadCompositionAnalyzer(hf_model)


# ---------------------------------------------------------------------------
# Canonical test inputs
# ---------------------------------------------------------------------------

IOI_PROMPT    = "When Mary and John went to the store, John gave a drink to"
IOI_CORRECT   = " Mary"
IOI_INCORRECT = " John"


@pytest.fixture(scope="session")
def ioi_result(gb):
    """Full analyze() result for the canonical IOI prompt."""
    return gb.analyze(IOI_PROMPT, IOI_CORRECT, IOI_INCORRECT)


@pytest.fixture(scope="session")
def ioi_tokens(hf_model):
    """Tokenized canonical IOI prompt."""
    return hf_model.to_tokens(IOI_PROMPT)


@pytest.fixture(scope="session")
def ioi_target_id(hf_model):
    """Token ID for ' Mary'."""
    return hf_model.to_single_token(IOI_CORRECT)


@pytest.fixture(scope="session")
def ioi_distractor_id(hf_model):
    """Token ID for ' John'."""
    return hf_model.to_single_token(IOI_INCORRECT)
