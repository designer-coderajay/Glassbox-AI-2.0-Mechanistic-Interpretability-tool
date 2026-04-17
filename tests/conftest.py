"""
conftest.py — shared pytest fixtures for the Glassbox test suite.

IMPORTANT architecture note
─────────────────────────────────────────────────────────────────────────────
test_engine.py is self-contained: it defines its own module-scoped ``engine``
fixture that loads GPT-2 once per module, plus all dependent fixtures
(``ioi_tokens``, ``ioi_result``, ``eap_result``, etc.) as module-scoped
fixtures.

This conftest intentionally does NOT redefine any fixture that test_engine.py
already owns.  Duplicating fixture names at session scope here would cause
pytest to pick whichever scope it resolves first, depending on the pytest
version — leading to hard-to-diagnose "wrong fixture" failures in CI.

If you add a second test module that needs the shared GPT-2 engine, add a
``gb`` session fixture here at that time and wire it up carefully.
─────────────────────────────────────────────────────────────────────────────

Offline-test shim
─────────────────────────────────────────────────────────────────────────────
Tests for the no-model modules (compliance, audit_log, risk_register, widget)
do not need PyTorch or TransformerLens.  We inject lightweight sys.modules
stubs HERE — before pytest collects any test file — so that
``glassbox/__init__.py`` can be imported without torch being installed.

The stubs are MagicMocks, which means any attribute access or call on the
fake modules silently returns another MagicMock.  Tests that actually need a
real model (test_engine.py) skip themselves gracefully when the real
transformer_lens is unavailable.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Inject stubs for heavy ML dependencies that aren't installed in the
# offline test environment.  This MUST happen at conftest import time
# (i.e. module level, not inside a fixture) so the stubs are present before
# pytest imports any test module.
# ---------------------------------------------------------------------------

_STUB_MODULES = [
    # torch and every submodule imported anywhere in glassbox/
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.autograd",
    "torch.autograd.functional",
    "torch.cuda",
    "torch.utils",
    "torch.utils.data",
    "torch.linalg",
    # transformer_lens
    "transformer_lens",
    "transformer_lens.hook_points",
    "transformer_lens.utilities",
    # other heavy deps
    "einops",
    "scipy",
    "scipy.stats",
    "scipy.spatial",
    "scipy.spatial.distance",
    # sae_lens (optional dep)
    "sae_lens",
]

for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
