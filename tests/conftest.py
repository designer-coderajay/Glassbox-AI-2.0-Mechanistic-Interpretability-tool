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

IMPORTANT: stubs are ONLY injected when the top-level package is genuinely
not installed.  If torch / transformer_lens are installed (e.g. in CI after
``pip install -e ".[dev]"``), we leave sys.modules alone so that real imports
work correctly in test_engine.py.
─────────────────────────────────────────────────────────────────────────────
"""

import importlib.util
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Inject stubs for heavy ML dependencies that aren't installed in the
# offline test environment.  This MUST happen at conftest import time
# (i.e. module level, not inside a fixture) so the stubs are present before
# pytest imports any test module.
#
# Each entry is (stub_module_path, top_level_package).  A stub is only
# injected when the top-level package cannot be found by importlib — i.e.
# it is genuinely NOT installed.  This prevents the stub from shadowing the
# real package when running in an environment that has torch/transformer_lens.
# ---------------------------------------------------------------------------

_STUB_MODULES = [
    # torch and every submodule imported anywhere in glassbox/
    ("torch",                       "torch"),
    ("torch.nn",                    "torch"),
    ("torch.nn.functional",         "torch"),
    ("torch.autograd",              "torch"),
    ("torch.autograd.functional",   "torch"),
    ("torch.cuda",                  "torch"),
    ("torch.utils",                 "torch"),
    ("torch.utils.data",            "torch"),
    ("torch.linalg",                "torch"),
    # transformer_lens
    ("transformer_lens",            "transformer_lens"),
    ("transformer_lens.hook_points","transformer_lens"),
    ("transformer_lens.utilities",  "transformer_lens"),
    # other heavy deps
    ("einops",                      "einops"),
    ("scipy",                       "scipy"),
    ("scipy.stats",                 "scipy"),
    ("scipy.spatial",               "scipy"),
    ("scipy.spatial.distance",      "scipy"),
    # sae_lens (optional dep)
    ("sae_lens",                    "sae_lens"),
]

# Track which top-level packages are genuinely absent (cache the spec lookup).
_absent: dict[str, bool] = {}

def _is_absent(top_level: str) -> bool:
    if top_level not in _absent:
        _absent[top_level] = importlib.util.find_spec(top_level) is None
    return _absent[top_level]

for _mod, _top in _STUB_MODULES:
    if _mod not in sys.modules and _is_absent(_top):
        sys.modules[_mod] = MagicMock()
