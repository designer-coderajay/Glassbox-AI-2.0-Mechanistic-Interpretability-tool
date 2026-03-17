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
"""
