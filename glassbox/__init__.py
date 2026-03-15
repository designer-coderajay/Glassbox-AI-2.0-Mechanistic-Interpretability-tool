"""
Glassbox 2.0 — Causal Mechanistic Interpretability Engine
==========================================================

Quick start
-----------
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2

    model = HookedTransformer.from_pretrained("gpt2")
    gb    = GlassboxV2(model)

    result = gb.analyze(
        prompt    = "When Mary and John went to the store, John gave a drink to",
        correct   = " Mary",
        incorrect = " John",
    )
    print(result["faithfulness"])
    # {'sufficiency': 1.0, 'comprehensiveness': 0.47, 'f1': 0.64,
    #  'category': 'backup_mechanisms', 'suff_is_approx': True}

Package layout
--------------
glassbox/
  __init__.py       ← you are here — re-exports the public API
  core.py           ← GlassboxV2 class (attribution patching, MFC, FCAS, bootstrap)
  cli.py            ← glassbox-ai CLI entry point
  alignment.py      ← DEPRECATED: thin shim kept for back-compat
"""

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "2.0.1"
__author__  = "Ajay Pravin Mahale"
__email__   = "mahale.ajay01@gmail.com"

# ---------------------------------------------------------------------------
# Public API — everything a user needs is importable from `glassbox` directly
# ---------------------------------------------------------------------------
from glassbox.core import GlassboxV2          # primary class

# Back-compat alias for any code that used the old name
GlassboxEngine = GlassboxV2

__all__ = [
    "GlassboxV2",
    "GlassboxEngine",    # deprecated alias — use GlassboxV2
    "__version__",
]
