"""
Example 1 — Indirect Object Identification (IOI) Circuit Discovery
===================================================================

Reproduces the IOI experiment from Wang et al. (2022):
"Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"
https://arxiv.org/abs/2211.00593

Run:
    python examples/01_ioi_circuit.py

Expected output (approx):
    Top-5 circuit heads: [(9,9), (9,6), (10,0), (11,10), (8,10)]
    Faithfulness → sufficiency=0.80  comprehensiveness=0.37  f1=0.49
    Category: backup_mechanisms
"""

from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

# ── load model ────────────────────────────────────────────────────────────────
model = HookedTransformer.from_pretrained("gpt2")   # GPT-2 small, ~117M params
gb    = GlassboxV2(model)

# ── run analysis ──────────────────────────────────────────────────────────────
result = gb.analyze(
    prompt    = "When Mary and John went to the store, John gave a drink to",
    correct   = " Mary",
    incorrect = " John",
)

# ── inspect circuit ───────────────────────────────────────────────────────────
circuit = result["circuit"]   # list of (layer, head) tuples, sorted by attribution
print(f"\nTop-10 circuit heads: {circuit[:10]}")
print(f"\nAttributions (top-5):")
for head, score in list(result["attributions"].items())[:5]:
    print(f"  L{head[0]}H{head[1]}: {score:.4f}")

# ── faithfulness ──────────────────────────────────────────────────────────────
faith = result["faithfulness"]
print(f"\nFaithfulness:")
print(f"  Sufficiency      = {faith['sufficiency']:.1%}")
print(f"  Comprehensiveness= {faith['comprehensiveness']:.1%}")
print(f"  F1               = {faith['f1']:.1%}")
print(f"  Category         = {faith['category']}")

# ── bootstrap confidence intervals ────────────────────────────────────────────
from glassbox import GlassboxV2   # already imported
bs = gb.bootstrap_faithfulness_metrics(
    prompt    = "When Mary and John went to the store, John gave a drink to",
    correct   = " Mary",
    incorrect = " John",
    n_boot    = 50,
    alpha     = 0.05,
)
print(f"\nBootstrap 95% CI (n=50):")
print(f"  Sufficiency CI       = [{bs['sufficiency_ci'][0]:.3f}, {bs['sufficiency_ci'][1]:.3f}]")
print(f"  Comprehensiveness CI = [{bs['comprehensiveness_ci'][0]:.3f}, {bs['comprehensiveness_ci'][1]:.3f}]")
