"""
Example 3 — Logit Lens
======================

Projects the residual stream at each layer through the final
LayerNorm + unembedding to show how predictions form layer by layer.

Reference: nostalgebraist (2020) "Interpreting GPT: the Logit Lens"
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

Run:
    python examples/03_logit_lens.py
"""

from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

model = HookedTransformer.from_pretrained("gpt2")
gb    = GlassboxV2(model)

prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)

lens = gb.logit_lens(tokens, " Mary", " John")

print(f"\nLogit Lens — '{prompt}'")
print(f"  Target: ' Mary'  |  Distractor: ' John'\n")
print(f"{'Layer':<8} {'Logit Diff':>12} {'Shift':>10}")
print("-" * 34)
for layer_idx, (ld, shift) in enumerate(
    zip(lens["logit_diffs"], lens["logit_shifts"])
):
    bar = "+" * max(0, int(ld * 5)) if ld > 0 else "-" * max(0, int(-ld * 5))
    print(f"  L{layer_idx:<5} {ld:>+12.3f} {shift:>+10.3f}  {bar}")

print(f"\nFinal logit diff: {lens['logit_diffs'][-1]:+.3f}")
print(f"Top head direct effects (layer, head → effect):")
for (l, h), eff in sorted(
    lens["head_direct_effects"].items(), key=lambda x: abs(x[1]), reverse=True
)[:10]:
    print(f"  L{l}H{h}: {eff:+.4f}")
