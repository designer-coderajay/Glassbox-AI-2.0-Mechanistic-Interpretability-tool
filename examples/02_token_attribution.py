"""
Example 2 — Token Attribution (Gradient × Embedding Saliency)
==============================================================

Shows which INPUT tokens drive the model's prediction.
Method: gradient × embedding (Simonyan et al. 2014).

Run:
    python examples/02_token_attribution.py
"""

from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

model = HookedTransformer.from_pretrained("gpt2")
gb    = GlassboxV2(model)

prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)

result = gb.token_attribution(
    tokens,
    target_token    = model.to_single_token(" Mary"),
    distractor_token= model.to_single_token(" John"),
)

print(f"\nToken attribution for: '{prompt}'")
print(f"  Target: ' Mary'  |  Distractor: ' John'\n")
print(f"{'Rank':<5} {'Token':<20} {'Attribution':>12} {'|Attribution|':>15}")
print("-" * 56)
for tok in result["top_tokens"]:
    bar = "█" * int(abs(tok["attribution"]) / max(result["abs_attributions"]) * 20)
    sign = "+" if tok["attribution"] > 0 else "-"
    print(f"  {tok['rank']:<3} {repr(tok['token_str']):<20} {tok['attribution']:>+12.4f}  {bar}")

print(f"\nFull attribution vector (all {result['n_tokens']} tokens):")
for i, (ts, a) in enumerate(zip(result["token_strs"], result["attributions"])):
    print(f"  [{i:2d}] {repr(ts):<20} {a:+.4f}")
