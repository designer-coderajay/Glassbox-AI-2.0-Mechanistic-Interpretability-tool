"""
Example 5 — Attention Pattern Analysis
=======================================

Extracts and classifies attention patterns for key circuit heads.
Computes Shannon entropy and classifies head types:
  induction_candidate, previous_token, focused, uniform, self_attn, mixed.

Run:
    python examples/05_attention_patterns.py
"""

import numpy as np
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

model = HookedTransformer.from_pretrained("gpt2")
gb    = GlassboxV2(model)

prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)

# Auto-select top-10 heads by entropy variance (most interesting patterns)
result = gb.attention_patterns(tokens, heads=None, top_k=10)

print(f"\nAttention Patterns — '{prompt[:50]}...'")
print(f"Sequence length: {result['seq_len']} tokens\n")

print(f"{'Head':<10} {'Type':<22} {'Entropy':>8} {'Max Attn':>10}")
print("-" * 54)
for hd in result["heads"]:
    l, h = hd["layer"], hd["head"]
    print(f"  L{l}H{h:<6} {hd['head_type']:<22} {hd['entropy']:>8.3f}  {hd['max_attention']:>10.4f}")

# Print the actual attention pattern for L9H9 (key IOI name-mover head)
patterns = gb.attention_patterns(tokens, heads=[(9, 9)])
pat_99   = np.array(patterns["heads"][0]["pattern"])   # [seq, seq]
tok_strs = model.to_str_tokens(tokens[0])

print(f"\nAttention pattern for L9H9 (name-mover head):")
print(f"  Rows = query positions, Cols = key positions")
print(f"  {'':20}", end="")
for ts in tok_strs:
    print(f"{repr(ts):>8}", end="")
print()
for i, (row, ts) in enumerate(zip(pat_99, tok_strs)):
    print(f"  {repr(ts):>20}", end="")
    for v in row:
        cell = "████"[:int(v * 4)] if v > 0.01 else "    "
        print(f"  {v:>6.3f}", end="")
    print()
