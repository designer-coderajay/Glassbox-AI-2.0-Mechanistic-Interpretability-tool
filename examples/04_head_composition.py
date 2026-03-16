"""
Example 4 — Head Composition Scores (Elhage et al. 2021)
=========================================================

Quantifies how much attention heads COMPOSE with each other via
Q/K/V virtual weight composition (Elhage et al. 2021 §3.2).

A large Q-composition score for (L1H1 → L2H2) means L2H2 attends
more to tokens that L1H1 wrote to the residual stream.

Run:
    python examples/04_head_composition.py
"""

from transformer_lens import HookedTransformer
from glassbox import GlassboxV2, HeadCompositionAnalyzer

model = HookedTransformer.from_pretrained("gpt2")
gb    = GlassboxV2(model)
comp  = HeadCompositionAnalyzer(model)

# First find the IOI circuit heads
result  = gb.analyze(
    "When Mary and John went to the store, John gave a drink to",
    " Mary", " John"
)
circuit = result["circuit"][:12]   # top-12 heads

print(f"\nTop circuit heads: {circuit}")

# Full Q/K/V composition for the circuit
all_scores = comp.all_composition_scores(circuit)
print(f"\nTop 10 composition edges (any kind):")
print(f"{'Sender':<14} {'Receiver':<14} {'Kind':<4} {'Score':>8}")
print("-" * 44)
for edge in all_scores["combined_edges"][:10]:
    sl, sh = edge["sender"]
    rl, rh = edge["receiver"]
    print(f"  L{sl}H{sh:<10} L{rl}H{rh:<10} {edge['kind']:<4} {edge['score']:>8.4f}")

# Single pairwise example — classic S-inhibition → name-mover
q_score = comp.q_composition_score(
    sender_layer=3, sender_head=0,
    recv_layer=9,   recv_head=9,
)
print(f"\nQ-composition L3H0 → L9H9 (name-mover ← dup-token): {q_score:.4f}")

# Composition matrix for Q-kind across all layers 0..5 → 6..11
matrix = comp.composition_matrix(
    senders   = [(l, h) for l in range(6)  for h in range(12)],
    receivers = [(l, h) for l in range(6, 12) for h in range(12)],
    kind      = "q",
)
print(f"\nQ-composition matrix shape: {matrix['matrix'].shape}")
print(f"Max score: {matrix['matrix'].max():.4f}  at sender={matrix['sender_labels'][matrix['matrix'].argmax() // matrix['matrix'].shape[1]]} → receiver={matrix['receiver_labels'][matrix['matrix'].argmax() % matrix['matrix'].shape[1]]}")
