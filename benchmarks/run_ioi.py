#!/usr/bin/env python
"""Benchmark: IOI (Indirect Object Identification)"""
import argparse, numpy as np
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

IOI_PROMPTS = [
    ("When Mary and John went to the store, John gave a bottle to", "Mary", "John"),
    ("When Sarah and Tom went to the park, Tom gave the ball to", "Sarah", "Tom"),
    ("When Alice and Bob went to the office, Bob gave the report to", "Alice", "Bob"),
    ("When Emma and James went to the cafe, James gave the cup to", "Emma", "James"),
    ("When Lisa and Mark went to the gym, Mark gave the towel to", "Lisa", "Mark"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()
    model = HookedTransformer.from_pretrained(args.model)
    gb = GlassboxV2(model)
    results = []
    for prompt, target, distractor in IOI_PROMPTS:
        r = gb.analyze([prompt], [prompt.replace(target, "X")], target, distractor)
        f = r["faithfulness"]
        print(f"  suff={f['sufficiency']:.3f}  comp={f['comprehensiveness']:.3f}  f1={f['f1']:.3f}")
        results.append(f)
    print(f"\nIOI Mean — suff={np.mean([r['sufficiency'] for r in results]):.3f}  comp={np.mean([r['comprehensiveness'] for r in results]):.3f}")

if __name__ == "__main__":
    main()
