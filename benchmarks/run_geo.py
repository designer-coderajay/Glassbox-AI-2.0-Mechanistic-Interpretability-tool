#!/usr/bin/env python
"""Benchmark: GEO (Country-Capital factual recall)"""
import argparse, numpy as np
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

GEO_PROMPTS = [
    ("The capital of France is", "Paris", "London"),
    ("The capital of Germany is", "Berlin", "Vienna"),
    ("The capital of Japan is", "Tokyo", "Seoul"),
    ("The capital of Italy is", "Rome", "Madrid"),
    ("The capital of Australia is", "Canberra", "Sydney"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()
    model = HookedTransformer.from_pretrained(args.model)
    gb = GlassboxV2(model)
    results = []
    for prompt, target, distractor in GEO_PROMPTS:
        r = gb.analyze([prompt], ["The capital of X is"], target, distractor)
        f = r["faithfulness"]
        print(f"  suff={f['sufficiency']:.3f}  comp={f['comprehensiveness']:.3f}  f1={f['f1']:.3f}")
        results.append(f)
    print(f"\nGEO Mean — suff={np.mean([r['sufficiency'] for r in results]):.3f}  comp={np.mean([r['comprehensiveness'] for r in results]):.3f}")

if __name__ == "__main__":
    main()
