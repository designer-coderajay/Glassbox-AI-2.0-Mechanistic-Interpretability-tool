#!/usr/bin/env python
"""Benchmark: SVA (Subject-Verb Agreement)"""
import argparse, numpy as np
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

SVA_PROMPTS = [
    ("The keys to the cabinet", "are", "is"),
    ("The manager of the stores", "was", "were"),
    ("The dog near the trees", "barks", "bark"),
    ("The student in the classes", "studies", "study"),
    ("The book on the shelves", "belongs", "belong"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()
    model = HookedTransformer.from_pretrained(args.model)
    gb = GlassboxV2(model)
    results = []
    for prompt, target, distractor in SVA_PROMPTS:
        r = gb.analyze([prompt], [" ".join(prompt.split()[1:])], target, distractor)
        f = r["faithfulness"]
        print(f"  suff={f['sufficiency']:.3f}  comp={f['comprehensiveness']:.3f}  f1={f['f1']:.3f}")
        results.append(f)
    print(f"\nSVA Mean — suff={np.mean([r['sufficiency'] for r in results]):.3f}  comp={np.mean([r['comprehensiveness'] for r in results]):.3f}")

if __name__ == "__main__":
    main()
