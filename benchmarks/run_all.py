#!/usr/bin/env python
"""Run all Glassbox 2.0 benchmarks."""
import argparse
import numpy as np
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

SUITES = {
    "IOI": [
        ("When Mary and John went to the store, John gave a bottle to", "Mary", "John"),
        ("When Sarah and Tom went to the park, Tom gave the ball to",   "Sarah", "Tom"),
        ("When Alice and Bob went to the office, Bob gave the report to", "Alice", "Bob"),
    ],
    "SVA": [
        ("The keys to the cabinet",    "are",   "is"),
        ("The manager of the stores",  "was",   "were"),
        ("The dog near the trees",     "barks", "bark"),
    ],
    "GEO": [
        ("The capital of France is",  "Paris",  "London"),
        ("The capital of Germany is", "Berlin", "Vienna"),
        ("The capital of Japan is",   "Tokyo",  "Seoul"),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()

    model = HookedTransformer.from_pretrained(args.model)
    gb    = GlassboxV2(model)

    print(f"\n{'Task':<8} {'Sufficiency':>12} {'Comprehensiveness':>18} {'F1':>6}")
    print("-" * 48)

    for name, prompts in SUITES.items():
        results = []
        for prompt, target, dist in prompts:
            # BUG FIX: original code called gb.analyze([prompt], [corrupt], target, dist)
            # — 4 args, prompt wrapped in a list, manual corruption passed in.
            # analyze() signature is analyze(prompt: str, correct: str, incorrect: str).
            # It takes 3 plain strings and handles corruption internally via _name_swap.
            # Also: the original used variable name 'distractor' which was never defined
            # in this scope (loop var is 'dist') -> NameError on every benchmark run.
            r = gb.analyze(prompt, target, dist)
            results.append(r["faithfulness"])

        suff = np.mean([r["sufficiency"]       for r in results])
        comp = np.mean([r["comprehensiveness"] for r in results])
        f1   = np.mean([r["f1"]                for r in results])
        print(f"{name:<8} {suff:>12.1%} {comp:>18.1%} {f1:>6.1%}")


if __name__ == "__main__":
    main()
