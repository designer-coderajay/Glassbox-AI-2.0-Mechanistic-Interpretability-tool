#!/usr/bin/env python3
"""
Generate the self-owned synthetic credit dataset (Phase 1.2 of PLAN_DECISION_AUDIT.md).

We own this data outright — no third-party rights, no PII, no license to satisfy.
Reproducible from SEED. Labels follow credit_rule.decide() exactly (ground truth).

Outputs (into ./data/):
  train.csv / test.csv      — tabular features + label
  train.jsonl / test.jsonl  — {"prompt","completion"} for LLM fine-tuning + Glassbox
  stats.json                — class balance + per-feature summary (reported as measured)
"""
from __future__ import annotations
import csv
import json
import os

import numpy as np

from credit_rule import decide, ALL_FEATURES

SEED = 42
N_TRAIN = 2000
N_TEST = 500
OUT = os.path.join(os.path.dirname(__file__), "data")


def _sample_pool(n: int, rng: np.random.Generator) -> list[dict]:
    """Realistic, mostly-independent features; labels = the rule (deterministic)."""
    rows = []
    for _ in range(n):
        credit_score = int(np.clip(rng.normal(680, 80), 300, 850))
        dti = float(np.clip(rng.normal(0.35, 0.15), 0.0, 0.80))
        num_defaults = int(rng.choice([0, 1, 2, 3], p=[0.72, 0.16, 0.08, 0.04]))
        annual_income = int(np.clip(rng.normal(70000, 30000), 15000, 250000))
        employment_years = float(np.clip(rng.exponential(7), 0, 40))
        age = int(rng.integers(18, 76))
        loan_amount = int(np.clip(rng.normal(18000, 12000), 1000, 80000))
        row = dict(
            credit_score=credit_score,
            dti=round(dti, 3),
            num_defaults=num_defaults,
            annual_income=annual_income,
            employment_years=round(employment_years, 1),
            age=age,
            loan_amount=loan_amount,
        )
        row["label"] = decide(**row)
        rows.append(row)
    return rows


def _balance(rows: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """Subsample to ~50/50 approved/denied so the model can't win by guessing."""
    approved = [r for r in rows if r["label"] == "approved"]
    denied = [r for r in rows if r["label"] == "denied"]
    k = min(len(approved), len(denied), n // 2)
    rng.shuffle(approved)
    rng.shuffle(denied)
    out = approved[:k] + denied[:k]
    rng.shuffle(out)
    return out


def _to_prompt(r: dict) -> str:
    return (
        f"Credit application — score: {r['credit_score']}, "
        f"debt-to-income: {r['dti']:.2f}, prior defaults: {r['num_defaults']}, "
        f"annual income: ${r['annual_income']}, age: {r['age']}, "
        f"employment: {r['employment_years']} yrs, loan: ${r['loan_amount']}. Decision:"
    )


def _write_split(rows: list[dict], name: str) -> None:
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ALL_FEATURES) + ["label"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps({"prompt": _to_prompt(r),
                                 "completion": " " + r["label"]}) + "\n")


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    approved = sum(1 for r in rows if r["label"] == "approved")
    feats = {}
    for f in ALL_FEATURES:
        vals = np.array([r[f] for r in rows], dtype=float)
        feats[f] = dict(min=float(vals.min()), max=float(vals.max()),
                        mean=round(float(vals.mean()), 2))
    return dict(n=n, approved=approved, denied=n - approved,
                approved_rate=round(approved / n, 3), features=feats)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # generate a large pool, then balance + split
    pool = _sample_pool(12000, rng)
    natural_rate = round(sum(r["label"] == "approved" for r in pool) / len(pool), 3)
    balanced = _balance(pool, N_TRAIN + N_TEST, rng)
    train, test = balanced[:N_TRAIN], balanced[N_TRAIN:N_TRAIN + N_TEST]

    _write_split(train, "train")
    _write_split(test, "test")

    stats = dict(seed=SEED, natural_approved_rate=natural_rate,
                 train=_stats(train), test=_stats(test))
    with open(os.path.join(OUT, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"natural approval rate (unbalanced pool): {natural_rate}")
    print(f"train: {stats['train']['n']} rows, approved_rate={stats['train']['approved_rate']}")
    print(f"test:  {stats['test']['n']} rows, approved_rate={stats['test']['approved_rate']}")
    print("example prompt:\n  ", _to_prompt(train[0]), "->", " " + train[0]["label"])
    print(f"wrote train/test .csv + .jsonl + stats.json to {OUT}")


if __name__ == "__main__":
    main()
