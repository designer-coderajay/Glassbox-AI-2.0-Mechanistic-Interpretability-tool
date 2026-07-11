# Phase 1 Results — Auditing a Real Credit Decision Model

*Honest findings note. All numbers are measured, reproducible from seed, and logged
with provenance in `docs/PLAN_DECISION_AUDIT.md`. Data is self-owned (synthetic);
model is EleutherAI/pythia-1.4b (Apache-2.0); audited with glassbox-mech-interp 4.5.1.*

## What we did
1. Defined a **known, deterministic credit rule** (`credit_rule.py`):
   `approve ⇔ score ≥ 640 ∧ dti < 0.40 ∧ defaults = 0`.
2. Generated **self-owned synthetic data** with that rule as ground truth (2,000 train /
   500 test, balanced, 100% labels match the rule).
3. **QLoRA fine-tuned** pythia-1.4b to make the decision.
4. **Audited** the fine-tuned model with Glassbox (circuit + faithfulness + Annex IV).
5. **Validated faithfulness against the known rule** — the payload of the whole exercise.

## Headline numbers (latest run, Colab A100, seed 42)
| Metric | Value |
|---|---|
| Held-out accuracy | **0.996** — the model genuinely learned the rule |
| Faithfulness F1 (distribution, N=27, per-label audit) | **0.690 ± 0.231** |
| Comprehensiveness | 0.826 (circuit is largely *necessary*) |
| Sufficiency | 0.628 |
| Grade distribution (per prompt) | A:6 · B:5 · C:11 · D:5 |
| Ground-truth flip — rule-relevant (credit_score) | **0.68** (want high) |
| Ground-truth flip — distractor (age, loan) | **0.00** (want ~0) |

## What it means (the honest reading)
- **Behavioural faithfulness to the rule is strong and reproducible.** Across every run,
  changing a rule-relevant feature flips the decision ~0.68 of the time while changing a
  distractor flips it ~0.00 — the model decides on the *right* features. This is the
  robust result.
- **Circuit-metric faithfulness is moderate and high-variance.** Mean F1 ranged
  **0.49–0.69 across runs** (fine-tuning is non-deterministic on GPU, and the per-prompt
  std is ~0.23). Per-prompt grades span A→D. The defensible statement is therefore the
  **distribution**, not a single point grade — consistent with Glassbox's own Tier-C
  warning that single-prompt audits are underpowered.
- **`exact_circuit` coincided with the default here.** We expected the scale-aware
  exact-sufficiency circuit to lift the grade (as it does at larger scale); it did not —
  for pythia-1.4b on this task the default circuit already meets the sufficiency target,
  so the two are identical. We report this because we measured it; we did not assume it.

## Methodological findings (reusable)
1. **Audit toward each example's true decision**, not a fixed token. On a balanced set,
   auditing every prompt as "approved" makes half the audits ungrounded (negative
   clean-logit-diff). Per-label auditing + skipping mispredictions is required.
2. **Default corruption is mis-suited to numeric/tabular prompts.** It often fails to
   move the decision ≥1% → `null_effect`, ungrounded attribution, added noise. Use
   `corruption_strategy="mean_ablation"` (or `gaussian_noise`) for decision tasks.
3. **Dogfooding caught a real bug.** The large-model VRAM-warning estimate over-counted
   by ~n_heads/3× (reported pythia-1.4b as 6.4B); fixed to `12·n_layers·d²` (Kaplan 2020)
   and shipped in **v4.5.1** — now correctly ~1.2B.

## Limitations (stated plainly)
- A **demonstration model on synthetic data** — not a production underwriter.
- **Accuracy ≠ faithfulness**: the model is 99.6% accurate yet circuit faithfulness is
  moderate; these are different quantities and we keep them separate.
- Single-task, single architecture; Phase 2 corroborates on the real UCI German Credit
  dataset (CC BY 4.0).

## Reproduce
`Glassbox_credit_experiment_v451.ipynb` (Colab GPU) → installs glassbox-mech-interp 4.5.1,
regenerates the data from seed 42, fine-tunes, merges, audits. Data generator:
`generate_credit_data.py`. Rule: `credit_rule.py`. Provenance + references:
`docs/PLAN_DECISION_AUDIT.md`.
