# Glassbox — Validation Log

A reproducible record of what has actually been run, on what hardware, with what
result. Only runs that happened are listed. "Not yet validated" means exactly
that — no claim is made beyond these rows.

Harnesses: `scripts/validate_auditable_gpt2.py` (single-model gate),
`scripts/validate_models_matrix.py` (multi-model matrix). Probe: IOI
("When Mary and John went to the store, John gave a drink to" → " Mary" vs
" John"), a known cross-model capability used to test that the pipeline works on
an architecture — not a business decision.

---

## Run 1 — multi-architecture + billion-scale matrix

- **Date:** 2026-06-14
- **Hardware:** Apple Silicon Mac (arm64), **CPU only**
- **Stack:** torch 2.10.0, transformer-lens 2.17.0, transformers 4.57.6,
  glassbox-mech-interp 4.5.0 (`.venv-torch`)
- **What ran:** for each model — head-level `AuditableModel` adapter +
  `run_conformance` (determinism + patch-identity on real forward hooks), then a
  full `analyze()` on the IOI probe (circuit + ERASER faithfulness).

| Model | Family | Arch | Conformance | Circuit | F1 | Grade | Time |
|---|---|---|---|---|---|---|---|
| distilgpt2 | GPT-2 | 6L×12H | **PASS** | 1 | 1.00 | A | 160s* |
| gpt2 | GPT-2 | 12L×12H | **PASS** | 1 | 0.704 | B | 7s |
| EleutherAI/pythia-70m | Pythia / GPT-NeoX | 6L×8H | **PASS** | 17 | 0.00 | D¹ | 76s |
| EleutherAI/pythia-160m | Pythia / GPT-NeoX | 12L×12H | **PASS** | 1 | 0.97 | A | 178s* |
| EleutherAI/gpt-neo-125M | GPT-Neo | 12L×12H | **PASS** | 1 | 1.00 | A | 119s* |
| facebook/opt-125m | OPT | 12L×12H | **PASS** | 1 | 0.554 | C | 128s* |
| EleutherAI/pythia-1b | Pythia / GPT-NeoX | 16L×8H | **PASS** | 1 | 0.867 | A | **1335s** |

\* Time includes the first-time model download. ¹ pythia-70m: the tool emitted
*"clean logit diff −0.974 (≤0)… the model prefers the distractor… treat results
as unreliable"* — i.e. the model cannot perform IOI, and Glassbox correctly
flagged the result as unreliable rather than reporting a misleading circuit. This
is intended honest behaviour, not a gate failure (conformance still PASSED).

**Result:** conformance gate **7/7 PASS** across **4 architecture families**
(GPT-2, Pythia/GPT-NeoX, GPT-Neo, OPT) and **up to ~1B parameters**
(`pythia-1b` ≈ 1.0B actual; TransformerLens's internal heuristic over-reported
~2.1B). The full analyze→circuit→faithfulness pipeline ran end to end on every
model, and self-flagged the one model (pythia-70m) that cannot perform the probe.

**Cost note:** the 1B audit took **~22 minutes on CPU**. This confirms Glassbox
is an **offline / sampled** audit tool (not real-time) and that **GPU is required
for speed and for anything larger**.

---

## What is NOT yet validated (no claims made here)

- **GQA architectures** (Llama-3, Mistral, Phi-3) — none of the models above use
  grouped-query attention; the `multi_arch` GQA adapter remains unproven on a real
  GQA model. *Highest-value next test* (needs an HF token).
- **7B+ frontier scale** — needs a GPU (24–80GB). Feasible next step on rented
  cloud GPU; not yet run.
- **70B–200B** — requires multi-GPU sharding via the native-HF/distributed
  backend, which is built as an interface but **not yet validated**. No 200B run
  has been performed; no 200B claim is made.
- **GPU batching / production throughput** — unproven.
- **Closed APIs** (GPT-4, Claude, Gemini) — out of scope by construction (no
  activations/gradients available).

---

## How to reproduce

```bash
pip install "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6" scipy
pip install -e . --no-deps
python scripts/validate_models_matrix.py            # the small-model matrix
python scripts/validate_models_matrix.py --models EleutherAI/pythia-1b --device cpu --max-units 12
# On a GPU box, add --device cuda and larger --models (e.g. a small GQA model with an HF token).
```
