# Large-Model Live Verification Runbook

**Purpose:** turn "Llama-3 support is implemented and unit-tested" into "verified end-to-end on a real ≥7B model, here is the receipt." This is the single highest-leverage technical de-risking step before enterprise outreach.

> **Why it matters commercially.** Registering `llama-3-70b` in a config dict is not the same as running a forward + attribution pass on it. No EU risk officer will accept "we support 70B" without a live artifact. This runbook produces that artifact.

---

## What's already in place

- `glassbox/large_model.py` — `estimate_memory()`, `analyze_large()`, `LargeModelAttributionPatcher`, `ActivationOffloader`, `classify_model_size()`.
- `tests/test_large_model_live.py` — opt-in, slow, real-model integration tests (added alongside this runbook).
- 183 mock-based architecture tests in `tests/test_multi_arch.py` (these prove detection logic, **not** live execution).

## The verification ladder

Run these in order. Stop and fix at the first failure — do not skip rungs.

### Rung 1 — GPT-2, CPU (proves the plumbing, costs nothing)
```bash
GLASSBOX_LIVE_TEST=1 pytest tests/test_large_model_live.py -v -s
```
Confirms `analyze_large` returns a valid result, the managed path matches the standard path, and the any-prompt engine fires on a non-IOI prompt. **Gate:** all 4 tests pass.

### Rung 2 — A real 7–8B model on GPU
Requires one A100/L4/4090-class GPU (or rented cloud GPU). Use bf16.
```bash
GLASSBOX_LIVE_TEST=1 \
GLASSBOX_LIVE_MODEL=meta-llama/Meta-Llama-3-8B \
pytest tests/test_large_model_live.py -v -s 2>&1 | tee verification-8b.log
```
**Gate:** `test_analyze_large_end_to_end` and `test_large_path_matches_standard_path` pass. F1 in range, circuit non-empty, corruption metadata present.

### Rung 3 — Capture the receipt
Save `verification-8b.log` plus:
- model name + revision/commit hash
- GPU type, dtype, `estimate_memory()` output and chosen strategy
- wall-clock audit time
- the generated Annex IV vault (`build_annex_iv_vault`) JSON hash

Publish a short note ("Verified: Annex IV audit on Llama-3-8B in N seconds on a single A100"). That sentence unlocks the enterprise pitch.

### Rung 4 — Wire into CI as an opt-in slow job
Add a manually-triggered (`workflow_dispatch`) GitHub Actions job on a GPU runner that runs Rung 2 monthly. Keep it **off** the default PR path so normal CI stays fast and free.

## Known pitfalls to watch (ML-correctness)

These are the failure modes most likely to bite the large path — review if a rung fails:

- **Integrated gradients inside `torch.no_grad()`** returns zero. Attribution must run under `torch.enable_grad()`. (Default method is `taylor`; only relevant if you pass `method="integrated_gradients"`.)
- **Hook accumulation across passes** — verify hooks are removed between the 3 forward passes; a leaked hook silently corrupts the second model's cache.
- **`run_with_cache` without `names_filter`** on a 70B model will OOM. The large path must cache only the hooks it needs.
- **KL / float32 summation over 128K-dim vocab** — convert to float64 before summing to avoid precision loss in edge pruning.
- **bf16 recomputation noise** — gradient checkpointing recomputes activations; expect small (<0.05) F1 differences vs the standard path in bf16. `test_large_path_matches_standard_path` encodes this tolerance.

## Honesty discipline (until Rung 3 is done)

In README, website, and sales material, say: **"large-model support: implemented and unit-tested; live ≥7B verification in progress."** Do not claim "tested on 70B" until a 70B log exists. The credibility cost of one inflated claim to a risk officer is higher than the benefit.

*Glassbox AI v4.3.0 — large-model verification runbook.*
