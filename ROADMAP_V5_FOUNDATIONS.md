# Glassbox V5 — From Research Demo to Production Compliance Engine
## The honest gap analysis, the mathematical foundation, and the path to any-model, any-prompt, billion-ops auditing

*Drafted 2026-06-12, revised same day after self-review (Part 9 holds the red-team findings). This document states problems plainly before solving them. Estimates are labeled. Research risks are named, not hidden. Regulatory note: the Digital Omnibus (provisionally agreed 7 May 2026, pending adoption) defers Annex III enforcement to 2 Dec 2027 — timelines here are driven by product strategy, not the legal date.*

---

## Part 0 — The five brutal truths

Before the solutions, the problems, stated the way a hostile reviewer would state them:

**Truth 1 — The counterfactual trap.** `analyze(prompt, correct, incorrect)` assumes every decision is a forced choice between two single tokens, with a clean corrupted twin of the prompt. IOI has this structure by construction. A bank's real prompt — three paragraphs of customer context ending in a free-text recommendation — has neither a single decision token nor a natural counterfactual. Today the user must compress reality into our API's shape, and that compression is unvalidated.

**Truth 2 — The scale wall.** Companies run 10⁶–10⁹ inferences/day. Circuit discovery costs ~2s on GPT-2 Small and minutes-to-hours on 70B. Per-request deep auditing is not 100x away — it is 10⁶x away, and no hardware roadmap closes that. Anyone selling per-request circuit discovery at production scale is lying. We must not be that company.

**Truth 3 — Faithfulness degrades exactly where the money is.** Our own published numbers: F1 0.49 on GPT-2 Small, worse on Medium/Large. Bigger models distribute computation; head-level circuits blur; first-order Taylor approximations strain. The models enterprises actually deploy are the ones where our current unit of analysis (attention heads) is weakest.

**Truth 4 — The TransformerLens ceiling.** Our model support is TL's model list, with TL's memory overhead (weight duplication), TL's release lag on new architectures, and no MoE/SSM story. "Works on any model" is today false; "works on any model TransformerLens has ported, smaller than your RAM" is true.

**Truth 5 — The regulation we cite is system-level; the anxiety we sell is decision-level.** Annex IV (Art. 11) documents the *system*. Logging (Art. 12), post-market monitoring (Art. 72), and the right to explanation of individual decisions (Art. 86) are per-operation. Conflating these is both a legal error and — handled correctly — our biggest product opportunity.

Everything below is engineered to convert each truth into a moat.

---

## Part 1 — The reframe that solves scale: audit the policy, not the request

The wrong mental model: "we must explain a billion requests." The right one, borrowed from every discipline that audits high-volume systems (financial audit sampling, SRE distributed tracing, clinical-trial statistics):

> **Deep analysis on a statistically representative sample, cheap invariant-monitoring on everything, with drift triggers that force re-analysis.**

This becomes the **two-loop architecture**:

### Loop 1 — Offline Deep Audit (slow, rigorous, per system version)
On each model/prompt-template release: draw a stratified sample from the *real production input distribution*, run full circuit discovery + faithfulness + bias probes on the sample, and emit the Annex IV technical file with **population-level confidence intervals** (Part 2.3). Sample size comes from a per-stratum power analysis via SampleSizeGate — the often-quoted n ≥ 126 is one example output (ρ_min = 0.25, power = 0.8), not a universal constant; smaller detectable effects need more. Cost: hours, monthly or per-release. This is what Art. 11 actually requires.

### Loop 2 — Online Fingerprint Monitor (fast, per request, microseconds)
Per request, do NOT discover circuits. Instead compute a **circuit fingerprint**: the model's activations at the ~k circuit-critical locations identified by Loop 1, projected to a low-dimensional sketch (Part 5). Log fingerprint + decision functional value (Art. 12 logging). *Cost reality check: 128-dim fp32 fingerprints at 10⁹ req/day ≈ 512 GB/day raw — the design therefore needs tiered retention (full fingerprints sampled at ~1%, per-window sufficient statistics for the rest), and fingerprints derived from user prompts are likely personal data under GDPR, requiring a DPIA before this loop ships (Part 9).* Statistical drift tests on the fingerprint stream (Part 2.5) answer one question continuously: *"is production still inside the regime the deep audit certified?"* When no → alert + automatic Loop 1 re-run (Art. 72 post-market monitoring, automated).

### Loop 3 — On-demand Decision Card (cheap, per contested decision)
When a specific decision is challenged (Art. 86 / GDPR 22), generate a **decision card** for that single request: token-level input attribution (O(2) passes), the decision functional value, the fingerprint's distance from the certified regime, and a pointer to the governing deep audit. Seconds, not hours — because the heavy lifting was amortized in Loop 1.

**This is the answer to "billions of ops per day":** nobody — not us, not Anthropic, not anyone — can deep-audit every request. The defensible claim is: *every request is monitored against a certified baseline; every challenged request gets a real explanation; the baseline is re-certified on drift or release.* That claim is achievable, mathematically grounded, and maps 1:1 onto Articles 11, 12, 72, and 86.

---

## Part 2 — The mathematical foundation

Five formal objects replace today's IOI-shaped assumptions.

### 2.1 The Decision Functional (kills Truth 1's single-token assumption)

Generalize "logit diff" to a functional over the output distribution. Let the model define p(y|x). A **decision functional** is

> **D(x) = g( E_{y~p(·|x)}[ f(y) ] )**

with three canonical instantiations:

- **Verbalizer sets** (classification-shaped decisions): partition answer space into semantic sets A (approve-cluster: "approved", "approve", "yes", "✓ Approved"…) and B (deny-cluster). Then D(x) = log Σ_{t∈A} p(t|x) − log Σ_{t∈B} p(t|x). This is the multi-token, multi-verbalization generalization of the current logit diff — backward compatible (singleton sets recover exactly today's metric).
- **Sequence decisions** (the model generates a recommendation): D(x) = log p(y_decision | x) aggregated over the decision span, where the decision span is located by constrained decoding or a verifier. Attribution flows through the same gradient machinery, summed over span positions. *Known weakness: the span-locating verifier is itself a model — a circularity (using a model to decide what to audit in a model) whose error rate must be measured and reported per task family, not assumed zero (see Part 9).*
- **Score decisions** (model outputs a number): D(x) = the expected numeric value under the output distribution over digit tokens.

Everything downstream (attribution, sufficiency, comprehensiveness, ACDC's KL pruning) is already defined relative to a scalar objective — swapping logit-diff for D is a *surgical* change, not a rewrite. **This single abstraction is the highest-leverage item in this document.**

### 2.2 Attribution over computation-graph partitions (kills Truth 4's transformer-lock)

Current attribution is defined over attention heads. Generalize: let the forward computation be a graph G; let **U = {u₁…u_m} be any partition of G's nodes into units**. First-order causal attribution of unit u:

> **φ(u) = E_c [ ∇_{a_u} D · (a_u(x) − a_u(c(x))) ]**

with the existing Hessian error certificate ε(u) = ½·δaᵀH_uδa bounding when first-order is trustworthy, and exact activation patching (already implemented) as the escalation for top-k units.

The partition U is supplied by an **architecture adapter**, not hard-coded:
- Transformer (MHA): heads + MLPs — today's behavior, unchanged.
- GQA: KV-groups (already in multi_arch.py).
- **MoE: experts as units** — expert-level attribution is arguably *more* natural than head-level (routing is already discrete).
- **SSM/Mamba: state channels per block.**
- Future architecture X: whatever decomposition its adapter declares.

The math never changes again; only adapters do. That is what "supports future models" means concretely (engineering side: Part 4).

### 2.3 Distributional faithfulness with confidence bounds (kills Truth 1's single-prompt audit; makes Truth 3 honest)

Today: faithfulness of one circuit on one prompt. Production: faithfulness of the *policy* over the input distribution P. Define for circuit C:

> **Suff(C) = E_{x~P̂}[ suff(C, x) ] , Comp(C) = E_{x~P̂}[ comp(C, x) ]**

estimated on the stratified sample P̂ (stratify by: decision outcome, input length, demographic-proxy bins for Art. 10(5), template variant). Report **BCa bootstrap CIs** (already implemented) plus an empirical-Bernstein lower bound, so the technical file states:

> "With 95% confidence, circuit sufficiency over the production distribution ≥ 0.68."

A regulator can consume that sentence. No competitor produces it. The grade (A–D) then attaches to the *lower bound*, not the point estimate — strictly more conservative than today, which is exactly the right direction for a compliance tool. Robustness across corruption strategies (already implemented as MultiCorruptionPipeline) becomes the inner expectation E_c of 2.2.

### 2.4 Causal abstraction as the compliance statement (the end-state moat)

The legally meaningful question is not "which heads fired" — it is *"does the model implement the decision policy the provider declared?"* The formal tool exists and is partially in the repo already (DAS): **causal abstraction / interchange intervention accuracy**. Provider declares a high-level causal model G_policy (e.g., `income, credit_history → affordability → decision`, with `gender, ethnicity ↛ decision`). We test alignment:

> **IIA(G_policy) = P( swapping the low-level representation of variable v swaps the high-level decision as G_policy predicts )**

- High IIA on permitted paths = the model implements the declared policy (Art. 13 transparency, Art. 9 risk).
- Any measurable IIA on *forbidden* paths (protected attributes → decision) = quantified discrimination evidence (Art. 10(2)(f), 10(5)) — with effect sizes, not vibes.

This turns the product from "explains circuits" into "**certifies policy-implementation alignment**" — the strongest possible compliance artifact, and a research lane where we already have code (das.py, causal_scrubbing.py) ahead of any commercial player.

### 2.5 Sequential drift detection (the Art. 72 engine)

The fingerprint stream (Loop 2) is a time series f_t ∈ ℝ^d. Per-batch: population-stability index / KL vs. the certified baseline distribution. Per-stream: **CUSUM sequential tests** with controlled false-alarm rate (ARL), so drift alerts have stated statistical guarantees rather than thresholds someone eyeballed. CircuitDiff (already shipped) becomes the *between-versions* member of the same family; this adds the *within-version, across-time* member.

### 2.6 What stays from today, what changes

| Today | V5 | Status |
|---|---|---|
| Logit diff (2 tokens) | Decision functional D (sets/spans/scores) | Generalization — backward compatible |
| Heads as units | Adapter-declared partitions | Generalization |
| Per-prompt faithfulness | Distribution-level with CI lower bounds | Strictly more rigorous |
| Name-swap counterfactuals | Counterfactual distributions, auto-generated + verified (Part 3) | Hardening |
| Grades on point estimates | Grades on CI lower bounds | More conservative |
| CircuitDiff per release | + CUSUM fingerprint drift per request stream | New |
| DAS as a feature | Causal abstraction as the headline certificate | Promotion |

---

## Part 3 — Auto-counterfactuals for arbitrary prompts (the "any random prompt" answer)

The pipeline for a prompt we've never seen:

1. **Factor extraction.** Identify the causal variables in the prompt: the decision-relevant factors (income, score, symptoms) and the protected/irrelevant ones (name, gender markers, nationality). Method: structured extraction with a small local model + rule patterns; output is a typed factor table. (Air-gap-safe: the extractor runs locally.)
2. **Counterfactual generation.** For each factor, minimal-pair edits: numeric perturbation for quantities, swap-from-calibrated-lists for protected attributes, antonym/negation for categorical factors (prompt_corruption.py already has 4 strategies — this extends them into a *distribution* over counterfactuals, not a single pair).
3. **Counterfactual verification — the step everyone skips.** A generated counterfactual is valid only if (a) it still parses as the same task, (b) token alignment is recoverable (or we use position-robust patching at shared anchor points), (c) the model's decision functional actually moves (|ΔD| above noise floor — a counterfactual that changes nothing measures nothing). Invalid counterfactuals are discarded and *reported as discarded* (the count goes in the technical file; silence is how tools lie).
4. **Attribution under the counterfactual distribution** per 2.2's outer E_c, robustness per MultiCorruption, sample size per SampleSizeGate.

When no valid counterfactual survives verification → **the report says so** and degrades one tier (Part 6) rather than fabricating a circuit. A compliance tool's killer feature is refusing to hallucinate evidence.

---

## Part 4 — Any model, including ones that don't exist yet (kills Truth 4)

### 4.1 The Auditable Interface

Define the minimal contract the math needs — nothing more:

```
AuditableModel (protocol):
  forward(tokens) -> logits                     # with grad
  units() -> list[UnitSpec]                     # the partition (2.2)
  read(unit, tokens) -> activation              # hook read
  patch(unit, value) -> context                 # hook write
  tokenizer, config (n_units, d_model, ...)
```

Five capabilities. Anything implementing them is fully auditable — white-box tier.

### 4.2 Three adapter backends, in priority order

1. **Native HF backend (the unlock).** Implement the protocol directly on `transformers` models with PyTorch forward hooks — no weight conversion, no duplication, works the day a model ships on HF, inherits HF's quantization/device-map/FlashAttention. TransformerLens becomes *one backend* (the most featureful for research) instead of *the* dependency. This single move takes "supported models" from TL's ported list to ~everything-on-HF and roughly halves memory. (NNsight/pyvene validate the approach; we can interop or implement lean. Effort honesty: this is a multi-week-to-multi-month build for one person, not a sprint item.)
2. **TL backend** — kept for research-grade features (it already works).
3. **Black-box backend** — already shipped (audit.py); it implements the same protocol with `units() = []`, which forces behavioral-tier reports. One API, three tiers, honest labels.

### 4.3 The conformance suite (how "future models flawless" is actually enforced)

A new adapter is accepted only if it passes `tests/adapter_conformance/`:
- Reconstruction: Σ unit contributions ≈ logits (tolerance bound)
- Patch identity: patching a unit with its own activation changes nothing
- Known-circuit recovery: on GPT-2-class reference models, recovers IOI top heads (regression vs. published research)
- Determinism: same input, same attribution, bit-for-bit
- Memory envelope: peak RAM within declared budget

New architecture day-one playbook: write adapter (~100–300 lines), pass conformance, ship. Community-contributable — the conformance suite, not trust, is the gatekeeper. This is the same governance trick that makes ONNX/TL themselves work, and it's how a solo-founder project absorbs the model-zoo treadmill.

### 4.4 MoE and beyond (the next-architecture proof)

Mixtral-class MoE is the immediate test of 2.2's generality: units = experts (+ router). Expert-level attribution is *coarser but more causal* than head-level (routing is literally a discrete causal decision). An MoE adapter shipping in the same quarter as the HF backend proves the abstraction isn't vaporware — and no compliance tool on the market has any MoE story at all.

---

## Part 5 — 100x faster (concrete, with mechanisms and honest multipliers)

Current: ~1.8s GPT-2 Small / CPU per audit. The path to effective 100–1000x on *throughput* (not magic latency on 70B):

| # | Technique | Mechanism | Expected gain (estimate) |
|---|---|---|---|
| 1 | GPU batch path | Today's numbers are CPU. Batched audits on one A10/A100, per-sample grads via vmap | 20–50x throughput |
| 2 | KV-prefix sharing | Clean/corrupted prompts share long prefixes; reuse KV cache, recompute only the edited suffix | 1.5–3x per audit — *unresolved risk: gradients must flow through the shared prefix, so naive cache reuse breaks the backward pass; may apply to forward-only passes (exact patching, fingerprints) and not to gradient attribution. Prototype before counting it.* |
| 3 | Hierarchical screening | Layer-level attribution first (O(L) units), expand only top-q layers to heads/features — beam-search the circuit | 3–6x on big models — *accuracy trade-off: screening can miss circuits distributed across many weakly-contributing layers; the false-negative rate must be measured against full search before this ships in a compliance path* |
| 4 | Circuit caching by task-family | Same template+model ⇒ same circuit (validated by fingerprint match); discovery amortized to once per family, verification per batch | 10–100x on repeated workloads — *this is the production reality: companies run templates, not novel prompts* |
| 5 | Sketched fingerprints | JL-projection of circuit activations to ℝ¹²⁸; Loop-2 monitoring cost → microseconds | enables per-request coverage at any scale |
| 6 | Quantized forward, fp32 grad accumulation | 8-bit weights for forwards; full-precision only where gradients accumulate | 2–4x memory ⇒ bigger models per box |
| 7 | Async audit queue | The API stops loading models per request (LRU cache, already specced) and runs audits as jobs | UX: seconds → instant submit |

Items 1+4 alone make the honest claim: **"deep-audit a production template library overnight; monitor every request in real time."** That sentence survives diligence. "Audit a billion requests deeply" never will — and our competitors saying it is our marketing material.

---

## Part 6 — Never fail: the degradation ladder (the "without failing" answer)

Every audit request returns a report. What varies is the **evidence tier**, printed on the report's first page:

| Tier | Method | When |
|---|---|---|
| **A — Causal-certified** | Full circuit + exact patching verification + causal abstraction (IIA) + CI bounds | Open weights, valid counterfactuals, adequate sample |
| **B — Causal-screened** | First-order attribution + Hessian certificate clean + distributional faithfulness | Open weights, Hessian dominance < threshold |
| **C — Behavioral** | Black-box: counterfactual probing, sensitivity, consistency | API-only models, or counterfactual verification failed |
| **D — Descriptive** | System metadata + logging architecture + monitoring plan only | Nothing else available — still satisfies the documentation *structure* |

Rules: a tier downgrade is never silent (the reason is stated in §4 of the report); the grade thresholds attach to CI lower bounds (2.3); a report that would require fabricating evidence instead states "insufficient evidence at tier X, escalation path: Y." **The product never errors out and never lies — it degrades with disclosure.** For a compliance buyer, that property is worth more than any benchmark.

---

## Part 7 — EU mapping (so every engineering choice has a legal address)

| Component | Article |
|---|---|
| Loop 1 deep audit + technical file | Art. 11 + Annex IV (all 9 sections) |
| Per-request fingerprint + D-value logging | Art. 12 (record-keeping) |
| CUSUM drift alerts + auto re-audit | Art. 72 (post-market monitoring) |
| Decision cards on challenge | Art. 86 (explanation of individual decisions) + GDPR interplay |
| IIA on forbidden paths (protected attrs) | Art. 10(2)(f), 10(5) (bias) |
| Hessian certificates + tier disclosure | Art. 13 (transparency — including about our own limits) |
| Human sign-off retained (§8, tier label visible) | Art. 14 (human oversight) |

*(Mapping to be reviewed by counsel; this table is engineering intent, not legal advice.)*

---

## Part 8 — Sequencing (what to build, in order)

**Phase A — weeks, before launch (mostly wiring around existing math). Note: the Digital Omnibus (agreed 7 May 2026, pending adoption) defers Annex III enforcement to 2 Dec 2027 — Phase A timing is driven by our launch, not the legal date:**
1. Decision functional v1: verbalizer sets + multi-token spans (2.1) — replaces the single-token straitjacket
2. Counterfactual verification gate + discard reporting (3.3)
3. Degradation ladder + tier labels on every report (6)
4. API: LRU model cache + async job queue (5.7)

**Phase B — Q3–Q4 2026 (the platform):**
5. Native HF backend + conformance suite (4.1–4.3) → "any HF model"
6. GPU batch path + KV-sharing + hierarchical screening (5.1–5.3)
7. Circuit cache + fingerprint Loop 2 + CUSUM drift (2.5, 5.4–5.5)
8. MoE adapter (Mixtral) as the generality proof (4.4)
9. Distributional faithfulness CIs in the vault (2.3)

**Phase C — 2027 (the moat):**
10. Causal-abstraction certificates as the headline product (2.4)
11. Feature-level units via SAEs/transcoders where head-level faithfulness is weak (the research field's direction; sae_attribution.py is the seed)
12. SSM/Mamba adapter; multi-framework report packs (ISO 42001, NIST AI RMF) from the same evidence base

**Named research risks (we do not pretend these are solved):** feature-level attribution at frontier scale is open research; IIA testing needs per-domain causal models (consulting-shaped work, also a revenue line); faithfulness on 70B+ may stay modest — our position is to *measure and disclose it*, which is a product property no competitor can copy without first admitting their own numbers.

---

*One-sentence summary: stop promising to explain every request — build the system that certifies the policy, monitors every request against the certificate, explains any challenged decision on demand, and never produces evidence it can't defend.*

---

## Part 9 — Red-team review & open problems (self-audit, 2026-06-12)

This roadmap was reviewed against its own evidence standard the day it was written. Findings, unvarnished:

### 9.1 Epistemic status of every major claim

| Claim | Status |
|---|---|
| Five brutal truths (Part 0) | **Fact** — each verifiable in the repo or the regulation text |
| Two-loop reframe (Part 1) | **Sound by analogy** to audit sampling / SPC; not yet validated with a single customer workload |
| Decision functional, verbalizer form (2.1) | **Established technique** (prompt-based classification literature); integration here is engineering |
| Decision functional, sequence form (2.1) | **Research-grade** — verifier circularity unmeasured |
| Graph-partition attribution (2.2) | **Straightforward generalization** of existing code; MoE/SSM instantiations unbuilt |
| Distributional faithfulness CIs (2.3) | **Statistics are standard**; the claim a regulator will accept the format is **untested — zero regulatory validation to date** |
| Causal-abstraction certificates (2.4) | **Open research at production scale**; per-customer causal models are consulting-shaped work |
| CUSUM drift on fingerprints (2.5) | **Classic math**; calibrating false-alarm rates on high-dim activation streams is unexplored engineering |
| Every multiplier in Part 5 | **Unmeasured estimate** — hypotheses to benchmark, not claims to repeat |

### 9.2 Known technical holes (flagged inline, collected here)

1. **KV-prefix sharing vs. autograd** (5.2) — may not survive the backward pass; prototype first.
2. **Hierarchical screening false negatives** (5.3) — distributed circuits can evade layer-level screening; quantify before use in compliance paths.
3. **Verifier circularity** (2.1) — span-locating models auditing models; per-task error rates required.
4. **Fingerprint economics** (Loop 2) — ~512 GB/day raw at 10⁹ req/day; tiered retention design required.
5. **Sample-size folklore** (Loop 1) — n ≥ 126 is one power-analysis output, not a constant.

### 9.3 What this document still lacks

- **A DPIA / privacy design** for Loop 2 — fingerprints derived from user prompts are likely personal data; the compliance tool must not create its own GDPR problem. *Must exist before Loop 2 ships.*
- **An evaluation benchmark for the decision functional** — a labeled suite of non-IOI tasks (credit, triage, screening) proving D-based attributions remain faithful. *Without this, Part 2.1 is theory.*
- **A current literature & competitor scan** — this document was written by an AI whose reliable knowledge ends May 2025, thirteen months before its drafting date; the interpretability field (feature-level attribution, attribution graphs, commercial entrants) has moved in ways not reflected here. *A fresh scan is a prerequisite for Phase C commitments.*
- **A cost model** for Loop 1 at enterprise scale (audit-hours × model size × release cadence).

### 9.4 Sequencing discipline (the business constraint that outranks the math)

Distribution precedes platform: Phase A ships only what launch and design partners need (tier labels and the counterfactual gate are days of work; the decision functional ships when a design partner's use case demands it). **Phase B does not start before the first paying customer.** The fastest way to falsify or validate everything in this document is five design partners running the tool on real workloads — no amount of further roadmap-writing substitutes for that.
