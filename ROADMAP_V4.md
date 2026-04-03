# Glassbox v4.0 Roadmap — Foundational Mathematical Rigor

> **From proof-of-concept to research-grade.**
> This roadmap closes every mathematical gap identified against Anthropic, DeepMind, Stanford, MIT, and Meta interpretability standards. Each fix is grounded in a specific peer-reviewed paper and ships as a versioned, testable feature.

---

## Current State Assessment (v3.6.0)

| Component | Status | Gap vs Major Labs |
|-----------|--------|-------------------|
| Attribution patching (Taylor) | ✅ Correct formula | ❌ Error unquantified |
| Head-level circuit discovery | ✅ Working | ❌ Edge-level missing |
| Faithfulness metrics (S, Comp, F1) | ✅ Correct formulas | ❌ n=8 → underpowered |
| Bootstrap CIs | ✅ Implemented | ❌ Needs n≥100 |
| Corruption strategy | ⚠️ Single type only | ❌ Multi-corruption required |
| Held-out validation | ❌ Not implemented | ❌ Overfitting possible |
| LayerNorm correction | ❌ Not implemented | ❌ Systematic bias |
| SAE feature decomposition | ❌ Not implemented | ❌ Polysemanticity ignored |
| Edge attribution (EAP) | ❌ Not implemented | ❌ Compositional structure lost |
| Approximation error bounds | ❌ Not implemented | ❌ Hessian terms ignored |
| Causal scrubbing | ❌ Not implemented | ❌ Causal verification weak |

---

## Release Plan

```
v3.6.0 (current)  →  v3.7.0 (3–4 weeks)  →  v4.0.0 (8–10 weeks)  →  v4.1.0 (4–5 months)
   proof-of-concept      statistical rigor        mathematical depth       research-grade
```

---

## v3.7.0 — Statistical Rigor  *(3–4 weeks, CPU-viable)*

**Goal:** Close all "Easy" gaps. Any lab can reproduce these results and trust the confidence intervals.

### Feature 1: Multiple Corruption Types

**Why it matters.** Using a single corruption (name-swap) means the circuit we find may be specific to that perturbation rather than genuinely causal. Wang et al. (2022) validated the IOI circuit against three independent corruption strategies. Without this, sufficiency/comprehensiveness are perturbation-specific, not mechanistic.

**What to implement:**

```python
class CorruptionStrategy(Enum):
    NAME_SWAP      = "name_swap"       # current: swap IO↔S names
    RANDOM_TOKEN   = "random_token"    # replace IO/S with random vocab token
    GAUSSIAN_NOISE = "gaussian_noise"  # add N(0, σ²) noise to token embeddings
    MEAN_ABLATION  = "mean_ablation"   # replace activations with dataset mean
```

**Mathematical definition for each:**

```
Name-swap:    x_corrupt = x[IO ↦ S, S ↦ IO]
Random token: x_corrupt = x[IO ↦ Uniform(V), S ↦ Uniform(V)]
Gaussian:     z_corrupt = z_clean + ε,  ε ~ N(0, σ²·I),  σ = std(z_clean)
Mean ablation: z_corrupt = E_{x ~ D}[z(x)]   (dataset mean activation)
```

**Stability criterion (from Wang et al., 2022):**

A circuit C is *robustly causal* if:

```
∀ corruption_type k:  |S_k(C) − S̄| < δ  and  |Comp_k(C) − Comp̄| < δ
```

where δ = 0.10 (10% tolerance across corruption types). If this fails for any corruption, the circuit is flagged as `perturbation_sensitive`.

**Output change in `analyze()`:**

```python
result["faithfulness_by_corruption"] = {
    "name_swap":    {"sufficiency": 1.00, "comprehensiveness": 0.675, "f1": 0.806},
    "random_token": {"sufficiency": ...,  "comprehensiveness": ...,  "f1": ...},
    "gaussian":     {"sufficiency": ...,  "comprehensiveness": ...,  "f1": ...},
    "mean_ablation":{"sufficiency": ...,  "comprehensiveness": ...,  "f1": ...},
    "robustness_pass": True/False,
    "cross_corruption_variance": float,
}
```

**References:** Wang et al. (2022) arXiv:2211.00593 §3.2; Conmy et al. (2023) arXiv:2304.14997 §4

---

### Feature 2: Held-Out Validation Split

**Why it matters.** Identifying a circuit on the same prompts used to measure faithfulness is circular. The circuit is tuned (implicitly) to perform well on training prompts. Held-out validation measures whether the circuit generalises.

**What to implement:**

```python
gb.analyze_with_validation(
    prompts,           # full prompt list
    train_frac=0.5,    # 50% for circuit discovery
    seed=42,
)
```

**Protocol:**

```
1. Split prompts → train_set (n/2), test_set (n/2)  [stratified by name pair]
2. Run attribution patching on train_set → circuit C_train
3. Measure S(C_train), Comp(C_train) on train_set  → train metrics
4. Measure S(C_train), Comp(C_train) on test_set   → test metrics (held-out)
5. Report: generalisation_gap = |F1_train − F1_test|
```

**Acceptance threshold:** `generalisation_gap < 0.10`. If gap ≥ 0.10, the circuit is marked `overfit_warning`.

**Statistical test for generalisation:**
Two-sample Welch's t-test on F1 distributions:

```
t = (F1̄_train − F1̄_test) / √(s²_train/n_train + s²_test/n_test)
```

If p < 0.05 (Bonferroni-corrected): emit `CircuitGeneralisationWarning`.

**References:** Standard ML practice; Bishop (2006) PRML §1.3; Geirhos et al. (2020) shortcut learning

---

### Feature 3: Sample Size Enforcement + Power Analysis Gate

**Why it matters.** At n=8, statistical power is ~30%. The library currently runs silently on any n. It should refuse (or warn loudly) when n is too small to support the claims being made.

**What to implement:**

```python
class SampleSizeGate:
    """
    Power analysis gate. Blocks or warns when n is insufficient.
    Formula: n_min = ((z_{α/2} + z_β) / atanh(ρ_min))² + 3
    Default: α=0.05, β=0.20 (80% power), ρ_min=0.28
    → n_min = 100
    """
    WARN_THRESHOLD  = 50   # warn if n < 50
    BLOCK_THRESHOLD = 20   # raise SampleSizeError if n < 20

    @staticmethod
    def check(n: int, alpha: float = 0.05, power: float = 0.80,
              rho_min: float = 0.28) -> PowerAnalysisResult:
        z_alpha = stats.norm.ppf(1 - alpha / 2)   # 1.960
        z_beta  = stats.norm.ppf(power)            # 0.842
        n_min   = ((z_alpha + z_beta) / np.arctanh(rho_min))**2 + 3
        return PowerAnalysisResult(n=n, n_min=n_min, adequate=(n >= n_min))
```

**References:** Cohen (1988) §3; Fisher (1915); `MATH_FOUNDATIONS.md §10`

---

### v3.7.0 API Surface

```python
gb.analyze(prompt, correct, incorrect,
    corruptions=["name_swap", "random_token", "gaussian"],  # NEW: multiple corruptions
    held_out_frac=0.5,                                       # NEW: validation split
    min_n=50,                                                # NEW: power gate
)
```

---

## v4.0.0 — Mathematical Depth  *(8–10 weeks, GPU recommended)*

**Goal:** Close all "Medium" gaps. Results are now comparable to what Anthropic, DeepMind, and Stanford would accept in a paper submission.

### Feature 4: Folded-LayerNorm Correction

**Why it matters.** The standard attribution patching Taylor approximation computes gradients through the raw residual stream. But the actual forward pass applies LayerNorm before the next layer's attention weights. LayerNorm is nonlinear:

```
LN(x) = (x − μ(x)) / σ(x)  ·  γ + β
```

This nonlinearity means head attributions computed without folding LN are systematically biased — the linear approximation assumes LN is identity. The bias is proportional to the variance of z_h across the sequence position dimension.

**What to implement:**

Fold LayerNorm into the weight matrices at inference time (weights become dataset-dependent but the forward pass becomes linear):

```
W_Q_folded = W_Q · diag(γ / σ(z))
W_K_folded = W_K · diag(γ / σ(z))
W_V_folded = W_V · diag(γ / σ(z))
```

Then run attribution patching on the folded model. The correction term is:

```
Δα(h) = α(h)_folded − α(h)_raw

Bias estimate: B̂ = (1/n) Σᵢ |Δαᵢ(h)|
```

Report `layernorm_bias_estimate` per head alongside each attribution score. Flag heads where `B̂ / |α(h)| > 0.15` (15% relative bias) as `layernorm_sensitive`.

**References:** Elhage et al. (2021) "Mathematical Framework for Transformer Circuits" §4; TransformerLens `fold_ln=True` flag; Nanda (2023) blog §LayerNorm

---

### Feature 5: SAE Feature Attribution

**Why it matters.** Head-level attribution tells you *which* heads are active. It does not tell you *what* those heads are computing. A single attention head can be responsible for multiple unrelated features simultaneously (polysemanticity/superposition — Elhage et al., 2022, arXiv:2209.10652). SAEs decompose the residual stream into a sparse set of monosemantic features, giving you a *feature-level* circuit rather than a *head-level* circuit.

This is Anthropic's current standard. Their 2024 "Scaling Monosemanticity" work scaled SAEs to Claude 3 Sonnet, identifying 34M latent features.

**What to implement:**

```python
class SAEFeatureAttributor:
    """
    Uses a pre-trained Sparse Autoencoder to decompose residual stream
    activations into interpretable features, then attributes each feature's
    contribution to the logit difference.

    Architecture (Bricken et al., 2023):
        encoder: x → ReLU(W_enc · (x − b_dec) + b_enc)   [sparse features]
        decoder: f → W_dec · f + b_dec                     [reconstruction]

    Attribution per feature k at layer l:
        α_SAE(k, l) = (∂LD / ∂f_k) · (f_k^clean − f_k^corrupt)

    where f_k is the activation of the k-th SAE feature at layer l.
    """

    def attribute_features(
        self,
        clean_tokens: Tensor,
        corrupt_tokens: Tensor,
        target_token: int,
        distractor_token: int,
        sae_layer: int,
        top_k_features: int = 20,
    ) -> SAEAttributionResult:
        ...
```

**Integration with existing API:**

```python
result = gb.analyze(prompt, correct, incorrect,
    sae_layer=9,            # which layer's residual stream to decompose
    sae_top_k=20,           # top-k features to report
)
result["sae_features"] = [
    {"feature_id": 14523, "attribution": +0.847, "label": "female names", "monosemantic": True},
    {"feature_id": 8901,  "attribution": +0.623, "label": "indirect object",  "monosemantic": True},
    ...
]
```

**Pre-trained SAEs to support:**
- `EleutherAI/sae-gpt2-small-res-jb` (GPT-2-small, Bloom et al. 2024, Neuronpedia)
- `jbloom/GPT2-Small-SAEs` (TransformerLens compatible)
- Custom-trained SAEs via `gb.train_sae(model, layer, n_features=4096)`

**Polysemanticity score:**

For each head h in the circuit, measure how many distinct SAE features it activates:

```
Polysemanticity(h) = H(p(feature | head_h))   [entropy of feature distribution]
```

High entropy → head is polysemantic → head-level attribution is misleading.
Low entropy → head is monosemantic → head-level attribution is trustworthy.

**References:** Bricken et al. (2023) transformer-circuits.pub; Templeton et al. (2024) transformer-circuits.pub; Gao et al. (2024) arXiv:2406.04093; Samuel et al. (2024) arXiv:2403.19647

---

### Feature 6: GPU-Accelerated n=100 Pipeline

**Why it matters.** n=100 is the minimum for 80% statistical power at |ρ|≥0.28 (from `MATH_FOUNDATIONS.md §10`). On CPU, this takes ~10 minutes. On a T4 GPU, it takes ~45 seconds. The library needs to be GPU-first at scale.

**What to implement:**

```python
gb = GlassboxEngine(model, device="cuda")   # already partially supported

gb.batch_analyze(
    prompts,
    method="taylor",
    batch_size=8,          # NEW: process 8 prompts in parallel on GPU
    compile=True,          # NEW: torch.compile for 2-3x speedup
    mixed_precision=True,  # NEW: fp16 for 2x memory, same results
    show_progress=True,
)
```

**Expected performance at n=100:**

| Hardware | Time (n=100) | Cost estimate |
|----------|--------------|---------------|
| CPU (M2 Pro) | ~10 min | $0 |
| T4 GPU (Colab) | ~45 sec | ~$0.10 |
| A100 GPU | ~12 sec | ~$0.05 |

**Also needed:** `gb.export_benchmark()` to generate standardised timing reports for the paper.

---

### Feature 7: Benjamini-Hochberg FDR Control

**Why it matters.** We currently use Bonferroni correction (α_adj = 0.05/K). Bonferroni is conservative — it controls the Family-Wise Error Rate (FWER). For circuit discovery across 144 heads, Bonferroni requires p < 0.000347 per head, which is extremely restrictive. The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR) instead, giving more statistical power while still providing rigorous guarantees.

**What to implement:**

```python
from scipy.stats import false_discovery_control

def apply_fdr_control(p_values: Dict[Tuple[int,int], float],
                      alpha: float = 0.05,
                      method: str = "bh") -> Dict[Tuple[int,int], bool]:
    """
    Benjamini-Hochberg FDR control.

    Procedure:
    1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
    2. Find largest k: p_(k) ≤ (k/m) · α
    3. Reject all H_(1), ..., H_(k)

    Guarantees: E[FDR] ≤ α  under independence (Benjamini & Hochberg, 1995)
    Under positive dependence (PRDS): use BY procedure instead.
    """
    heads = list(p_values.keys())
    pvals = np.array([p_values[h] for h in heads])
    rejected = false_discovery_control(pvals, alpha=alpha, method=method)
    return {h: bool(r) for h, r in zip(heads, rejected)}
```

**Both methods available:**

```python
result["circuit_significance"] = {
    "bonferroni": {...},      # conservative, FWER control
    "benjamini_hochberg": {...}, # powerful, FDR control  ← new default
    "recommended": "benjamini_hochberg",
    "n_heads_significant_bh": 12,
    "n_heads_significant_bonferroni": 7,
    "expected_false_discoveries": 0.6,  # 5% FDR × 12 significant
}
```

**References:** Benjamini & Hochberg (1995) JRSS-B 57(1); Benjamini & Yekutieli (2001) Annals of Statistics

---

### v4.0.0 API Surface

```python
gb = GlassboxEngine(model,
    device="cuda",
    sae_path="EleutherAI/sae-gpt2-small-res-jb",  # NEW
    fold_layernorm=True,                            # NEW
)

result = gb.analyze(prompt, correct, incorrect,
    corruptions=["name_swap", "random_token", "gaussian", "mean_ablation"],
    sae_layer=9,
    fdr_method="benjamini_hochberg",
)

# Result now contains:
result["faithfulness_by_corruption"]   # robustness across perturbations
result["sae_features"]                 # feature-level decomposition
result["layernorm_bias_estimate"]      # bias correction per head
result["circuit_significance"]         # BH + Bonferroni significance
```

---

## v4.1.0 — Research Grade  *(4–5 months, GPU required)*

**Goal:** Close "Hard" gaps. Results are publishable at NeurIPS / ICML / ICLR Mechanistic Interpretability Workshop.

### Feature 8: Edge Attribution Patching (EAP)

**Why it matters.** Head-level attribution gives you nodes in the computational graph. But the mechanistic story is about *edges* — which head's output flows into which head's input (via Q, K, or V). Two circuits with identical head sets but different edge patterns are mechanistically different. EAP is the current standard at DeepMind and Anthropic for circuit-level analysis.

**Mathematical formulation (Syed et al., 2023, arXiv:2310.10348):**

For each directed edge e = (h_src → h_dst, role) where role ∈ {Q, K, V}:

```
α_EAP(e) ≈ (∂LD / ∂z_{h_dst}^{role})|_{z = z^clean}  ·  (z_{h_src}^{clean} − z_{h_src}^{corrupt})
```

This decomposes the head-level attribution into:

```
α(h_dst) = Σ_{h_src, role} α_EAP(h_src → h_dst, role)
```

The edge graph is a DAG over (layer, head, role) tuples with O(n_layers × n_heads × 3) edges.

**What to implement:**

```python
class EdgeAttributionPatcher:
    """
    Computes edge-level attribution scores.
    2 forward passes + 1 backward pass = same cost as head-level AtP.

    Output: sparse edge graph as adjacency dict
        {(src_head, dst_head, role): attribution_score}
    """

    def __call__(
        self,
        clean_tokens: Tensor,
        corrupt_tokens: Tensor,
        target_token: int,
        distractor_token: int,
        threshold: float = 0.05,
    ) -> EdgeAttributionResult:

        # Step 1: clean forward pass, save all head outputs z_h^clean
        # Step 2: corrupt forward pass, save all head outputs z_h^corrupt
        # Step 3: backward pass on clean run, accumulate ∂LD/∂z_h for all roles
        # Step 4: α_EAP(e) = grad · (z_src^clean − z_src^corrupt)
        # Step 5: threshold at top 5%, build sparse DAG
        ...
```

**Visualisation:** Export as DOT graph (Graphviz) or interactive D3 DAG.

**References:** Syed, Rager, Conmy (2023) arXiv:2310.10348; Kramár et al. (2024) arXiv:2403.00745 (AtP*)

---

### Feature 9: Approximation Error Bounds (Hessian-Vector Products)

**Why it matters.** The Taylor approximation error for head h is:

```
ε(h) = (1/2) · δz_h^T · H_h · δz_h + O(||δz_h||³)

where:
  δz_h = z_h^clean − z_h^corrupt    [activation difference]
  H_h = ∂²LD / ∂z_h²               [Hessian of logit diff w.r.t. head output]
```

We don't need the full Hessian (d × d matrix, expensive). We just need the quadratic form `δz_h^T · H_h · δz_h`, which can be computed via a Hessian-vector product in one additional backward pass:

```
H_h · v = ∂/∂z_h [(∂LD/∂z_h)^T · v]
```

Setting v = δz_h gives us exactly the error bound.

**What to implement:**

```python
def attribution_error_bounds(
    self,
    clean_tokens: Tensor,
    corrupt_tokens: Tensor,
    target_token: int,
    distractor_token: int,
) -> Dict[Tuple[int,int], float]:
    """
    Returns second-order error bound ε(h) for each head.

    Implementation:
    1. Standard attribution patching → α(h), gradients g_h
    2. For each head h: compute Hv = torch.autograd.functional.vhp(
           loss_fn, z_h, v=delta_z_h
       )[1]
    3. ε(h) = 0.5 * (delta_z_h · Hv)
    4. Relative error = |ε(h)| / |α(h)|
    """
```

**Flag heads where `|ε(h)| / |α(h)| > 0.20`** (20% relative error) as `approximation_unreliable`. These heads need exact activation patching to confirm.

**Computational cost:** 1 extra backward pass per head flagged → negligible if only a small fraction are flagged.

**References:** Nocedal & Wright (2006) "Numerical Optimization" §6.1; Pearlmutter (1994) "Fast Exact Multiplication by the Hessian"; Kramár et al. (2024) §3.2

---

### Feature 10: Causal Scrubbing

**Why it matters.** Attribution patching tells you which heads are important. Causal scrubbing (Chan et al., 2022) tells you *whether the circuit is implementing the hypothesis you think it is*. You specify a causal hypothesis (e.g., "L9H9 copies the IO token because it attends to the IO position") and causal scrubbing tests whether resampling the activations consistent with that hypothesis recovers model performance.

This is what Anthropic uses to validate circuits beyond just "this head is important."

**Mathematical formulation:**

Given a causal hypothesis H as a causal graph over model components:

```
P_H = P(activations | hypothesis H is the computational mechanism)
```

Causal scrubbing metric:

```
CS(H) = E_{x ~ D}[LD(x; do(activations ~ P_H))] / LD_clean
```

A perfect hypothesis: CS(H) = 1.0. Random baseline: CS(random) ≈ 0.0.

**References:** Chan, Garriga-Alonso, Goldowsky-Dill, et al. (2022) "Causal Scrubbing: a method for rigorously testing interpretability hypotheses" AI Alignment Forum

---

### Feature 11: Distributed Alignment Search (DAS)

**Why it matters.** Standard circuit analysis assumes each computational role (e.g., "track IO name") maps to a single attention head. DAS (Geiger et al., 2023, arXiv:2303.02536) tests whether a concept is represented *distributedly* across multiple heads or neurons. This is critical for larger models where concepts are rarely localised to single components.

**What to implement:**

```python
class DistributedAlignmentSearch:
    """
    Finds the linear subspace of the residual stream that encodes
    a specified concept, allowing it to be distributed across heads.

    Algorithm (Geiger et al., 2023):
    1. Define counterfactual pairs (x_base, x_source) that differ in concept C
    2. Learn rotation R such that:
         R · h_base[concept_dims] = R · h_source[concept_dims]
       preserves model behaviour (interchange intervention)
    3. The concept subspace = span of concept_dims in the rotated basis

    Output: R (rotation matrix), concept_dims (which dimensions encode C),
            DAS_score (how well the aligned subspace explains the behaviour)
    """
```

**References:** Geiger et al. (2023) arXiv:2303.02536; Wu et al. (2023) "Interpretability at Scale" arXiv:2305.08809

---

### v4.1.0 API Surface

```python
# Edge-level circuit
edge_result = gb.edge_attribution_patching(
    clean_tokens, corrupt_tokens, target_token, distractor_token
)
edge_result.to_dot()          # export as Graphviz DAG
edge_result.top_edges(k=20)   # top 20 edges

# Error bounds
bounds = gb.attribution_error_bounds(clean_tokens, corrupt_tokens, ...)
bounds["L9H9"]  # {"alpha": 3.41, "epsilon": 0.12, "relative_error": 0.035}

# Causal scrubbing
hypothesis = gb.CircuitHypothesis(
    components=[(9,9), (9,6), (10,0)],
    mechanism="name_mover_copies_io_to_output",
)
cs_score = gb.causal_scrubbing(prompts, hypothesis)  # 0.0–1.0

# DAS
das = gb.distributed_alignment_search(
    prompts, concept="indirect_object_position", layer=9
)
das.concept_dims        # which residual stream dimensions encode concept
das.das_score           # alignment quality
das.visualise()         # PCA plot of concept subspace
```

---

## Summary Table

| Feature | Version | Effort | Lab Standard | Paper |
|---------|---------|--------|--------------|-------|
| Multiple corruptions | v3.7.0 | Easy | Wang et al. 2022 | arXiv:2211.00593 |
| Held-out validation | v3.7.0 | Easy | Standard ML | Bishop (2006) |
| Sample size gate | v3.7.0 | Easy | Cohen (1988) | MATH_FOUNDATIONS.md §10 |
| Folded LayerNorm | v4.0.0 | Medium | Anthropic circuits | transformer-circuits.pub 2021 |
| SAE feature attribution | v4.0.0 | Medium | Anthropic (2023–24) | arXiv:2406.04093 |
| GPU n=100 pipeline | v4.0.0 | Medium | All major labs | — |
| FDR control (BH) | v4.0.0 | Easy/Medium | Statistics standard | BH (1995) JRSS-B |
| Edge attribution (EAP) | v4.1.0 | Hard | DeepMind / Anthropic | arXiv:2310.10348 |
| Hessian error bounds | v4.1.0 | Hard | DeepMind AtP* | arXiv:2403.00745 |
| Causal scrubbing | v4.1.0 | Hard | Anthropic (2022) | AI Alignment Forum |
| Distributed alignment (DAS) | v4.1.0 | Hard | Stanford (2023) | arXiv:2303.02536 |

---

## Mathematical Completeness Scorecard

After v4.1.0 ships, Glassbox will implement or integrate:

| Mathematical Framework | Paper | Status after v4.1.0 |
|------------------------|-------|---------------------|
| First-order Taylor attribution | Nanda (2023) | ✅ Current |
| Second-order error bounds | Pearlmutter (1994) | ✅ v4.1.0 |
| Fisher Z / Pearson correlation test | Fisher (1915) | ✅ Current |
| Bootstrap BCa CIs | Efron & Tibshirani (1993) | ✅ Current |
| Bonferroni FWER correction | Bland & Altman (1995) | ✅ Current |
| Benjamini-Hochberg FDR | BH (1995) | ✅ v4.0.0 |
| Welch's t-test (unequal variance) | Welch (1947) | ✅ Current |
| Cohen's d effect size | Cohen (1988) | ✅ Current |
| Sufficiency / Comprehensiveness | Conmy et al. (2023) | ✅ Current |
| Power analysis (n_min formula) | Cohen (1988) | ✅ Current |
| LayerNorm folding correction | Elhage et al. (2021) | ✅ v4.0.0 |
| SAE sparse feature decomposition | Bricken et al. (2023) | ✅ v4.0.0 |
| Edge attribution patching | Syed et al. (2023) | ✅ v4.1.0 |
| AtP* error-bounded attribution | Kramár et al. (2024) | ✅ v4.1.0 |
| Causal scrubbing | Chan et al. (2022) | ✅ v4.1.0 |
| Distributed alignment search | Geiger et al. (2023) | ✅ v4.1.0 |
| Polysemanticity / superposition | Elhage et al. (2022) | ✅ v4.0.0 |
| EU AI Act Annex IV compliance | Regulation (EU) 2024/1689 | ✅ Current |

**After v4.1.0: Glassbox covers 18 distinct mathematical frameworks, cited to the exact paper that introduced each one.**

---

## EU AI Act Alignment (v4.0.0+)

Each new feature maps to a specific EU AI Act article:

| Feature | Article | Evidence provided |
|---------|---------|-------------------|
| Multi-corruption robustness | Art.15(1) — Robustness | Stability across perturbation types |
| Held-out validation | Art.9(5) — Risk management | Generalisation, not overfitting |
| LayerNorm correction | Art.13(1) — Transparency | Bias-corrected, honest attributions |
| SAE features | Art.13(1) — Transparency | Human-interpretable feature names |
| Error bounds | Art.15(1) — Robustness | Quantified approximation reliability |
| EAP edge graph | Art.13(2) — Explainability | Causal computational graph |
| FDR control | Art.9(7) — Accuracy | Controlled false discovery rate |
| DAS | Art.13(1) — Transparency | Distributed representation evidence |

---

## References

1. **Wang et al.** (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. arXiv:2211.00593.
2. **Conmy et al.** (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS. arXiv:2304.14997.
3. **Nanda, N.** (2023). Attribution Patching: Activation Patching At Industrial Scale. neelnanda.io.
4. **Kramár et al.** (2024). AtP*: An efficient and scalable method for localizing LLM behaviour to components. arXiv:2403.00745. *DeepMind.*
5. **Syed, Rager, Conmy** (2023). Attribution Patching Outperforms Automated Circuit Discovery. ACL BlackboxNLP. arXiv:2310.10348. *DeepMind.*
6. **Bricken, Templeton et al.** (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. transformer-circuits.pub. *Anthropic.*
7. **Templeton et al.** (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. transformer-circuits.pub. *Anthropic.*
8. **Gao et al.** (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093. *OpenAI/DeepMind.*
9. **Samuel et al.** (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
10. **Elhage et al.** (2021). A Mathematical Framework for Transformer Circuits. transformer-circuits.pub. *Anthropic.*
11. **Elhage et al.** (2022). Toy Models of Superposition. arXiv:2209.10652. *Anthropic.*
12. **Geiger et al.** (2023). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. arXiv:2303.02536. *Stanford.*
13. **Wu et al.** (2023). Interpretability at Scale: Identifying Causal Mechanisms in Alpaca. arXiv:2305.08809. *Stanford.*
14. **Chan et al.** (2022). Causal Scrubbing: a method for rigorously testing interpretability hypotheses. AI Alignment Forum. *Anthropic.*
15. **Zou et al.** (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405. *UC Berkeley/CAIS.*
16. **Belinkov** (2022). Probing Classifiers: Promises, Shortcomings, and Advances. arXiv:2102.12452.
17. **Geiger et al.** (2023). Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability. arXiv:2301.04709. *Stanford.*
18. **Benjamini & Hochberg** (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. JRSS-B 57(1):289–300.
19. **Fisher, R.A.** (1915). Frequency distribution of the values of the correlation coefficient. Biometrika 10(4):507–521.
20. **Cohen, J.** (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum.
21. **Efron & Tibshirani** (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
22. **Pearlmutter, B.A.** (1994). Fast Exact Multiplication by the Hessian. Neural Computation 6(1):147–160.
23. **Regulation (EU) 2024/1689** (AI Act). EUR-Lex CELEX:32024R1689.
24. **Mahale, A.** (2026). Glassbox: Mechanistic Interpretability and EU AI Act Compliance. arXiv:2603.09988.

---

*Document version: 1.0.0*
*Glassbox v4.0 Roadmap · April 2026*
*Mathematical foundations: MATH_FOUNDATIONS.md*
*Current implementation: experiments/cross_model_study.py*
