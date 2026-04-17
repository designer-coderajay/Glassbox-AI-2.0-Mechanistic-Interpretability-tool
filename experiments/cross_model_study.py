"""
Cross-Model Faithfulness Study
================================
Glassbox v4.2.6 — Multi-LLM Evaluation Harness
Author: Ajay Pravin Mahale <mahale.ajay01@gmail.com>
Paper: "Does Confidence–Faithfulness Independence Generalise Across LLM Families?"

Research question
-----------------
Mahale (2026, arXiv:2603.09988) reports r(confidence, faithfulness) = 0.009 on GPT-2 small
for the IOI task. This study replicates the finding across 4 model families:

    M1: GPT-2 small    (117M params)  — baseline, already published
    M2: GPT-2 XL       (1.5B params)  — scale control, same architecture
    M3: Pythia-1.4B    (1.4B params)  — different architecture/training data
    M4: Llama-2-7B     (7B params)    — modern instruction-tuned, RLHF

Hypotheses (see MATH_FOUNDATIONS.md §7, §14 for full statistical framework)
---------------------------------------------------------------------------
H0: r(confidence, faithfulness) = 0   for each model
H1: |r| ≥ 0.30 (small-medium effect detectable with n=100, power=0.86)

H_cross: All four r values are drawn from the same distribution (no model-family effect)
H_circuit: Circuit heads occupy equivalent relative positions across models (J ≥ 0.50)

Statistical framework
---------------------
- n = 100 IOI prompts per model (80% power to detect |r| ≥ 0.28, see §14.2)
- Fisher Z transformation for CI construction (§7.2)
- Welch's t-test for cross-model mean comparison (§8.1)
- Bonferroni-corrected α = 0.05/6 = 0.0083 for 6 pairwise tests (§8.3)
- Jaccard similarity for circuit positional overlap (§9.2)
- Bootstrap B=2000 for all CI estimates (§6.1)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cross_model_study")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODELS: Dict[str, Dict] = {
    "gpt2-small": {
        "hf_name":    "gpt2",
        "params_M":   117,
        "n_layers":   12,
        "n_heads":    12,
        "d_model":    768,
        "family":     "GPT-2",
        "training":   "CLM on WebText",
        "published":  True,   # baseline: r=0.009 from arXiv:2603.09988
        "published_r": 0.009,
        "tl_supported": True,  # TransformerLens native support
    },
    "gpt2-xl": {
        "hf_name":    "gpt2-xl",
        "params_M":   1542,
        "n_layers":   48,
        "n_heads":    25,
        "d_model":    1600,
        "family":     "GPT-2",
        "training":   "CLM on WebText",
        "published":  False,
        "tl_supported": True,
    },
    "pythia-1.4b": {
        "hf_name":    "EleutherAI/pythia-1.4b",
        "params_M":   1400,
        "n_layers":   24,
        "n_heads":    16,
        "d_model":    2048,
        "family":     "Pythia",
        "training":   "CLM on The Pile",
        "published":  False,
        "tl_supported": True,
    },
    "llama-2-7b": {
        "hf_name":    "meta-llama/Llama-2-7b-hf",
        "params_M":   7000,
        "n_layers":   32,
        "n_heads":    32,
        "d_model":    4096,
        "family":     "Llama",
        "training":   "CLM + RLHF on diverse corpus",
        "published":  False,
        "tl_supported": True,   # Added in TransformerLens ≥2.0
        "notes":      "Requires HF_TOKEN for download. GQA — TL handles correctly.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. IOI PROMPT SUITE
# ─────────────────────────────────────────────────────────────────────────────

# Name pairs: (subject_1, subject_2). S1 is the correct answer (indirect object).
NAME_PAIRS: List[Tuple[str, str]] = [
    ("Mary",    "John"),
    ("Alice",   "Bob"),
    ("Sarah",   "Tom"),
    ("Emma",    "James"),
    ("Olivia",  "William"),
    ("Sophie",  "Michael"),
    ("Grace",   "David"),
    ("Hannah",  "Daniel"),
    ("Laura",   "Matthew"),
    ("Claire",  "Andrew"),
    ("Rachel",  "Christopher"),
    ("Julia",   "Alexander"),
    ("Anna",    "Edward"),
    ("Kate",    "George"),
    ("Lucy",    "Henry"),
    ("Mia",     "Oliver"),
    ("Ella",    "Harry"),
    ("Lily",    "Jack"),
    ("Charlotte","Noah"),
    ("Amelia",  "Liam"),
]

# Verb frames for prompt variation (reduces task-specific overfitting)
VERB_FRAMES: List[str] = [
    "went to the store, {S2} gave a drink to",
    "visited the market, {S2} handed a gift to",
    "attended the party, {S2} offered flowers to",
    "arrived at school, {S2} showed a book to",
    "came to dinner, {S2} passed the salt to",
]


def build_ioi_prompts(
    name_pairs: List[Tuple[str, str]] = NAME_PAIRS,
    verb_frames: List[str] = VERB_FRAMES,
    n_prompts: int = 100,
) -> List[Dict]:
    """
    Build n_prompts IOI prompt triplets.

    Each prompt contains:
        prompt    : "When {S1} and {S2} {verb_frame} {S2} ..."  (truncated)
        correct   : " {S1}"   (the indirect object / name NOT repeated)
        incorrect : " {S2}"   (the subject who gave — repeated)

    The IOI task: predict the name that was NOT the actor in the second clause.
    Wang et al. (2022) establish this as the canonical circuit benchmark.

    Returns list of dicts: {prompt, correct, incorrect, s1, s2, frame_idx}
    """
    prompts = []
    idx = 0
    pair_cycle = 0

    while len(prompts) < n_prompts:
        s1, s2 = name_pairs[pair_cycle % len(name_pairs)]
        frame = verb_frames[idx % len(verb_frames)]
        filled = frame.format(S2=s2)
        prompt_text = f"When {s1} and {s2} {filled}"

        prompts.append({
            "prompt":    prompt_text,
            "correct":   f" {s1}",
            "incorrect": f" {s2}",
            "s1": s1,
            "s2": s2,
            "frame_idx": idx % len(verb_frames),
        })
        idx += 1
        if idx % len(verb_frames) == 0:
            pair_cycle += 1

    return prompts[:n_prompts]


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESULT DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    """Single-prompt result for one model."""
    prompt: str
    correct: str
    incorrect: str
    confidence: float           # softmax P(correct_token)
    logit_diff: float           # logit(correct) - logit(incorrect)
    sufficiency: float          # S(C) — see MATH_FOUNDATIONS §4.1
    comprehensiveness: float    # Comp(C) — see MATH_FOUNDATIONS §4.2
    f1: float                   # F1_faith — see MATH_FOUNDATIONS §4.3
    top_heads: List[Tuple]      # [(layer, head, score), ...]
    n_circuit_heads: int        # |C*| — minimum faithful circuit size
    analysis_time_s: float


@dataclass
class ModelResult:
    """Aggregated results for one model across all prompts."""
    model_key: str
    model_info: Dict
    prompt_results: List[PromptResult] = field(default_factory=list)

    # Aggregate statistics (filled by .aggregate())
    r_confidence_f1: float = 0.0         # Pearson r(confidence, F1)
    r_fisher_z: float = 0.0              # atanh(r) — Fisher Z
    r_ci_lo: float = 0.0                 # 95% CI lower bound (Fisher Z back-transformed)
    r_ci_hi: float = 0.0                 # 95% CI upper bound
    r_pvalue: float = 0.0                # two-tailed p for H0: r=0

    mean_sufficiency: float = 0.0
    mean_comprehensiveness: float = 0.0
    mean_f1: float = 0.0
    std_f1: float = 0.0

    suff_ci_lo: float = 0.0
    suff_ci_hi: float = 0.0
    comp_ci_lo: float = 0.0
    comp_ci_hi: float = 0.0
    f1_ci_lo: float = 0.0
    f1_ci_hi: float = 0.0

    top_circuit_heads: List[Tuple] = field(default_factory=list)  # most-frequent across prompts
    mean_circuit_size: float = 0.0
    total_time_s: float = 0.0

    def aggregate(self, n_boot: int = 2000, alpha: float = 0.05) -> None:
        """Compute all aggregate statistics. Call after all prompt_results are added."""
        if not self.prompt_results:
            raise ValueError("No prompt results to aggregate.")

        conf  = np.array([p.confidence for p in self.prompt_results])
        suff  = np.array([p.sufficiency for p in self.prompt_results])
        comp  = np.array([p.comprehensiveness for p in self.prompt_results])
        f1    = np.array([p.f1 for p in self.prompt_results])
        n = len(f1)

        # ── Pearson r(confidence, F1) ──────────────────────────────────────
        # Formula: MATH_FOUNDATIONS §7.1
        r, pval = stats.pearsonr(conf, f1)
        self.r_confidence_f1 = float(r)
        self.r_pvalue = float(pval)

        # Fisher Z transformation: MATH_FOUNDATIONS §7.2
        z = np.arctanh(r)
        se_z = 1.0 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        self.r_fisher_z = float(z)
        self.r_ci_lo = float(np.tanh(z - z_crit * se_z))
        self.r_ci_hi = float(np.tanh(z + z_crit * se_z))

        # ── Means ─────────────────────────────────────────────────────────
        self.mean_sufficiency      = float(np.mean(suff))
        self.mean_comprehensiveness = float(np.mean(comp))
        self.mean_f1               = float(np.mean(f1))
        self.std_f1                = float(np.std(f1, ddof=1))

        # ── Bootstrap CIs (percentile method) — MATH_FOUNDATIONS §6.1 ─────
        def boot_ci(arr: np.ndarray) -> Tuple[float, float]:
            boot_means = np.array([
                np.mean(np.random.choice(arr, size=len(arr), replace=True))
                for _ in range(n_boot)
            ])
            return (
                float(np.percentile(boot_means, 100 * alpha / 2)),
                float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
            )

        self.suff_ci_lo, self.suff_ci_hi = boot_ci(suff)
        self.comp_ci_lo, self.comp_ci_hi = boot_ci(comp)
        self.f1_ci_lo,   self.f1_ci_hi   = boot_ci(f1)

        # ── Circuit statistics ─────────────────────────────────────────────
        self.mean_circuit_size = float(np.mean([p.n_circuit_heads for p in self.prompt_results]))
        self.total_time_s      = sum(p.analysis_time_s for p in self.prompt_results)

        # Most frequent top heads across prompts
        from collections import Counter
        head_counts: Counter = Counter()
        for p in self.prompt_results:
            for layer, head, _ in p.top_heads[:3]:
                head_counts[(layer, head)] += 1
        self.top_circuit_heads = [
            (l, h, cnt / n) for (l, h), cnt in head_counts.most_common(5)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class CrossModelStatistics:
    """
    Between-model statistical comparisons.
    See MATH_FOUNDATIONS §8, §9.
    """

    def __init__(self, results: Dict[str, ModelResult]) -> None:
        self.results = results
        self.model_keys = list(results.keys())

    def welch_ttest_f1(self, m1: str, m2: str) -> Dict:
        """
        Welch's t-test for H0: mean_F1(M1) = mean_F1(M2).
        Formula: MATH_FOUNDATIONS §8.1
        """
        f1_m1 = np.array([p.f1 for p in self.results[m1].prompt_results])
        f1_m2 = np.array([p.f1 for p in self.results[m2].prompt_results])
        t, p  = stats.ttest_ind(f1_m1, f1_m2, equal_var=False)  # Welch

        n1, n2  = len(f1_m1), len(f1_m2)
        s1, s2  = np.std(f1_m1, ddof=1), np.std(f1_m2, ddof=1)
        s_pool  = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        cohens_d = (np.mean(f1_m1) - np.mean(f1_m2)) / s_pool

        return {
            "m1": m1, "m2": m2,
            "t_statistic": float(t),
            "p_value_raw": float(p),
            "p_value_bonferroni": float(min(p * 6, 1.0)),  # 6 pairwise comparisons
            "cohens_d": float(cohens_d),
            "significant_bonferroni": float(p) < (0.05 / 6),
        }

    def fisher_z_comparison(self, m1: str, m2: str) -> Dict:
        """
        Compare r(confidence, F1) between two models.
        Formula: MATH_FOUNDATIONS §8.4
        H0: r_M1 = r_M2
        """
        r1 = self.results[m1].r_confidence_f1
        r2 = self.results[m2].r_confidence_f1
        n1 = len(self.results[m1].prompt_results)
        n2 = len(self.results[m2].prompt_results)

        z1, z2 = np.arctanh(r1), np.arctanh(r2)
        se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
        z_diff  = (z1 - z2) / se_diff
        p_diff  = 2 * stats.norm.sf(abs(z_diff))

        return {
            "m1": m1, "r1": r1, "z1": float(z1),
            "m2": m2, "r2": r2, "z2": float(z2),
            "z_diff": float(z_diff),
            "p_value": float(p_diff),
            "significant": float(p_diff) < 0.05,
            "interpretation": "r values differ significantly" if p_diff < 0.05
                              else "r values are statistically indistinguishable",
        }

    def jaccard_circuit_similarity(self, m1: str, m2: str, epsilon: float = 0.05) -> float:
        """
        Normalised Jaccard similarity between top-5 circuit heads.
        Formula: MATH_FOUNDATIONS §9.2

        Heads are mapped to normalised position [0,1]x[0,1] before comparison.
        """
        info1 = self.results[m1].model_info
        info2 = self.results[m2].model_info
        L1, H1 = info1["n_layers"] - 1, info1["n_heads"] - 1
        L2, H2 = info2["n_layers"] - 1, info2["n_heads"] - 1

        def normalise(heads, L, H):
            return {(l / L, h / H) for l, h, _ in heads}

        set1 = normalise(self.results[m1].top_circuit_heads, L1, H1)
        set2 = normalise(self.results[m2].top_circuit_heads, L2, H2)

        # ε-approximate intersection
        intersection = sum(
            1 for p1 in set1
            if any(abs(p1[0]-p2[0]) <= epsilon and abs(p1[1]-p2[1]) <= epsilon
                   for p2 in set2)
        )
        union = len(set1) + len(set2) - intersection
        return float(intersection / union) if union > 0 else 0.0

    def all_pairwise(self) -> Dict:
        """Run all 6 pairwise comparisons across the 4 models."""
        keys = self.model_keys
        pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i+1, len(keys))]
        return {
            "welch_ttest":    [self.welch_ttest_f1(a, b) for a, b in pairs],
            "fisher_z_diff":  [self.fisher_z_comparison(a, b) for a, b in pairs],
            "jaccard":        {f"{a}_vs_{b}": self.jaccard_circuit_similarity(a, b) for a, b in pairs},
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_model(
    model_key: str,
    prompts: List[Dict],
    device: str = "cpu",
    dry_run: bool = False,
) -> ModelResult:
    """
    Run the full Glassbox analysis for one model across all prompts.

    Parameters
    ----------
    model_key : one of MODELS.keys()
    prompts   : list of {prompt, correct, incorrect} dicts from build_ioi_prompts()
    device    : "cpu" | "cuda" | "mps"
    dry_run   : if True, generate synthetic results for testing pipeline logic

    Returns
    -------
    ModelResult with prompt_results populated and aggregate() called.
    """
    info = MODELS[model_key]
    result = ModelResult(model_key=model_key, model_info=info)

    if dry_run:
        log.info(f"[DRY RUN] Generating synthetic results for {model_key}")
        rng = np.random.default_rng(seed=hash(model_key) % (2**32))
        # Synthetic: r ≈ 0 (null hypothesis true) with model-family noise
        base_f1 = {"gpt2-small": 0.64, "gpt2-xl": 0.66, "pythia-1.4b": 0.59, "llama-2-7b": 0.71}
        for p_dict in prompts:
            conf = float(rng.uniform(0.4, 0.95))
            f1   = float(np.clip(rng.normal(base_f1.get(model_key, 0.65), 0.08), 0.0, 1.0))
            suff = float(np.clip(rng.normal(0.85, 0.10), 0.0, 1.0))
            comp = float(np.clip(rng.normal(0.50, 0.12), 0.0, 1.0))
            result.prompt_results.append(PromptResult(
                prompt=p_dict["prompt"], correct=p_dict["correct"], incorrect=p_dict["incorrect"],
                confidence=conf, logit_diff=float(rng.normal(2.5, 0.8)),
                sufficiency=suff, comprehensiveness=comp, f1=f1,
                top_heads=[(9, 6, 0.58), (9, 9, 0.43), (10, 0, 0.31)],
                n_circuit_heads=3, analysis_time_s=float(rng.uniform(1.0, 1.5)),
            ))
        result.aggregate()
        return result

    # ── Real run ────────────────────────────────────────────────────────────
    try:
        import glassbox
        import torch
    except ImportError as e:
        raise RuntimeError(f"glassbox and torch required: {e}")

    log.info(f"Loading model: {info['hf_name']} on {device}")
    gb = glassbox.GlassboxV2(
        model_name=info["hf_name"],
        device=device,
    )

    for i, p_dict in enumerate(prompts):
        log.info(f"  [{model_key}] Prompt {i+1}/{len(prompts)}: {p_dict['prompt'][:60]}...")
        t0 = time.time()

        try:
            analysis = gb.analyze(
                prompt=p_dict["prompt"],
                correct=p_dict["correct"],
                incorrect=p_dict["incorrect"],
            )

            # Confidence = softmax P(correct_token)
            confidence = float(torch.softmax(
                torch.tensor([analysis.logit_correct, analysis.logit_incorrect]), dim=0
            )[0])

            faith = analysis.faithfulness
            suff  = faith.get("sufficiency", 0.0)
            comp  = faith.get("comprehensiveness", 0.0)
            f1_val = 2*suff*comp/(suff+comp) if (suff+comp) > 0 else 0.0

            top_heads = [
                (l, h, s)
                for (l, h), s in sorted(analysis.attributions.items(), key=lambda x: -abs(x[1]))[:5]
            ]

            result.prompt_results.append(PromptResult(
                prompt=p_dict["prompt"],
                correct=p_dict["correct"],
                incorrect=p_dict["incorrect"],
                confidence=confidence,
                logit_diff=float(analysis.logit_diff),
                sufficiency=float(suff),
                comprehensiveness=float(comp),
                f1=float(f1_val),
                top_heads=top_heads,
                n_circuit_heads=len(analysis.circuit),
                analysis_time_s=time.time() - t0,
            ))

        except Exception as e:
            log.warning(f"  Prompt {i+1} failed: {e}")
            continue

    if not result.prompt_results:
        raise RuntimeError(f"All prompts failed for {model_key}")

    result.aggregate()
    log.info(
        f"  [{model_key}] Done. "
        f"r={result.r_confidence_f1:.3f} "
        f"F1={result.mean_f1:.3f} "
        f"n={len(result.prompt_results)}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    model_results: Dict[str, ModelResult],
    pairwise: Dict,
    output_dir: Path,
) -> Path:
    """
    Generate JSON results + Markdown summary table.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON dump ────────────────────────────────────────────────────────────
    report = {
        "study": "Cross-Model Confidence–Faithfulness Study",
        "version": "4.2.6",
        "date": time.strftime("%Y-%m-%d"),
        "hypothesis": "H0: r(confidence, F1) = 0 for each model family",
        "n_prompts_per_model": 100,
        "bootstrap_n": 2000,
        "alpha": 0.05,
        "models": {},
        "pairwise_comparisons": pairwise,
    }

    for key, mr in model_results.items():
        report["models"][key] = {
            "model_info":           mr.model_info,
            "n_prompts_analyzed":   len(mr.prompt_results),
            "r_confidence_f1":      mr.r_confidence_f1,
            "r_pvalue":             mr.r_pvalue,
            "r_95ci":               [mr.r_ci_lo, mr.r_ci_hi],
            "r_fisher_z":           mr.r_fisher_z,
            "h0_rejected":          mr.r_pvalue < 0.05,
            "mean_sufficiency":     mr.mean_sufficiency,
            "mean_comprehensiveness": mr.mean_comprehensiveness,
            "mean_f1":              mr.mean_f1,
            "std_f1":               mr.std_f1,
            "f1_95ci":              [mr.f1_ci_lo, mr.f1_ci_hi],
            "suff_95ci":            [mr.suff_ci_lo, mr.suff_ci_hi],
            "comp_95ci":            [mr.comp_ci_lo, mr.comp_ci_hi],
            "mean_circuit_size":    mr.mean_circuit_size,
            "top_circuit_heads":    mr.top_circuit_heads,
            "total_time_s":         mr.total_time_s,
        }

    json_path = output_dir / "cross_model_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Results saved: {json_path}")

    # ── Markdown table ────────────────────────────────────────────────────────
    md_lines = [
        "# Cross-Model Faithfulness Study — Results",
        "",
        "**Hypothesis:** H₀: r(confidence, faithfulness) = 0 for each model family.",
        "**Method:** Glassbox v4.2.6 attribution patching on IOI task, n=100 prompts per model.",
        "**Statistics:** Pearson r, Fisher Z 95% CI, Welch's t-test (Bonferroni-corrected, α=0.0083).",
        "",
        "## Table 1: Confidence–Faithfulness Correlation by Model",
        "",
        "| Model | Params | r(conf,F1) | 95% CI | p-value | H₀ rejected? |",
        "|-------|--------|-----------|--------|---------|-------------|",
    ]
    for key, mr in model_results.items():
        info = mr.model_info
        ci = f"[{mr.r_ci_lo:.3f}, {mr.r_ci_hi:.3f}]"
        rejected = "✅ Yes" if mr.r_pvalue < 0.05 else "❌ No"
        md_lines.append(
            f"| {key} | {info['params_M']}M | "
            f"{mr.r_confidence_f1:.3f} | {ci} | "
            f"{mr.r_pvalue:.3f} | {rejected} |"
        )

    md_lines += [
        "",
        "## Table 2: Faithfulness Metrics by Model",
        "",
        "| Model | S (95% CI) | Comp (95% CI) | F1 (95% CI) | Circuit |",
        "|-------|-----------|--------------|------------|---------|",
    ]
    for key, mr in model_results.items():
        md_lines.append(
            f"| {key} | "
            f"{mr.mean_sufficiency:.3f} [{mr.suff_ci_lo:.3f},{mr.suff_ci_hi:.3f}] | "
            f"{mr.mean_comprehensiveness:.3f} [{mr.comp_ci_lo:.3f},{mr.comp_ci_hi:.3f}] | "
            f"{mr.mean_f1:.3f} [{mr.f1_ci_lo:.3f},{mr.f1_ci_hi:.3f}] | "
            f"{mr.mean_circuit_size:.1f} heads |"
        )

    md_lines += [
        "",
        "## Table 3: Circuit Positional Overlap (Jaccard Similarity)",
        "",
        "| Pair | J | Interpretation |",
        "|------|---|---------------|",
    ]
    for pair_key, j in pairwise.get("jaccard", {}).items():
        interp = "High overlap" if j > 0.5 else ("Moderate" if j > 0.25 else "Low overlap")
        md_lines.append(f"| {pair_key} | {j:.3f} | {interp} |")

    md_lines += [
        "",
        "## Interpretation",
        "",
        "- **H₀ not rejected** in all models → confidence–faithfulness independence generalises",
        "- **H₀ rejected** in any model → model family or training regime breaks the pattern",
        "",
        f"*Generated by Glassbox v4.2.6 — {time.strftime('%Y-%m-%d')}*",
        "*Mathematical framework: MATH_FOUNDATIONS.md (arXiv:2603.09988)*",
    ]

    md_path = output_dir / "cross_model_results.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log.info(f"Markdown report: {md_path}")

    return json_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main(
    models: Optional[List[str]] = None,
    n_prompts: int = 100,
    device: str = "cpu",
    output_dir: str = "experiments/results",
    dry_run: bool = False,
) -> None:
    """
    Run the full cross-model study.

    Parameters
    ----------
    models    : list of keys from MODELS dict. None = all 4 models.
    n_prompts : prompts per model (default 100; provides 86% power, §14.2)
    device    : "cpu" | "cuda" | "mps"
    output_dir: where to write results
    dry_run   : generate synthetic data (for pipeline testing)
    """
    if models is None:
        models = list(MODELS.keys())

    log.info("=" * 60)
    log.info("Glassbox Cross-Model Faithfulness Study")
    log.info(f"Models: {models}")
    log.info(f"Prompts per model: {n_prompts}")
    log.info(f"Device: {device}  |  Dry run: {dry_run}")
    log.info("=" * 60)

    prompts = build_ioi_prompts(n_prompts=n_prompts)
    log.info(f"Built {len(prompts)} IOI prompts")

    model_results: Dict[str, ModelResult] = {}
    for mkey in models:
        log.info(f"\n{'─'*40}")
        log.info(f"Running: {mkey}  ({MODELS[mkey]['params_M']}M params)")
        log.info(f"{'─'*40}")
        try:
            model_results[mkey] = run_model(mkey, prompts, device=device, dry_run=dry_run)
        except Exception as e:
            log.error(f"Model {mkey} failed entirely: {e}")
            continue

    if len(model_results) < 2:
        log.error("Need at least 2 models for comparison. Aborting.")
        return

    stats_engine = CrossModelStatistics(model_results)
    pairwise = stats_engine.all_pairwise()

    out_dir = Path(output_dir)
    generate_report(model_results, pairwise, out_dir)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'r':>8}  {'p':>8}  {'F1':>8}  {'H₀':<12}")
    print("-" * 60)
    for key, mr in model_results.items():
        h0_str = "REJECTED" if mr.r_pvalue < 0.05 else "not rejected"
        print(f"{key:<20} {mr.r_confidence_f1:>8.3f}  {mr.r_pvalue:>8.3f}  {mr.mean_f1:>8.3f}  {h0_str}")
    print("=" * 60)
    print(f"\nFull results: {out_dir}/cross_model_results.json")
    print(f"Summary table: {out_dir}/cross_model_results.md")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Glassbox cross-model confidence–faithfulness study"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list(MODELS.keys()),
        help="Models to test (default: all 4)",
    )
    parser.add_argument("--n-prompts",  type=int, default=100)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Generate synthetic data to test pipeline without model loading")
    args = parser.parse_args()

    main(
        models=args.models,
        n_prompts=args.n_prompts,
        device=args.device,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
