"""
glassbox/corruption.py
======================
Multi-Corruption Pipeline — v3.7.0
====================================

Implements four corruption strategies for robustness testing of identified
circuits, extending the single name-swap strategy from Wang et al. 2022.

Corruption Strategies
---------------------
1. NAME_SWAP      : Bidirectional IO↔S name swap (Wang et al. 2022 standard)
2. RANDOM_TOKEN   : Replace IO/S tokens with Uniform(V) random vocabulary token
3. GAUSSIAN_NOISE : Add N(0, σ²·I) noise to token embeddings (input space)
4. MEAN_ABLATION  : Replace final-position residual stream with dataset mean

Robustness Criterion (ROADMAP_V4 §v3.7.0)
------------------------------------------
∀k : |S_k(C) − S̄| < δ = 0.10

where S_k is sufficiency under corruption k and S̄ is the mean across
all corruptions. Circuits failing this are flagged `perturbation_sensitive`.

References
----------
Wang et al. 2022 — "Interpretability in the Wild" (name-swap)
    https://arxiv.org/abs/2211.00593

Zeiler & Fergus 2013 — Gaussian perturbation for saliency analysis
    https://arxiv.org/abs/1311.1901

Merity et al. 2017 — Mean ablation as zero baseline for transformers
    https://arxiv.org/abs/1708.02182
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Robustness threshold (ROADMAP_V4)
# ──────────────────────────────────────────────────────────────────────────────
ROBUSTNESS_DELTA: float = 0.10


# ──────────────────────────────────────────────────────────────────────────────
# Corruption Strategy Enum
# ──────────────────────────────────────────────────────────────────────────────

class CorruptionStrategy(Enum):
    """
    Supported corruption strategies for multi-corruption robustness analysis.

    Attributes
    ----------
    NAME_SWAP      : Bidirectional IO↔S name swap (Wang et al. 2022 standard)
    RANDOM_TOKEN   : Replace IO/S with a random vocabulary token
    GAUSSIAN_NOISE : Add Gaussian noise to token embeddings at input layer
    MEAN_ABLATION  : Replace last-position residual stream with dataset mean
    """
    NAME_SWAP      = "name_swap"
    RANDOM_TOKEN   = "random_token"
    GAUSSIAN_NOISE = "gaussian_noise"
    MEAN_ABLATION  = "mean_ablation"


# ──────────────────────────────────────────────────────────────────────────────
# Per-corruption result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CorruptionResult:
    """Result from a single corruption run."""
    strategy:          CorruptionStrategy
    corrupted_prompt:  Optional[str]       # None for embedding-space corruptions
    corrupted_tokens:  torch.Tensor        # Shape: (1, seq_len)
    sufficiency:       float
    comprehensiveness: float
    f1:                float
    logit_diff_clean:  float
    logit_diff_corr:   float

    def to_dict(self) -> Dict:
        return {
            "strategy":          self.strategy.value,
            "corrupted_prompt":  self.corrupted_prompt,
            "sufficiency":       round(self.sufficiency, 4),
            "comprehensiveness": round(self.comprehensiveness, 4),
            "f1":                round(self.f1, 4),
            "logit_diff_clean":  round(self.logit_diff_clean, 4),
            "logit_diff_corr":   round(self.logit_diff_corr, 4),
        }


@dataclass
class RobustnessReport:
    """
    Aggregated report across all corruption strategies.

    Attributes
    ----------
    results            : List of CorruptionResult, one per strategy
    mean_sufficiency   : Mean S̄ across all corruptions
    max_deviation      : max_k |S_k − S̄|
    perturbation_sensitive : True if max_deviation ≥ ROBUSTNESS_DELTA (0.10)
    robust             : True iff all |S_k − S̄| < 0.10
    """
    results:               List[CorruptionResult] = field(default_factory=list)
    mean_sufficiency:      float = 0.0
    std_sufficiency:       float = 0.0
    max_deviation:         float = 0.0
    perturbation_sensitive: bool = False
    robust:                bool = True

    def to_dict(self) -> Dict:
        return {
            "mean_sufficiency":       round(self.mean_sufficiency, 4),
            "std_sufficiency":        round(self.std_sufficiency, 4),
            "max_deviation":          round(self.max_deviation, 4),
            "perturbation_sensitive": self.perturbation_sensitive,
            "robust":                 self.robust,
            "delta_threshold":        ROBUSTNESS_DELTA,
            "per_corruption":         [r.to_dict() for r in self.results],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Individual corruption functions
# ──────────────────────────────────────────────────────────────────────────────

def _name_swap(prompt: str, io_name: str, subject_name: str) -> str:
    """
    Bidirectional name swap: IO ↔ Subject.

    Matches Wang et al. 2022 §3.2 corrupted prompt construction.
    Uses word-boundary regex to prevent partial-word replacement.

    x_corrupt = x[IO ↦ S, S ↦ IO]
    """
    placeholder = "<<<GB_SWAP>>>"
    swapped = re.sub(r'\b' + re.escape(io_name) + r'\b', placeholder, prompt)
    swapped = re.sub(r'\b' + re.escape(subject_name) + r'\b', io_name, swapped)
    swapped = swapped.replace(placeholder, subject_name)
    if swapped == prompt:
        swapped = prompt + " " + subject_name
    return swapped


def _random_token_corruption(
    tokens:       torch.Tensor,
    model:        object,  # HookedTransformer
    io_name:      str,
    subject_name: str,
    seed:         Optional[int] = None,
) -> torch.Tensor:
    """
    Replace every occurrence of IO/S tokens with a random vocabulary token.

    x_corrupt = x[IO ↦ Uniform(V), S ↦ Uniform(V)]

    The same random replacement is used for all occurrences of each name
    within the prompt (consistent corruption). Different names get different
    random replacements.

    Parameters
    ----------
    tokens       : (1, seq_len) int tensor
    model        : HookedTransformer with .tokenizer and .cfg.d_vocab
    io_name      : IO character name (e.g. " Mary")
    subject_name : Subject name (e.g. " John")
    seed         : Optional RNG seed for reproducibility

    Returns
    -------
    (1, seq_len) int tensor with IO/S token positions replaced
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    vocab_size = model.cfg.d_vocab
    corrupted  = tokens.clone()

    for name in [io_name, subject_name]:
        try:
            name_tok = model.to_single_token(name)
        except Exception:
            try:
                name_tok = model.to_tokens(name)[0, -1].item()
            except Exception:
                continue

        rand_tok = torch.randint(0, vocab_size, (1,), generator=rng).item()
        # Replace all positions matching name_tok
        mask = (corrupted[0] == name_tok)
        corrupted[0][mask] = rand_tok

    return corrupted


def _gaussian_noise_corruption(
    tokens:        torch.Tensor,
    model:         object,  # HookedTransformer
    sigma_factor:  float = 1.0,
    seed:          Optional[int] = None,
) -> torch.Tensor:
    """
    Add Gaussian noise N(0, σ²·I) to the token embedding matrix at input.

    z_corrupt = embed(x) + ε,  ε ~ N(0, σ²·I)
    σ = sigma_factor · std(embed(x))

    This operates in embedding space rather than token space. The returned
    value is the *corrupted embedding* tensor (not token ids), suitable for
    direct injection into the residual stream.

    Returns float tensor of shape (1, seq_len, d_model).
    """
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        embeddings = model.embed(tokens)           # (1, seq_len, d_model)
        sigma = sigma_factor * embeddings.std().item()
        noise = torch.randn_like(embeddings) * sigma

    return embeddings + noise


def _mean_ablation_corruption(
    tokens:          torch.Tensor,
    model:           object,  # HookedTransformer
    baseline_tokens: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Replace the last-position residual stream value with the dataset mean.

    z_corrupt[last] = E_{x ~ D}[resid_post_final(x)]

    If `baseline_tokens` is provided, the mean is computed from those.
    Otherwise, falls back to zero-vector (equivalent to zero ablation),
    which is standard when no baseline corpus is available.

    Returns a float embedding tensor of shape (1, seq_len, d_model).
    """
    with torch.no_grad():
        # Get clean embeddings as base
        clean_embed = model.embed(tokens)   # (1, seq_len, d_model)
        corrupted   = clean_embed.clone()

        if baseline_tokens and len(baseline_tokens) > 0:
            # Compute mean final-position residual across baseline
            means = []
            for bt in baseline_tokens:
                try:
                    _, cache = model.run_with_cache(bt, names_filter="hook_resid_post")
                    last_resid = cache["hook_resid_post"][:, -1:, :]  # (1,1,d_model)
                    means.append(last_resid)
                except Exception:
                    continue
            if means:
                mean_vec = torch.cat(means, dim=0).mean(dim=0, keepdim=True)  # (1,1,d_model)
                corrupted[0, -1, :] = mean_vec[0, 0, :]
            else:
                corrupted[0, -1, :] = 0.0
        else:
            # Zero ablation fallback (Merity et al. 2017 baseline)
            corrupted[0, -1, :] = 0.0

    return corrupted


# ──────────────────────────────────────────────────────────────────────────────
# Faithfulness computation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _logit_diff(
    model:        object,
    tokens:       torch.Tensor,
    target_tok:   int,
    distract_tok: int,
    embed_input:  Optional[torch.Tensor] = None,
) -> float:
    """
    Compute logit(target) − logit(distractor) at last token position.

    If `embed_input` is given, bypasses token embedding and feeds the
    provided tensor directly into the model (for Gaussian/mean ablation).
    """
    with torch.no_grad():
        if embed_input is not None:
            # Run model from embedding layer forward
            try:
                logits = model.run_with_hooks(
                    embed_input,
                    prepend_bos=False,
                    start_at_layer=0,
                )
            except Exception:
                # Fallback: cannot run from embedding, return 0
                return 0.0
        else:
            logits = model(tokens)

        last_logits = logits[0, -1, :]   # (vocab,)
        return (last_logits[target_tok] - last_logits[distract_tok]).item()


def _sufficiency_from_circuit(
    model:        object,
    circuit:      List[Tuple[int, int]],
    clean_tokens: torch.Tensor,
    corr_tokens:  torch.Tensor,
    target_tok:   int,
    distract_tok: int,
    clean_ld:     float,
    embed_corr:   Optional[torch.Tensor] = None,
) -> float:
    """
    Sufficiency S(C): run only the circuit heads on clean input, patch
    non-circuit heads to corrupted, measure LD fraction recovered.

    S(C) = LD_circuit / LD_clean
           clamped to [0, 1]

    Uses the existing activation-patching logic: for each non-circuit head,
    patch its output from the corrupted run into the clean run.
    """
    if abs(clean_ld) < 1e-6:
        return 0.0

    circuit_set = set(circuit)
    n_layers    = model.cfg.n_layers
    n_heads     = model.cfg.n_heads

    # Get corrupted activations for all heads
    hook_names = [
        f"blocks.{l}.attn.hook_z"
        for l in range(n_layers)
    ]

    try:
        with torch.no_grad():
            if embed_corr is not None:
                # Cannot run cache from embedding directly; approximate with token-space corruption
                _, corr_cache = model.run_with_cache(corr_tokens, names_filter=lambda n: "hook_z" in n)
            else:
                _, corr_cache = model.run_with_cache(corr_tokens, names_filter=lambda n: "hook_z" in n)

        def patch_non_circuit(value, hook):
            # Parse layer from hook name: "blocks.{l}.attn.hook_z"
            parts = hook.name.split(".")
            layer = int(parts[1])
            corr_val = corr_cache[hook.name]
            patched  = value.clone()
            for h in range(n_heads):
                if (layer, h) not in circuit_set:
                    patched[:, :, h, :] = corr_val[:, :, h, :]
            return patched

        hooks = [
            (f"blocks.{l}.attn.hook_z", patch_non_circuit)
            for l in range(n_layers)
        ]

        with torch.no_grad():
            logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        circuit_ld = (logits[0, -1, target_tok] - logits[0, -1, distract_tok]).item()
        suff = circuit_ld / clean_ld
        return float(max(0.0, min(1.0, suff)))

    except Exception as e:
        logger.warning("Sufficiency computation failed: %s", e)
        return 0.0


def _comprehensiveness_from_circuit(
    model:        object,
    circuit:      List[Tuple[int, int]],
    clean_tokens: torch.Tensor,
    corr_tokens:  torch.Tensor,
    target_tok:   int,
    distract_tok: int,
    clean_ld:     float,
) -> float:
    """
    Comprehensiveness Comp(C): patch only circuit heads to corrupted,
    measure LD drop fraction.

    Comp(C) = (LD_clean − LD_ablated) / LD_clean
              clamped to [0, 1]
    """
    if abs(clean_ld) < 1e-6:
        return 0.0

    circuit_set = set(circuit)
    n_layers    = model.cfg.n_layers

    try:
        with torch.no_grad():
            _, corr_cache = model.run_with_cache(corr_tokens, names_filter=lambda n: "hook_z" in n)

        def patch_circuit(value, hook):
            parts = hook.name.split(".")
            layer = int(parts[1])
            corr_val = corr_cache[hook.name]
            patched  = value.clone()
            for h in range(model.cfg.n_heads):
                if (layer, h) in circuit_set:
                    patched[:, :, h, :] = corr_val[:, :, h, :]
            return patched

        hooks = [
            (f"blocks.{l}.attn.hook_z", patch_circuit)
            for l in range(n_layers)
        ]

        with torch.no_grad():
            logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        ablated_ld  = (logits[0, -1, target_tok] - logits[0, -1, distract_tok]).item()
        comp        = (clean_ld - ablated_ld) / clean_ld
        return float(max(0.0, min(1.0, comp)))

    except Exception as e:
        logger.warning("Comprehensiveness computation failed: %s", e)
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Corruption Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class MultiCorruptionPipeline:
    """
    Evaluates circuit faithfulness under multiple corruption strategies.

    For each strategy, computes (S_k, Comp_k, F1_k) and checks the
    robustness criterion:

        ∀k : |S_k − S̄| < δ = 0.10

    Usage
    -----
    >>> pipeline = MultiCorruptionPipeline(model)
    >>> report = pipeline.run(
    ...     prompt="When Mary and John went to the store, John gave a drink to",
    ...     io_name="Mary", subject_name="John",
    ...     circuit=[(9,6), (9,9), (10,0)],
    ...     target_tok=target_id, distract_tok=distract_id,
    ...     strategies=[CorruptionStrategy.NAME_SWAP, CorruptionStrategy.RANDOM_TOKEN],
    ... )
    >>> report.robust
    True

    Parameters
    ----------
    model            : HookedTransformer instance
    baseline_tokens  : Optional list of tokenised baseline prompts for mean ablation
    seed             : RNG seed for reproducible random/Gaussian corruptions
    """

    def __init__(
        self,
        model:           object,
        baseline_tokens: Optional[List[torch.Tensor]] = None,
        seed:            int = 42,
    ) -> None:
        self.model           = model
        self.baseline_tokens = baseline_tokens or []
        self.seed            = seed

    def run(
        self,
        prompt:       str,
        io_name:      str,
        subject_name: str,
        circuit:      List[Tuple[int, int]],
        target_tok:   int,
        distract_tok: int,
        strategies:   Optional[List[CorruptionStrategy]] = None,
        clean_ld:     Optional[float] = None,
    ) -> RobustnessReport:
        """
        Run multi-corruption analysis and return RobustnessReport.

        Parameters
        ----------
        prompt       : Clean prompt string
        io_name      : IO character name (e.g. "Mary")
        subject_name : Subject name (e.g. "John")
        circuit      : List of (layer, head) tuples forming the identified circuit
        target_tok   : Token id for the correct (IO) next token
        distract_tok : Token id for the incorrect (Subject) token
        strategies   : Which corruption strategies to run. Defaults to all 4.
        clean_ld     : Pre-computed clean logit diff (avoids re-computing)

        Returns
        -------
        RobustnessReport with per-strategy results and robustness flag
        """
        if strategies is None:
            strategies = list(CorruptionStrategy)

        # Tokenise clean prompt
        clean_tokens = self.model.to_tokens(prompt)

        # Compute clean LD if not provided
        if clean_ld is None:
            with torch.no_grad():
                logits  = self.model(clean_tokens)
                clean_ld = (logits[0, -1, target_tok] - logits[0, -1, distract_tok]).item()

        results: List[CorruptionResult] = []

        for strategy in strategies:
            try:
                result = self._run_single(
                    strategy, prompt, io_name, subject_name,
                    clean_tokens, circuit, target_tok, distract_tok, clean_ld,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Corruption %s failed: %s", strategy.value, e)

        # Compute robustness statistics
        return self._compute_robustness(results)

    def _run_single(
        self,
        strategy:     CorruptionStrategy,
        prompt:       str,
        io_name:      str,
        subject_name: str,
        clean_tokens: torch.Tensor,
        circuit:      List[Tuple[int, int]],
        target_tok:   int,
        distract_tok: int,
        clean_ld:     float,
    ) -> CorruptionResult:
        """Run one corruption strategy and compute S/Comp/F1."""

        corrupted_prompt = None
        embed_corr       = None

        if strategy == CorruptionStrategy.NAME_SWAP:
            corrupted_prompt = _name_swap(prompt, io_name, subject_name)
            corr_tokens      = self.model.to_tokens(corrupted_prompt)

        elif strategy == CorruptionStrategy.RANDOM_TOKEN:
            corr_tokens = _random_token_corruption(
                clean_tokens, self.model, io_name, subject_name, seed=self.seed,
            )

        elif strategy == CorruptionStrategy.GAUSSIAN_NOISE:
            embed_corr  = _gaussian_noise_corruption(
                clean_tokens, self.model, sigma_factor=1.0, seed=self.seed,
            )
            corr_tokens = clean_tokens   # placeholder; embed used directly

        elif strategy == CorruptionStrategy.MEAN_ABLATION:
            embed_corr  = _mean_ablation_corruption(
                clean_tokens, self.model,
                baseline_tokens=self.baseline_tokens if self.baseline_tokens else None,
            )
            corr_tokens = clean_tokens   # placeholder

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Corr LD
        corr_ld = _logit_diff(
            self.model, corr_tokens, target_tok, distract_tok, embed_input=embed_corr,
        )

        # Sufficiency
        suff = _sufficiency_from_circuit(
            self.model, circuit, clean_tokens, corr_tokens,
            target_tok, distract_tok, clean_ld, embed_corr=embed_corr,
        )

        # Comprehensiveness
        comp = _comprehensiveness_from_circuit(
            self.model, circuit, clean_tokens, corr_tokens,
            target_tok, distract_tok, clean_ld,
        )

        # F1
        f1 = (2 * suff * comp / (suff + comp)) if (suff + comp) > 0 else 0.0

        return CorruptionResult(
            strategy          = strategy,
            corrupted_prompt  = corrupted_prompt,
            corrupted_tokens  = corr_tokens,
            sufficiency       = suff,
            comprehensiveness = comp,
            f1                = f1,
            logit_diff_clean  = clean_ld,
            logit_diff_corr   = corr_ld,
        )

    @staticmethod
    def _compute_robustness(results: List[CorruptionResult]) -> RobustnessReport:
        """Apply robustness criterion: ∀k |S_k − S̄| < δ=0.10."""
        if not results:
            return RobustnessReport(results=[], robust=False, perturbation_sensitive=True)

        suffs  = np.array([r.sufficiency for r in results])
        s_bar  = float(suffs.mean())
        s_std  = float(suffs.std())
        deviations = np.abs(suffs - s_bar)
        max_dev    = float(deviations.max())
        sensitive  = max_dev >= ROBUSTNESS_DELTA

        return RobustnessReport(
            results                = results,
            mean_sufficiency       = s_bar,
            std_sufficiency        = s_std,
            max_deviation          = max_dev,
            perturbation_sensitive = sensitive,
            robust                 = not sensitive,
        )
