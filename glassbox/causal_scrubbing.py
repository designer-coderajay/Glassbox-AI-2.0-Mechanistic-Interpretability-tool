"""
glassbox/causal_scrubbing.py
==============================
Causal Scrubbing — v4.1.0
===========================

Implements the Causal Scrubbing framework from Anthropic (Chan et al. 2022)
for rigorous causal evaluation of circuit hypotheses.

Background
----------
Attribution patching identifies *which* heads are important, but does not
answer: "Does the identified circuit causally implement the computation we
claim it does?"

Causal scrubbing answers this by evaluating a *circuit hypothesis* H — a
structured claim about which information flows through which edges — against
the empirical behaviour of the model.

The causal scrubbing score CS(H) measures what fraction of the model's
performance is explained by the hypothesis:

    CS(H) = E[LD(x; do(acts ~ P_H))] / LD_clean

where `do(acts ~ P_H)` means: resample activations at non-hypothesised
nodes from inputs that are *equivalent under H* (i.e. the hypothesis
predicts the same activation distribution for these inputs).

Algorithm (Simplified)
----------------------
1. Define `CircuitHypothesis H`: specify which (layer, head) → (input_type)
   mappings constitute the circuit. Non-hypothesised heads get their
   activations resampled from the corrupted run.
2. For each head NOT in the hypothesis: replace activation with a sample
   from P_corrupt(z_h) = the head's activation on a corrupted input.
3. For heads IN the hypothesis: keep clean activation.
4. Measure LD on this patched run.
5. CS(H) = mean(LD_patched) / LD_clean

The key insight: if H is correct, CS(H) ≈ 1.0 (hypothesis explains everything).
If H misses important components, CS(H) < 1.0.

Interpretation
--------------
CS(H) ∈ [0, 1] (approximately; can exceed 1.0 due to superposition)
- CS(H) ≥ 0.80 : Strong evidence hypothesis is correct
- CS(H) ∈ [0.50, 0.80] : Partial hypothesis, missing components
- CS(H) < 0.50 : Hypothesis insufficient

References
----------
Chan et al. 2022 — "Causal Scrubbing: a method for rigorously testing
    interpretability hypotheses"  (Anthropic Interpretability Team)
    https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/
    causal-scrubbing-a-method-for-rigorously-testing

Geiger et al. 2021 — "Causal Abstractions of Neural Networks"
    https://arxiv.org/abs/2106.02997

Conmy et al. 2023 — "Towards Automated Circuit Discovery"
    https://arxiv.org/abs/2304.14997
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)

# CS score thresholds (Chan et al. 2022)
CS_STRONG_THRESHOLD: float = 0.80
CS_PARTIAL_THRESHOLD: float = 0.50


# ──────────────────────────────────────────────────────────────────────────────
# Circuit Hypothesis specification
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CircuitHypothesis:
    """
    A structured hypothesis about which attention heads form a circuit.

    Attributes
    ----------
    name          : Human-readable name (e.g. "IOI_name_mover_circuit")
    heads         : Set of (layer, head) tuples claimed to be in the circuit
    description   : Free-text description of what the circuit computes
    expected_roles: Optional mapping {(layer,head): "role_name"} from Wang et al.
    """
    name:           str
    heads:          FrozenSet[Tuple[int, int]]
    description:    str = ""
    expected_roles: Dict[Tuple[int, int], str] = field(default_factory=dict)

    @classmethod
    def from_list(
        cls,
        name:        str,
        heads:       List[Tuple[int, int]],
        description: str = "",
        roles:       Optional[Dict[Tuple[int, int], str]] = None,
    ) -> "CircuitHypothesis":
        """Convenience constructor from a list of (layer, head) tuples."""
        return cls(
            name           = name,
            heads          = frozenset(heads),
            description    = description,
            expected_roles = roles or {},
        )

    @classmethod
    def from_wang2022_ioi(cls) -> "CircuitHypothesis":
        """
        The canonical IOI circuit from Wang et al. (2022).

        Includes name-mover heads, S-inhibition heads, induction heads,
        duplicate token heads, and previous-token heads.
        """
        wang_circuit = [
            (9, 6), (9, 9), (10, 0),   # Name Mover heads
            (10, 6),                    # Name Mover (backup)
            (8, 6), (7, 3),            # Negative Name Movers
            (4, 11), (8, 10),          # S-Inhibition heads
            (7, 9), (3, 0),            # Induction heads
            (6, 9), (5, 5),            # Duplicate Token heads
            (0, 1),                    # Previous Token head
        ]
        roles = {
            (9, 6): "Name Mover",  (9, 9): "Name Mover",  (10, 0): "Name Mover",
            (10, 6): "Name Mover (backup)",
            (8, 6): "Negative NM", (7, 3): "Negative NM",
            (4, 11): "S-Inhibition", (8, 10): "S-Inhibition",
            (7, 9): "Induction", (3, 0): "Induction",
            (6, 9): "Dup Token", (5, 5): "Dup Token",
            (0, 1): "Prev Token",
        }
        return cls.from_list(
            name        = "Wang2022_IOI_Circuit",
            heads       = wang_circuit,
            description = "Full IOI circuit: name-movers, S-inhibition, induction, dup-token (Wang et al. 2022)",
            roles       = roles,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Causal Scrubbing Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalScrubbingResult:
    """
    Result of causal scrubbing evaluation for one hypothesis.

    Attributes
    ----------
    hypothesis         : The CircuitHypothesis that was tested
    cs_score           : CS(H) = E[LD_scrubbed] / LD_clean  ∈ [0,1+]
    ld_clean           : Clean logit difference (reference)
    ld_scrubbed_mean   : Mean LD after causal scrubbing
    ld_scrubbed_std    : Std of LD across scrubbing samples
    n_samples          : Number of corruption samples used
    interpretation     : "strong" / "partial" / "insufficient"
    fraction_explained : cs_score (alias for reporting)
    n_hypothesised     : Number of heads in hypothesis
    n_total_heads      : Total heads in the model
    """
    hypothesis:         CircuitHypothesis
    cs_score:           float
    ld_clean:           float
    ld_scrubbed_mean:   float
    ld_scrubbed_std:    float
    n_samples:          int
    interpretation:     str
    fraction_explained: float
    n_hypothesised:     int
    n_total_heads:      int

    def to_dict(self) -> Dict:
        return {
            "hypothesis":         self.hypothesis.name,
            "circuit_heads":      [f"L{l}H{h}" for l, h in sorted(self.hypothesis.heads)],
            "cs_score":           round(self.cs_score, 4),
            "ld_clean":           round(self.ld_clean, 4),
            "ld_scrubbed_mean":   round(self.ld_scrubbed_mean, 4),
            "ld_scrubbed_std":    round(self.ld_scrubbed_std, 4),
            "n_samples":          self.n_samples,
            "interpretation":     self.interpretation,
            "fraction_explained": round(self.fraction_explained, 4),
            "n_hypothesised":     self.n_hypothesised,
            "n_total_heads":      self.n_total_heads,
            "cs_strong_threshold": CS_STRONG_THRESHOLD,
            "cs_partial_threshold": CS_PARTIAL_THRESHOLD,
        }

    def summary_line(self) -> str:
        icons = {"strong": "✓✓", "partial": "~", "insufficient": "✗"}
        icon  = icons.get(self.interpretation, "?")
        return (
            f"CausalScrubbing [{self.hypothesis.name}] {icon} | "
            f"CS={self.cs_score:.4f} ({self.interpretation}) | "
            f"LD_clean={self.ld_clean:.4f} LD_scrubbed={self.ld_scrubbed_mean:.4f}±{self.ld_scrubbed_std:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CausalScrubbing class
# ──────────────────────────────────────────────────────────────────────────────

class CausalScrubbing:
    """
    Causal scrubbing evaluation for circuit hypotheses.

    Evaluates whether a CircuitHypothesis causally explains the model's
    behaviour on a task, following Chan et al. (2022).

    Algorithm:
    1. Cache all head activations on the clean run
    2. For each scrubbing sample:
       a. Run model on clean input
       b. For each head NOT in the hypothesis: patch its activation from
          a corrupted run (resample from P_corrupt)
       c. Measure LD on the patched run
    3. CS(H) = mean(LD_patched) / LD_clean

    Parameters
    ----------
    model    : HookedTransformer instance
    n_samples: Number of corruption samples for Monte Carlo estimation (default 5)

    Usage
    -----
    >>> scrubber = CausalScrubbing(model, n_samples=5)
    >>> hypothesis = CircuitHypothesis.from_wang2022_ioi()
    >>> result = scrubber.evaluate(
    ...     hypothesis=hypothesis,
    ...     prompt="When Mary and John went to the store, John gave a drink to",
    ...     corr_prompt="When John and Mary went to the store, Mary gave a drink to",
    ...     target_tok=target_id, distract_tok=distract_id,
    ... )
    >>> print(result.summary_line())
    CausalScrubbing [Wang2022_IOI_Circuit] ✓✓ | CS=0.8923 (strong) | LD_clean=4.23...
    """

    def __init__(
        self,
        model:     object,
        n_samples: int = 5,
    ) -> None:
        self.model     = model
        self.n_samples = n_samples
        self._n_layers = model.cfg.n_layers
        self._n_heads  = model.cfg.n_heads

    def evaluate(
        self,
        hypothesis:   CircuitHypothesis,
        prompt:       str,
        corr_prompt:  str,
        target_tok:   int,
        distract_tok: int,
    ) -> CausalScrubbingResult:
        """
        Evaluate CS(H) for a single prompt pair.

        Parameters
        ----------
        hypothesis   : CircuitHypothesis to evaluate
        prompt       : Clean input prompt
        corr_prompt  : Corrupted input (e.g. name-swapped)
        target_tok   : Correct next-token id
        distract_tok : Distractor token id

        Returns
        -------
        CausalScrubbingResult
        """
        clean_tokens = self.model.to_tokens(prompt)
        corr_tokens  = self.model.to_tokens(corr_prompt)

        # Clean LD
        with torch.no_grad():
            clean_logits = self.model(clean_tokens)
            ld_clean     = (clean_logits[0, -1, target_tok] - clean_logits[0, -1, distract_tok]).item()

        if abs(ld_clean) < 1e-6:
            logger.warning("Clean LD ≈ 0; CS score will be degenerate")

        # Cache corrupted activations for all heads
        with torch.no_grad():
            _, corr_cache = self.model.run_with_cache(
                corr_tokens,
                names_filter=lambda n: "hook_z" in n,
            )

        # Run scrubbing samples
        ld_scrubbed_samples: List[float] = []

        for _ in range(self.n_samples):
            ld_s = self._scrubbing_run(
                hypothesis, clean_tokens, corr_cache,
                target_tok, distract_tok,
            )
            ld_scrubbed_samples.append(ld_s)

        ld_mean = float(np.mean(ld_scrubbed_samples))
        ld_std  = float(np.std(ld_scrubbed_samples))

        cs_score = ld_mean / ld_clean if abs(ld_clean) > 1e-6 else 0.0

        if cs_score >= CS_STRONG_THRESHOLD:
            interpretation = "strong"
        elif cs_score >= CS_PARTIAL_THRESHOLD:
            interpretation = "partial"
        else:
            interpretation = "insufficient"

        return CausalScrubbingResult(
            hypothesis         = hypothesis,
            cs_score           = cs_score,
            ld_clean           = ld_clean,
            ld_scrubbed_mean   = ld_mean,
            ld_scrubbed_std    = ld_std,
            n_samples          = self.n_samples,
            interpretation     = interpretation,
            fraction_explained = cs_score,
            n_hypothesised     = len(hypothesis.heads),
            n_total_heads      = self._n_layers * self._n_heads,
        )

    def evaluate_batch(
        self,
        hypothesis:    CircuitHypothesis,
        prompts:       List[Tuple[str, str, int, int]],  # (clean, corr, target, distract)
    ) -> List[CausalScrubbingResult]:
        """
        Evaluate hypothesis on multiple prompts.

        Parameters
        ----------
        hypothesis : CircuitHypothesis to evaluate
        prompts    : List of (clean_prompt, corr_prompt, target_tok, distract_tok) tuples

        Returns
        -------
        List of CausalScrubbingResult (one per prompt)
        """
        results = []
        for clean_p, corr_p, t_tok, d_tok in prompts:
            try:
                r = self.evaluate(hypothesis, clean_p, corr_p, t_tok, d_tok)
                results.append(r)
            except Exception as e:
                logger.warning("CausalScrubbing failed for prompt '%s...': %s", clean_p[:40], e)
        return results

    def mean_cs_score(
        self,
        results: List[CausalScrubbingResult],
    ) -> Dict:
        """
        Aggregate CS scores across multiple prompts.

        Returns
        -------
        Dict with mean_cs, std_cs, n_strong, n_partial, n_insufficient
        """
        if not results:
            return {"mean_cs": 0.0, "std_cs": 0.0, "n": 0}

        scores = [r.cs_score for r in results]
        return {
            "mean_cs":        round(float(np.mean(scores)), 4),
            "std_cs":         round(float(np.std(scores)), 4),
            "n":              len(scores),
            "n_strong":       sum(1 for r in results if r.interpretation == "strong"),
            "n_partial":      sum(1 for r in results if r.interpretation == "partial"),
            "n_insufficient": sum(1 for r in results if r.interpretation == "insufficient"),
            "hypothesis":     results[0].hypothesis.name if results else "",
        }

    def _scrubbing_run(
        self,
        hypothesis:   CircuitHypothesis,
        clean_tokens: torch.Tensor,
        corr_cache:   object,   # ActivationCache from run_with_cache
        target_tok:   int,
        distract_tok: int,
    ) -> float:
        """
        One causal scrubbing run: patch non-hypothesised heads with corrupt activations.

        Returns LD on the patched run.
        """
        def patch_hook(value, hook):
            parts = hook.name.split(".")
            layer = int(parts[1])
            corr_val = corr_cache[hook.name]   # (1, seq, n_heads, d_head)
            patched  = value.clone()

            for h in range(self._n_heads):
                if (layer, h) not in hypothesis.heads:
                    # Resample from corrupted run for this head
                    patched[:, :, h, :] = corr_val[:, :, h, :]

            return patched

        hooks = [
            (f"blocks.{l}.attn.hook_z", patch_hook)
            for l in range(self._n_layers)
        ]

        with torch.no_grad():
            logits = self.model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        ld = (logits[0, -1, target_tok] - logits[0, -1, distract_tok]).item()
        return float(ld)
