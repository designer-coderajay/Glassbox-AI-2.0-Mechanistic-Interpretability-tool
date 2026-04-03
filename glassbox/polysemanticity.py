"""
glassbox/polysemanticity.py
============================
SAE-Based Polysemanticity Score — v4.0.0
==========================================

Quantifies the polysemanticity of attention heads by measuring the entropy
of the feature activation distribution over SAE (Sparse Autoencoder) features.

Background
----------
A *monosemantic* head activates a small number of interpretable SAE features
for any given input. A *polysemantic* head activates many features, meaning it
compresses multiple unrelated computations into a single head.

Polysemanticity Score
---------------------
For head h, the polysemanticity score P(h) is the Shannon entropy of the
feature activation probability distribution:

    P(h) = H(p(feature | head_h))
         = -Σ_f p_f · log₂(p_f)

where p_f = normalized mean activation of feature f on head h's output.

High P(h) → high polysemanticity (many features, head hard to interpret)
Low P(h)  → low polysemanticity (few features, head is monosemantic)

Normalized polysemanticity:

    P_norm(h) = P(h) / log₂(n_features)   ∈ [0, 1]

Implementation
--------------
For each prompt in the batch:
1. Run GlassboxEngine.analyze() with circuit heads
2. For each circuit head h, extract output z_h (shape: seq × d_head)
3. Project through SAE encoder: f = ReLU(W_enc · z_h − b_enc)
4. Accumulate mean activations per feature across prompts
5. Normalize to probability distribution and compute entropy

If no SAE is available (sae-lens not installed), falls back to a
PCA-based effective dimensionality estimate using participation ratio:

    PR(h) = (Σ λ_i)² / Σ λ_i²

where λ_i are eigenvalues of the activation covariance matrix.
PR(h) / n_components gives a [0,1] polysemanticity estimate.

References
----------
Elhage et al. 2022 — "Toy Models of Superposition"
    https://transformer-circuits.pub/2022/toy_model/index.html
    Introduces polysemanticity and superposition in neural networks.

Bricken et al. 2023 — "Towards Monosemanticity"
    Anthropic Interpretability Team.
    https://transformer-circuits.pub/2023/monosemanticity/index.html
    Uses SAE features to measure and reduce polysemanticity.

Templeton et al. 2024 — "Scaling Monosemanticity"
    https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadPolysemanticity:
    """
    Polysemanticity score for a single attention head.

    Attributes
    ----------
    layer              : Transformer layer index
    head               : Attention head index
    entropy_bits       : H(p(feature|head_h)) in bits
    entropy_normalized : entropy_bits / log₂(n_features)  ∈ [0, 1]
    n_features_active  : Mean number of features active per prompt (> threshold)
    method             : "sae_entropy" or "pca_participation_ratio"
    monosemantic       : True if entropy_normalized < 0.25
    """
    layer:              int
    head:               int
    entropy_bits:       float
    entropy_normalized: float
    n_features_active:  float
    method:             str
    monosemantic:       bool

    def to_dict(self) -> Dict:
        return {
            "head":               f"L{self.layer}H{self.head}",
            "layer":              self.layer,
            "head_idx":           self.head,
            "entropy_bits":       round(self.entropy_bits, 4),
            "entropy_normalized": round(self.entropy_normalized, 4),
            "n_features_active":  round(self.n_features_active, 1),
            "method":             self.method,
            "monosemantic":       self.monosemantic,
        }


@dataclass
class PolysemanticitySummary:
    """
    Summary of polysemanticity scores across all circuit heads.

    Attributes
    ----------
    head_scores      : Per-head polysemanticity results
    mean_entropy_norm: Mean normalized entropy across all circuit heads
    monosemantic_fraction: Fraction of heads with entropy_norm < 0.25
    most_polysemantic: (layer, head) with highest entropy
    most_monosemantic: (layer, head) with lowest entropy
    method           : Computation method used
    """
    head_scores:           List[HeadPolysemanticity]
    mean_entropy_norm:     float
    monosemantic_fraction: float
    most_polysemantic:     Optional[Tuple[int, int]]
    most_monosemantic:     Optional[Tuple[int, int]]
    method:                str

    def to_dict(self) -> Dict:
        return {
            "mean_entropy_normalized": round(self.mean_entropy_norm, 4),
            "monosemantic_fraction":   round(self.monosemantic_fraction, 4),
            "most_polysemantic": (
                f"L{self.most_polysemantic[0]}H{self.most_polysemantic[1]}"
                if self.most_polysemantic else None
            ),
            "most_monosemantic": (
                f"L{self.most_monosemantic[0]}H{self.most_monosemantic[1]}"
                if self.most_monosemantic else None
            ),
            "method": self.method,
            "per_head": [h.to_dict() for h in self.head_scores],
        }

    def summary_line(self) -> str:
        mono_pct = self.monosemantic_fraction * 100
        return (
            f"Polysemanticity [method={self.method}] | "
            f"mean_entropy={self.mean_entropy_norm:.3f} "
            f"monosemantic={mono_pct:.0f}% of circuit heads"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Polysemanticity Scorer
# ──────────────────────────────────────────────────────────────────────────────

class PolysemanticityScorerSAE:
    """
    Compute polysemanticity scores for attention heads using SAE features.

    Falls back to PCA participation ratio if sae-lens is not installed.

    Parameters
    ----------
    model            : HookedTransformer instance
    sae              : Optional SAE model from sae-lens (dict {layer: SAE} or None)
    activation_threshold : Feature activation threshold for counting active features
    monosemantic_threshold: entropy_norm below this → monosemantic (default 0.25)

    Usage
    -----
    >>> scorer = PolysemanticityScorerSAE(model)
    >>> summary = scorer.score_circuit(
    ...     circuit=[(9,6), (9,9), (10,0)],
    ...     prompts_tokens=[tokens1, tokens2, ...],
    ... )
    >>> print(summary.summary_line())
    Polysemanticity [method=pca_participation_ratio] | mean_entropy=0.312 monosemantic=67% of circuit heads
    """

    def __init__(
        self,
        model:                  object,
        sae:                    Optional[Dict] = None,
        activation_threshold:   float = 0.01,
        monosemantic_threshold: float = 0.25,
    ) -> None:
        self.model                  = model
        self.sae                    = sae
        self.activation_threshold   = activation_threshold
        self.monosemantic_threshold = monosemantic_threshold
        self._d_head                = model.cfg.d_head

    def score_circuit(
        self,
        circuit:       List[Tuple[int, int]],
        prompts_tokens: List[torch.Tensor],
    ) -> PolysemanticitySummary:
        """
        Compute polysemanticity for each head in the circuit.

        Parameters
        ----------
        circuit        : List of (layer, head) tuples
        prompts_tokens : List of tokenised prompts (1, seq_len) tensors

        Returns
        -------
        PolysemanticitySummary
        """
        if not circuit:
            return PolysemanticitySummary(
                head_scores=[], mean_entropy_norm=0.0,
                monosemantic_fraction=0.0,
                most_polysemantic=None, most_monosemantic=None,
                method="none",
            )

        # Collect head activations across prompts
        head_activations: Dict[Tuple[int, int], List[torch.Tensor]] = {
            h: [] for h in circuit
        }

        for tokens in prompts_tokens:
            try:
                hook_names = {
                    f"blocks.{l}.attn.hook_z": (l, h)
                    for l, h in circuit
                }

                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        tokens,
                        names_filter=lambda n: n in hook_names,
                    )

                for hook_name, (l, h) in hook_names.items():
                    if hook_name in cache:
                        z = cache[hook_name][0, :, h, :]   # (seq, d_head)
                        head_activations[(l, h)].append(z.cpu().float())
            except Exception as e:
                logger.debug("Polysemanticity activation collection failed: %s", e)

        head_scores: List[HeadPolysemanticity] = []

        for (l, h), activations in head_activations.items():
            if not activations:
                continue
            score = self._score_head(l, h, activations)
            head_scores.append(score)

        if not head_scores:
            return PolysemanticitySummary(
                head_scores=[], mean_entropy_norm=0.0,
                monosemantic_fraction=0.0,
                most_polysemantic=None, most_monosemantic=None,
                method="unavailable",
            )

        # Aggregate
        entropies = [s.entropy_normalized for s in head_scores]
        mean_ent  = float(np.mean(entropies))
        mono_frac = float(sum(1 for s in head_scores if s.monosemantic) / len(head_scores))

        sorted_by_entropy = sorted(head_scores, key=lambda s: s.entropy_normalized)
        most_mono = (sorted_by_entropy[0].layer, sorted_by_entropy[0].head)
        most_poly = (sorted_by_entropy[-1].layer, sorted_by_entropy[-1].head)
        method    = head_scores[0].method if head_scores else "unavailable"

        return PolysemanticitySummary(
            head_scores           = head_scores,
            mean_entropy_norm     = mean_ent,
            monosemantic_fraction = mono_frac,
            most_polysemantic     = most_poly,
            most_monosemantic     = most_mono,
            method                = method,
        )

    def _score_head(
        self,
        layer:       int,
        head:        int,
        activations: List[torch.Tensor],
    ) -> HeadPolysemanticity:
        """Score a single head using SAE entropy or PCA fallback."""
        all_acts = torch.cat(activations, dim=0)   # (n_tokens, d_head)

        # Try SAE-based entropy if available
        if self.sae is not None and layer in self.sae:
            return self._sae_entropy(layer, head, all_acts)

        # Fall back to PCA participation ratio
        return self._pca_participation_ratio(layer, head, all_acts)

    def _sae_entropy(
        self,
        layer:    int,
        head:     int,
        acts:     torch.Tensor,   # (n_tokens, d_head)
    ) -> HeadPolysemanticity:
        """SAE-based entropy: H(p(feature | head_h))."""
        try:
            sae_model = self.sae[layer]
            with torch.no_grad():
                features = sae_model.encode(acts)    # (n_tokens, n_features)
                features = torch.relu(features)

            # Mean activation per feature (proxy for p_f)
            mean_acts = features.mean(dim=0)         # (n_features,)
            total     = mean_acts.sum().item()

            if total < 1e-8:
                prob = torch.ones_like(mean_acts) / len(mean_acts)
            else:
                prob = mean_acts / total

            prob_np = prob.cpu().numpy()
            prob_np = prob_np[prob_np > 1e-12]       # remove zeros for log

            entropy_bits = -float(np.sum(prob_np * np.log2(prob_np)))
            n_features   = len(mean_acts)
            entropy_norm = entropy_bits / np.log2(max(n_features, 2))
            n_active     = float((features > self.activation_threshold).float().mean(dim=0).sum().item())

            return HeadPolysemanticity(
                layer              = layer,
                head               = head,
                entropy_bits       = entropy_bits,
                entropy_normalized = float(np.clip(entropy_norm, 0.0, 1.0)),
                n_features_active  = n_active,
                method             = "sae_entropy",
                monosemantic       = entropy_norm < self.monosemantic_threshold,
            )
        except Exception as e:
            logger.debug("SAE entropy failed for L%dH%d: %s", layer, head, e)
            return self._pca_participation_ratio(layer, head, acts)

    def _pca_participation_ratio(
        self,
        layer: int,
        head:  int,
        acts:  torch.Tensor,   # (n_tokens, d_head)
    ) -> HeadPolysemanticity:
        """
        PCA participation ratio fallback.

        PR = (Σ λ_i)² / Σ λ_i²
        P_norm = PR / n_components   ∈ [0, 1]

        High PR → many dimensions active → polysemantic
        Low PR  → few dimensions active → monosemantic
        """
        try:
            acts_np = acts.cpu().numpy().astype(np.float32)

            # Remove near-zero rows
            norms = np.linalg.norm(acts_np, axis=1)
            acts_np = acts_np[norms > 1e-6]

            if len(acts_np) < 2:
                # Not enough data, return neutral score
                return HeadPolysemanticity(
                    layer=layer, head=head,
                    entropy_bits=0.0, entropy_normalized=0.5,
                    n_features_active=self._d_head / 2,
                    method="pca_participation_ratio",
                    monosemantic=False,
                )

            # Center
            acts_centered = acts_np - acts_np.mean(axis=0, keepdims=True)

            # Covariance eigenvalues via SVD (numerically stable)
            _, s, _ = np.linalg.svd(acts_centered, full_matrices=False)
            lambdas  = s ** 2 / (len(acts_np) - 1)
            lambdas  = lambdas[lambdas > 1e-10]

            if len(lambdas) == 0:
                pr_norm = 0.0
            else:
                pr = (lambdas.sum()) ** 2 / (lambdas ** 2).sum()
                pr_norm = float(pr / max(len(lambdas), 1))
                pr_norm = float(np.clip(pr_norm, 0.0, 1.0))

            # Map PR to entropy-equivalent bits for unified reporting
            n_comp         = self._d_head
            entropy_bits   = pr_norm * np.log2(max(n_comp, 2))
            n_active       = float(pr_norm * n_comp)

            return HeadPolysemanticity(
                layer              = layer,
                head               = head,
                entropy_bits       = float(entropy_bits),
                entropy_normalized = pr_norm,
                n_features_active  = n_active,
                method             = "pca_participation_ratio",
                monosemantic       = pr_norm < self.monosemantic_threshold,
            )

        except Exception as e:
            logger.warning("PCA participation ratio failed for L%dH%d: %s", layer, head, e)
            return HeadPolysemanticity(
                layer=layer, head=head,
                entropy_bits=0.0, entropy_normalized=0.0,
                n_features_active=0.0,
                method="failed",
                monosemantic=True,
            )
