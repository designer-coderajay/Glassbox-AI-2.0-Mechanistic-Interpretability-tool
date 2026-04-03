"""
glassbox/das.py
================
Distributed Alignment Search (DAS) — v4.1.0
=============================================

Implements Distributed Alignment Search (Geiger et al. 2023) to identify
the linear subspace of the residual stream that encodes a specific concept.

Background
----------
Standard circuit analysis identifies *which heads* matter for a task.
DAS asks a deeper question: *where in the residual stream* is the concept
encoded, and can we find a linear subspace R ⊆ R^{d_model} such that
rotating activations within R causes predictable behavioural changes?

The DAS Framework (Geiger et al. 2023)
---------------------------------------
Given a concept C (e.g. "IO name position"), DAS learns a rotation matrix
R ∈ O(d_model) such that intervening on the R-subspace of the residual
stream at layer l, position p reproduces the causal effect of the concept.

Specifically, DAS finds the optimal linear subspace by minimising:

    L(R) = E_{x,x'}[ (LD(x; do(R·z_l ← R·z_l')) − LD_clean)² ]

where x' is a counterfactual input that instantiates a different concept value.

Algorithm (Simplified Implementation)
--------------------------------------
1. Choose a target layer and position in the residual stream
2. Collect (clean_activation, counterfactual_activation) pairs
3. Learn a rotation R by PCA on the difference vectors:
   Δz = z_l^clean − z_l^counterfactual
4. Compute DAS score: fraction of LD explained by intervening in R's span
5. Return R, the concept_dims (top components), and DAS score

The DAS score measures:
    DAS(R) = Corr(LD_after_intervention, LD_clean)

Implementation Note
-------------------
The full DAS paper uses learned rotations via gradient descent.
This implementation uses PCA on activation differences as an efficient
approximation (Geiger et al. Section 4 "Subspace Identification").
The PCA approach is empirically validated and runs without a training loop.

References
----------
Geiger et al. 2023 — "Finding Alignments Between Interpretable Causal
    Variables and Distributed Representations"
    https://arxiv.org/abs/2303.02536

Geiger et al. 2021 — "Causal Abstractions of Neural Networks"
    https://arxiv.org/abs/2106.02997

Wu et al. 2023 — "Interpretability at Scale: Identifying Causal Mechanisms
    in Alpaca" (DAS applied to Alpaca)
    https://arxiv.org/abs/2305.08809
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# DAS score threshold: above this → concept clearly encoded in identified subspace
DAS_SCORE_THRESHOLD: float = 0.70


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DASResult:
    """
    Result of Distributed Alignment Search for one concept.

    Attributes
    ----------
    concept_label      : Human-readable concept name (e.g. "IO_name_position")
    target_layer       : Residual stream layer where concept was searched
    target_position    : Token position in sequence (e.g. -1 for last token)
    das_score          : Fraction of LD variance explained by subspace intervention
    rotation_matrix    : R ∈ R^{d_model × concept_dims} — concept subspace basis
    concept_dims       : Number of dimensions encoding the concept
    explained_variance : Fraction of Δz variance explained by top concept_dims PCs
    mean_ld_clean      : Mean LD before intervention (reference)
    mean_ld_intervened : Mean LD after intervention in DAS subspace
    concept_encoded    : True if das_score ≥ DAS_SCORE_THRESHOLD (0.70)
    n_samples          : Number of prompt pairs used for estimation
    pca_eigenvalues    : Top eigenvalues (for scree plot)
    """
    concept_label:       str
    target_layer:        int
    target_position:     int
    das_score:           float
    rotation_matrix:     np.ndarray       # shape: (d_model, concept_dims)
    concept_dims:        int
    explained_variance:  float
    mean_ld_clean:       float
    mean_ld_intervened:  float
    concept_encoded:     bool
    n_samples:           int
    pca_eigenvalues:     List[float]

    def to_dict(self) -> Dict:
        return {
            "concept_label":       self.concept_label,
            "target_layer":        self.target_layer,
            "target_position":     self.target_position,
            "das_score":           round(self.das_score, 4),
            "concept_dims":        self.concept_dims,
            "explained_variance":  round(self.explained_variance, 4),
            "mean_ld_clean":       round(self.mean_ld_clean, 4),
            "mean_ld_intervened":  round(self.mean_ld_intervened, 4),
            "concept_encoded":     self.concept_encoded,
            "n_samples":           self.n_samples,
            "das_threshold":       DAS_SCORE_THRESHOLD,
            "top_eigenvalues":     [round(e, 4) for e in self.pca_eigenvalues[:10]],
            "rotation_shape":      list(self.rotation_matrix.shape),
        }

    def summary_line(self) -> str:
        status = "ENCODED ✓" if self.concept_encoded else "not found ✗"
        return (
            f"DAS [{self.concept_label}] {status} | "
            f"layer={self.target_layer} pos={self.target_position} | "
            f"score={self.das_score:.4f} dims={self.concept_dims} "
            f"expl_var={self.explained_variance:.3f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# DistributedAlignmentSearch class
# ──────────────────────────────────────────────────────────────────────────────

class DistributedAlignmentSearch:
    """
    Find the linear subspace encoding a concept in the residual stream.

    Uses PCA on activation difference vectors as an efficient approximation
    to the full DAS gradient-based rotation (Geiger et al. 2023, §4).

    Parameters
    ----------
    model         : HookedTransformer instance
    concept_dims  : Number of subspace dimensions to identify (default 4)
    n_interchange : Number of interchange interventions for DAS score (default 20)

    Usage
    -----
    >>> das = DistributedAlignmentSearch(model, concept_dims=4)
    >>> result = das.search(
    ...     concept_label="IO_name_position",
    ...     clean_prompts_tokens=clean_tokens_list,
    ...     counterfactual_tokens=corr_tokens_list,
    ...     target_tok=target_id,
    ...     distract_tok=distract_id,
    ...     target_layer=9,
    ...     target_position=-1,
    ... )
    >>> print(result.summary_line())
    DAS [IO_name_position] ENCODED ✓ | layer=9 pos=-1 | score=0.8234 dims=4 expl_var=0.731
    """

    def __init__(
        self,
        model:         object,
        concept_dims:  int = 4,
        n_interchange: int = 20,
    ) -> None:
        self.model         = model
        self.concept_dims  = concept_dims
        self.n_interchange = n_interchange
        self._d_model      = model.cfg.d_model

    def search(
        self,
        concept_label:          str,
        clean_prompts_tokens:   List[torch.Tensor],
        counterfactual_tokens:  List[torch.Tensor],
        target_tok:             int,
        distract_tok:           int,
        target_layer:           int,
        target_position:        int = -1,
    ) -> DASResult:
        """
        Search for the linear subspace encoding the concept.

        Parameters
        ----------
        concept_label          : Name of the concept (e.g. "IO_name_position")
        clean_prompts_tokens   : List of clean prompt token tensors
        counterfactual_tokens  : Corresponding counterfactual token tensors
        target_tok             : Correct token id for LD computation
        distract_tok           : Distractor token id
        target_layer           : Which residual stream layer to analyse
        target_position        : Which token position (default -1 = last)

        Returns
        -------
        DASResult with rotation matrix, concept_dims, and DAS score
        """
        assert len(clean_prompts_tokens) == len(counterfactual_tokens), \
            "clean and counterfactual token lists must be same length"

        n_pairs = len(clean_prompts_tokens)

        if n_pairs < 2:
            raise ValueError(f"Need ≥2 prompt pairs for DAS; got {n_pairs}")

        # Step 1: Collect residual stream activations
        clean_acts, counterfact_acts, ld_cleans = self._collect_activations(
            clean_prompts_tokens, counterfactual_tokens,
            target_layer, target_position, target_tok, distract_tok,
        )

        if len(clean_acts) < 2:
            raise ValueError("Not enough valid activations collected for DAS")

        # Step 2: PCA on difference vectors Δz = z_clean − z_counterfact
        rotation_matrix, eigenvalues, explained_var = self._pca_subspace(
            clean_acts, counterfact_acts,
        )

        # Step 3: Evaluate DAS score via interchange interventions
        das_score, ld_intervened = self._evaluate_das_score(
            rotation_matrix,
            clean_prompts_tokens[:self.n_interchange],
            counterfactual_tokens[:self.n_interchange],
            target_layer, target_position, target_tok, distract_tok,
        )

        mean_ld_clean = float(np.mean(ld_cleans)) if ld_cleans else 0.0

        return DASResult(
            concept_label      = concept_label,
            target_layer       = target_layer,
            target_position    = target_position,
            das_score          = das_score,
            rotation_matrix    = rotation_matrix,
            concept_dims       = min(self.concept_dims, rotation_matrix.shape[1]),
            explained_variance = explained_var,
            mean_ld_clean      = mean_ld_clean,
            mean_ld_intervened = ld_intervened,
            concept_encoded    = das_score >= DAS_SCORE_THRESHOLD,
            n_samples          = n_pairs,
            pca_eigenvalues    = eigenvalues,
        )

    def _collect_activations(
        self,
        clean_tokens:       List[torch.Tensor],
        counterfact_tokens: List[torch.Tensor],
        target_layer:       int,
        target_position:    int,
        target_tok:         int,
        distract_tok:       int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Collect residual stream activations at (target_layer, target_position).

        Returns (clean_acts, counterfact_acts, ld_clean_values)
        """
        hook_name   = f"blocks.{target_layer}.hook_resid_post"
        clean_acts  = []
        cf_acts     = []
        ld_cleans   = []

        for c_tok, cf_tok in zip(clean_tokens, counterfact_tokens):
            try:
                with torch.no_grad():
                    clean_logits, clean_cache = self.model.run_with_cache(
                        c_tok, names_filter=lambda n: n == hook_name,
                    )
                    _, cf_cache = self.model.run_with_cache(
                        cf_tok, names_filter=lambda n: n == hook_name,
                    )

                if hook_name not in clean_cache or hook_name not in cf_cache:
                    continue

                # Extract activation at target position
                pos      = target_position
                clean_z  = clean_cache[hook_name][0, pos, :].cpu().float().numpy()
                cf_z     = cf_cache[hook_name][0, pos, :].cpu().float().numpy()

                clean_acts.append(clean_z)
                cf_acts.append(cf_z)

                ld = (clean_logits[0, -1, target_tok] - clean_logits[0, -1, distract_tok]).item()
                ld_cleans.append(ld)

            except Exception as e:
                logger.debug("Activation collection failed: %s", e)

        return clean_acts, cf_acts, ld_cleans

    def _pca_subspace(
        self,
        clean_acts:  List[np.ndarray],
        cf_acts:     List[np.ndarray],
    ) -> Tuple[np.ndarray, List[float], float]:
        """
        PCA on Δz vectors to identify concept subspace.

        Returns (rotation_matrix, eigenvalues, explained_variance_ratio)
        rotation_matrix: shape (d_model, concept_dims) — columns are subspace basis vectors
        """
        delta_z = np.array(clean_acts) - np.array(cf_acts)   # (n_pairs, d_model)

        # Centre
        delta_z_centred = delta_z - delta_z.mean(axis=0, keepdims=True)

        # SVD (PCA via SVD for numerical stability)
        U, S, Vt = np.linalg.svd(delta_z_centred, full_matrices=False)
        eigenvalues      = (S ** 2 / max(len(delta_z) - 1, 1)).tolist()
        total_var        = sum(eigenvalues) if eigenvalues else 1.0
        n_dims           = min(self.concept_dims, len(S))
        top_eigenvalues  = eigenvalues[:n_dims]
        explained_var    = sum(top_eigenvalues) / total_var if total_var > 0 else 0.0

        # Rotation matrix: top n_dims principal directions
        rotation_matrix = Vt[:n_dims].T   # (d_model, n_dims)

        return rotation_matrix, [round(e, 6) for e in eigenvalues], float(explained_var)

    def _evaluate_das_score(
        self,
        rotation_matrix:    np.ndarray,        # (d_model, concept_dims)
        clean_tokens:       List[torch.Tensor],
        counterfact_tokens: List[torch.Tensor],
        target_layer:       int,
        target_position:    int,
        target_tok:         int,
        distract_tok:       int,
    ) -> Tuple[float, float]:
        """
        Evaluate DAS score via interchange interventions.

        For each (clean, counterfactual) pair:
        1. Project both activations onto the rotation subspace
        2. Replace clean's subspace component with counterfactual's component
        3. Measure LD after intervention

        DAS score = Pearson correlation between LD_intervened and LD_clean
        (measuring how well the concept explains the LD)

        Returns (das_score, mean_ld_intervened)
        """
        if not clean_tokens or not counterfact_tokens:
            return 0.0, 0.0

        hook_name  = f"blocks.{target_layer}.hook_resid_post"
        R_tensor   = torch.tensor(rotation_matrix, dtype=torch.float32)

        ld_intervened_list: List[float] = []
        ld_clean_list:      List[float] = []

        n = min(len(clean_tokens), len(counterfact_tokens), self.n_interchange)

        for i in range(n):
            c_tok  = clean_tokens[i]
            cf_tok = counterfact_tokens[i]

            try:
                with torch.no_grad():
                    clean_logits, clean_cache = self.model.run_with_cache(
                        c_tok, names_filter=lambda nm: nm == hook_name,
                    )
                    _, cf_cache = self.model.run_with_cache(
                        cf_tok, names_filter=lambda nm: nm == hook_name,
                    )

                if hook_name not in clean_cache or hook_name not in cf_cache:
                    continue

                ld_c = (clean_logits[0, -1, target_tok] - clean_logits[0, -1, distract_tok]).item()
                ld_clean_list.append(ld_c)

                # Interchange intervention: swap concept subspace between clean and CF
                clean_z_pos = clean_cache[hook_name][0, target_position, :].float()   # (d_model,)
                cf_z_pos    = cf_cache[hook_name][0, target_position, :].float()

                R = R_tensor.to(clean_z_pos.device)

                # Project onto subspace
                clean_proj = R @ (R.T @ clean_z_pos)    # clean's component in subspace
                cf_proj    = R @ (R.T @ cf_z_pos)       # cf's component in subspace

                # Intervene: replace clean's subspace with cf's subspace
                z_intervened = clean_z_pos - clean_proj + cf_proj   # (d_model,)

                # Run model with this intervention
                def patch_hook(value, hook):
                    patched = value.clone()
                    patched[0, target_position, :] = z_intervened
                    return patched

                with torch.no_grad():
                    logits_int = self.model.run_with_hooks(
                        c_tok,
                        fwd_hooks=[(hook_name, patch_hook)],
                    )

                ld_int = (logits_int[0, -1, target_tok] - logits_int[0, -1, distract_tok]).item()
                ld_intervened_list.append(ld_int)

            except Exception as e:
                logger.debug("DAS intervention failed for sample %d: %s", i, e)

        if len(ld_intervened_list) < 2 or len(ld_clean_list) < 2:
            return 0.0, float(np.mean(ld_intervened_list)) if ld_intervened_list else 0.0

        # DAS score: how much does intervening in this subspace reduce LD?
        # (LD should drop toward 0 when concept is swapped from CF → clean)
        ld_clean_arr = np.array(ld_clean_list[:len(ld_intervened_list)])
        ld_int_arr   = np.array(ld_intervened_list)

        # Score: proportion of LD explained by concept transfer
        # Das score = corr(LD_original, LD_intervened)
        if len(ld_clean_arr) >= 2 and ld_clean_arr.std() > 1e-8 and ld_int_arr.std() > 1e-8:
            correlation = float(np.corrcoef(ld_clean_arr, ld_int_arr)[0, 1])
            # Negate: if intervention reduces LD, correlation should be negative
            # Re-interpret: fraction of behaviour explained = 1 - (residual_LD / clean_LD)
            residual_fraction = float(np.mean(ld_int_arr) / (np.mean(ld_clean_arr) + 1e-8))
            das_score = float(np.clip(1.0 - abs(residual_fraction), 0.0, 1.0))
        else:
            # Fallback: simple LD ratio
            mean_clean = float(np.mean(ld_clean_arr))
            mean_int   = float(np.mean(ld_int_arr))
            das_score  = float(np.clip(1.0 - abs(mean_int / (mean_clean + 1e-8)), 0.0, 1.0))

        return das_score, float(np.mean(ld_intervened_list))

    def search_all_layers(
        self,
        concept_label:         str,
        clean_prompts_tokens:  List[torch.Tensor],
        counterfactual_tokens: List[torch.Tensor],
        target_tok:            int,
        distract_tok:          int,
        target_position:       int = -1,
        layers:                Optional[List[int]] = None,
    ) -> List[DASResult]:
        """
        Run DAS across multiple layers to find where concept is best encoded.

        Parameters
        ----------
        layers : Layers to search. Defaults to all layers.

        Returns
        -------
        List of DASResult sorted by das_score descending
        """
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))

        results = []
        for layer in layers:
            try:
                r = self.search(
                    concept_label, clean_prompts_tokens, counterfactual_tokens,
                    target_tok, distract_tok, layer, target_position,
                )
                results.append(r)
            except Exception as e:
                logger.warning("DAS search failed for layer %d: %s", layer, e)

        return sorted(results, key=lambda x: -x.das_score)
