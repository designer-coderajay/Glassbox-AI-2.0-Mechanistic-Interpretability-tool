"""
Glassbox — Head Composition Analysis
======================================

Quantifies how much transformer attention heads COMPOSE with each other —
i.e., how much the output of one head feeds causally into the queries, keys,
or values of a later head.

This implements the three composition scores from Elhage et al. (2021)
§3.2 "Compositional Scores":

  Q-composition  : ||W_QK^{L2H2}  ·  W_OV^{L1H1}||_F  /  norm
  K-composition  : ||W_QK^{L2H2ᵀ} ·  W_OV^{L1H1}||_F  /  norm
  V-composition  : ||W_OV^{L2H2}   ·  W_OV^{L1H1}||_F  /  norm

where
  W_OV  = W_V  W_O  ∈ ℝ^{d_model × d_model}   (value-out virtual weight)
  W_QK  = W_Q  W_Kᵀ ∈ ℝ^{d_model × d_model}   (query-key virtual weight)

Interpretation
--------------
A large composition score for (L1H1 → L2H2) means:
  Q-composition: L2H2's queries attend more to tokens written by L1H1.
  K-composition: L2H2's keys are influenced by L1H1's output.
  V-composition: L2H2's value computation builds on L1H1's output.

In the IOI circuit, the S-inhibition heads (layer 7-9) Q-compose strongly
with the duplicate-token heads (layers 0-3), as they need to identify the
duplicated name to suppress it.

References
----------
Elhage et al. (2021). "A Mathematical Framework for Transformer Circuits."
    Transformer Circuits Thread.
    https://transformer-circuits.pub/2021/framework/index.html
    §3.2 "Compositional Scores" defines QK and OV composition.

Wang et al. (2022). "Interpretability in the Wild: a Circuit for Indirect
    Object Identification in GPT-2 Small." https://arxiv.org/abs/2211.00593
    Uses composition scores to validate the IOI circuit's causal structure.

Mathematical Notes
------------------
The Frobenius norm is used throughout.  The normalisation denominator is:
    ||A||_F × ||B||_F
This makes the score scale-invariant (in [0, 1] in theory, but practically
can exceed 1 due to structured weight sharing).

All weights are assumed to be the parameter-only matrices (no biases) as
stored by TransformerLens.  Shapes:
    W_Q, W_K  :  [n_heads, d_model, d_head]
    W_V, W_O  :  W_V [n_heads, d_model, d_head],  W_O [n_heads, d_head, d_model]

The virtual weight W_OV for a single head h at layer l:
    W_OV[l,h] = W_V[l,h]ᵀ @ W_O[l,h]   shape [d_model, d_model]
    (reading convention: W_V maps d_model → d_head, W_O maps d_head → d_model)

The virtual weight W_QK for a single head h at layer l:
    W_QK[l,h] = W_Q[l,h] @ W_K[l,h]ᵀ   shape [d_head, d_head]
    — but the composition score uses:
    W_Q[l,h] ∈ ℝ^{d_model × d_head}  (after transposing TransformerLens layout)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = ["HeadCompositionAnalyzer"]

logger = logging.getLogger(__name__)


class HeadCompositionAnalyzer:
    """
    Computes Q/K/V composition scores between pairs of attention heads.

    Parameters
    ----------
    model : HookedTransformer
        TransformerLens model instance.

    Examples
    --------
    >>> from transformer_lens import HookedTransformer
    >>> from glassbox.composition import HeadCompositionAnalyzer
    >>>
    >>> model   = HookedTransformer.from_pretrained("gpt2")
    >>> comp    = HeadCompositionAnalyzer(model)
    >>>
    >>> # Q-composition between (6, 9) → (9, 9) in the IOI circuit
    >>> score = comp.q_composition_score(6, 9, 9, 9)
    >>> print(f"Q-comp (6,9)→(9,9): {score:.4f}")
    >>>
    >>> # Full composition matrix for the circuit
    >>> circuit  = [(9, 9), (9, 6), (10, 0), (7, 3), (8, 6)]
    >>> mat      = comp.composition_matrix(circuit, circuit, kind="q")
    """

    def __init__(self, model) -> None:
        self.model    = model
        self.n_layers = model.cfg.n_layers
        self.n_heads  = model.cfg.n_heads
        self._wov_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._wqk_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # INTERNAL — virtual weight matrices (cached)
    # -----------------------------------------------------------------------

    def _W_OV(self, layer: int, head: int) -> torch.Tensor:
        """W_OV = W_V[l,h]ᵀ @ W_O[l,h]  ∈ ℝ^{d_model × d_model}."""
        key = (layer, head)
        if key not in self._wov_cache:
            blk = self.model.blocks[layer].attn
            W_V = blk.W_V.detach().float()[head]   # [d_model, d_head]
            W_O = blk.W_O.detach().float()[head]   # [d_head,  d_model]
            # W_V: d_model → d_head  then W_O: d_head → d_model
            self._wov_cache[key] = W_V @ W_O       # [d_model, d_model]
        return self._wov_cache[key]

    def _W_Q(self, layer: int, head: int) -> torch.Tensor:
        """W_Q[l,h] ∈ ℝ^{d_model × d_head}."""
        return self.model.blocks[layer].attn.W_Q.detach().float()[head]

    def _W_K(self, layer: int, head: int) -> torch.Tensor:
        """W_K[l,h] ∈ ℝ^{d_model × d_head}."""
        return self.model.blocks[layer].attn.W_K.detach().float()[head]

    @staticmethod
    def _frobenius_norm(t: torch.Tensor) -> float:
        return float(torch.norm(t, p="fro").item())

    # -----------------------------------------------------------------------
    # PUBLIC — three composition scores
    # -----------------------------------------------------------------------

    def q_composition_score(
        self,
        sender_layer: int,
        sender_head:  int,
        recv_layer:   int,
        recv_head:    int,
    ) -> float:
        """
        Q-composition score: how much does (sender) write into (receiver)'s queries?

        Formula (Elhage et al. 2021, §3.2):
            C_Q = ||W_Q^{recv} · W_OV^{sender}||_F
                / (||W_Q^{recv}||_F · ||W_OV^{sender}||_F)

        where W_Q^{recv} ∈ ℝ^{d_model × d_head} and
              W_OV^{sender} ∈ ℝ^{d_model × d_model}.

        Returns float in [0, ∞).  Values >> 1 are unusual and indicate a
        highly aligned write-then-query pathway.
        """
        if recv_layer <= sender_layer:
            return 0.0   # causally impossible: receiver must be later

        W_OV = self._W_OV(sender_layer, sender_head)   # [d_model, d_model]
        W_Q  = self._W_Q(recv_layer,   recv_head)      # [d_model, d_head]

        # W_Q reads from residual stream (W_OV output): W_Q @ W_OV
        product    = W_Q.T @ W_OV                       # [d_head, d_model]
        score      = self._frobenius_norm(product)
        denom      = self._frobenius_norm(W_Q) * self._frobenius_norm(W_OV)
        return score / denom if denom > 0 else 0.0

    def k_composition_score(
        self,
        sender_layer: int,
        sender_head:  int,
        recv_layer:   int,
        recv_head:    int,
    ) -> float:
        """
        K-composition score: how much does (sender) write into (receiver)'s keys?

        Formula (Elhage et al. 2021, §3.2):
            C_K = ||W_K^{recv} · W_OV^{sender}||_F
                / (||W_K^{recv}||_F · ||W_OV^{sender}||_F)
        """
        if recv_layer <= sender_layer:
            return 0.0

        W_OV = self._W_OV(sender_layer, sender_head)
        W_K  = self._W_K(recv_layer,   recv_head)      # [d_model, d_head]

        product = W_K.T @ W_OV                         # [d_head, d_model]
        score   = self._frobenius_norm(product)
        denom   = self._frobenius_norm(W_K) * self._frobenius_norm(W_OV)
        return score / denom if denom > 0 else 0.0

    def v_composition_score(
        self,
        sender_layer: int,
        sender_head:  int,
        recv_layer:   int,
        recv_head:    int,
    ) -> float:
        """
        V-composition score: how much does (sender) feed into (receiver)'s values?

        Formula (Elhage et al. 2021, §3.2):
            C_V = ||W_OV^{recv} · W_OV^{sender}||_F
                / (||W_OV^{recv}||_F · ||W_OV^{sender}||_F)
        """
        if recv_layer <= sender_layer:
            return 0.0

        W_OV_s = self._W_OV(sender_layer, sender_head)
        W_OV_r = self._W_OV(recv_layer,   recv_head)

        product = W_OV_r @ W_OV_s                      # [d_model, d_model]
        score   = self._frobenius_norm(product)
        denom   = self._frobenius_norm(W_OV_r) * self._frobenius_norm(W_OV_s)
        return score / denom if denom > 0 else 0.0

    # -----------------------------------------------------------------------
    # HIGHER-LEVEL API
    # -----------------------------------------------------------------------

    def composition_matrix(
        self,
        senders:   List[Tuple[int, int]],
        receivers: List[Tuple[int, int]],
        kind:      str = "q",
    ) -> Dict:
        """
        Compute a full composition score matrix between sender and receiver heads.

        Parameters
        ----------
        senders   : List of (layer, head) tuples for the sending heads.
        receivers : List of (layer, head) tuples for the receiving heads.
        kind      : "q", "k", or "v"  (which composition score to compute).

        Returns
        -------
        dict with keys:
          "matrix"    : np.ndarray  [len(receivers), len(senders)]
          "senders"   : List[str]   — "L{l}H{h}" labels
          "receivers" : List[str]   — "L{l}H{h}" labels
          "kind"      : str
          "top_pairs" : List[dict]  — top 10 pairs by score
        """
        score_fn = {
            "q": self.q_composition_score,
            "k": self.k_composition_score,
            "v": self.v_composition_score,
        }.get(kind.lower())
        if score_fn is None:
            raise ValueError(f"kind must be 'q', 'k', or 'v', got {kind!r}")

        n_recv  = len(receivers)
        n_send  = len(senders)
        matrix  = np.zeros((n_recv, n_send), dtype=np.float32)
        pairs   = []

        for i, (rl, rh) in enumerate(receivers):
            for j, (sl, sh) in enumerate(senders):
                score = score_fn(sl, sh, rl, rh)
                matrix[i, j] = score
                pairs.append({
                    "sender":   f"L{sl:02d}H{sh:02d}",
                    "receiver": f"L{rl:02d}H{rh:02d}",
                    "score":    float(score),
                    "kind":     kind,
                })

        pairs.sort(key=lambda p: p["score"], reverse=True)

        return {
            "matrix":    matrix,
            "senders":   [f"L{l:02d}H{h:02d}" for l, h in senders],
            "receivers": [f"L{l:02d}H{h:02d}" for l, h in receivers],
            "kind":      kind,
            "top_pairs": pairs[:10],
        }

    def full_circuit_composition(
        self,
        circuit:     List[Tuple[int, int]],
        kind:        str  = "q",
        min_score:   float = 0.05,
    ) -> Dict:
        """
        Compute all pairwise composition scores within a circuit.

        Internally calls composition_matrix(circuit, circuit) and filters
        to causally valid pairs (sender_layer < receiver_layer) with score
        above min_score.

        Parameters
        ----------
        circuit   : circuit heads as (layer, head) tuples.
        kind      : "q", "k", or "v".
        min_score : minimum score to include in significant_edges.

        Returns
        -------
        dict with keys:
          "matrix"            : np.ndarray  [n_heads, n_heads]
          "head_labels"       : List[str]
          "significant_edges" : List[dict]  — pairs with score >= min_score
          "kind"              : str
          "mean_score"        : float
          "max_score"         : float
        """
        result  = self.composition_matrix(circuit, circuit, kind=kind)
        matrix  = result["matrix"]
        labels  = result["senders"]

        n = len(circuit)
        significant = []
        for i in range(n):
            for j in range(n):
                sl, _ = circuit[j]   # sender
                rl, _ = circuit[i]   # receiver
                if rl > sl and matrix[i, j] >= min_score:
                    significant.append({
                        "sender":   labels[j],
                        "receiver": labels[i],
                        "score":    float(matrix[i, j]),
                        "kind":     kind,
                    })

        significant.sort(key=lambda e: e["score"], reverse=True)

        return {
            "matrix":            matrix,
            "head_labels":       labels,
            "significant_edges": significant,
            "kind":              kind,
            "mean_score":        float(np.mean(matrix)),
            "max_score":         float(np.max(matrix)),
        }

    def all_composition_scores(
        self,
        circuit: List[Tuple[int, int]],
        min_score: float = 0.05,
    ) -> Dict:
        """
        Compute Q, K, and V composition matrices for a circuit simultaneously.

        Returns a combined dict with keys "q", "k", "v" each containing the
        result of full_circuit_composition(), plus a "combined_edges" list
        that merges significant edges across all three score types.

        Parameters
        ----------
        circuit    : List of (layer, head) tuples.
        min_score  : Minimum score to include in significant_edges.

        Returns
        -------
        dict with keys "q", "k", "v", "combined_edges", "head_labels".
        """
        q_result = self.full_circuit_composition(circuit, kind="q", min_score=min_score)
        k_result = self.full_circuit_composition(circuit, kind="k", min_score=min_score)
        v_result = self.full_circuit_composition(circuit, kind="v", min_score=min_score)

        combined: Dict[str, Dict] = {}
        for edge in q_result["significant_edges"]:
            key = (edge["sender"], edge["receiver"])
            combined[key] = {"sender": edge["sender"], "receiver": edge["receiver"],
                              "q": edge["score"], "k": 0.0, "v": 0.0}
        for edge in k_result["significant_edges"]:
            key = (edge["sender"], edge["receiver"])
            if key in combined:
                combined[key]["k"] = edge["score"]
            else:
                combined[key] = {"sender": edge["sender"], "receiver": edge["receiver"],
                                  "q": 0.0, "k": edge["score"], "v": 0.0}
        for edge in v_result["significant_edges"]:
            key = (edge["sender"], edge["receiver"])
            if key in combined:
                combined[key]["v"] = edge["score"]
            else:
                combined[key] = {"sender": edge["sender"], "receiver": edge["receiver"],
                                  "q": 0.0, "k": 0.0, "v": edge["score"]}

        combined_list = sorted(combined.values(),
                               key=lambda e: e["q"] + e["k"] + e["v"], reverse=True)

        return {
            "q":              q_result,
            "k":              k_result,
            "v":              v_result,
            "combined_edges": combined_list,
            "head_labels":    q_result["head_labels"],
        }
