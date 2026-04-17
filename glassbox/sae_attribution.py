"""
Glassbox — SAE Feature Attribution
====================================

Sparse Autoencoder (SAE) feature-level attribution for transformer circuits.

This module bridges circuit-level mechanistic interpretability (attribution
patching, EAP) with feature-level interpretability (SAEs, superposition).

The key insight: attribution patching tells you WHICH heads matter.
SAE attribution tells you WHAT FEATURES those heads write to the residual
stream, and how much each feature contributes to the final prediction.

References
----------
Bloom et al. (2024). "Open Source Sparse Autoencoders for all Residual Stream
    Layers of GPT2-Small." https://www.neuronpedia.org/gpt2-small
    Pre-trained residual-stream SAEs released as the ``sae-lens`` library.
    https://github.com/jbloomAus/SAELens

Bricken et al. (2023). "Towards Monosemanticity: Decomposing Language Models
    With Dictionary Learning." Transformer Circuits Thread.
    https://transformer-circuits.pub/2023/monosemanticity/index.html
    Establishes SAE features as the right unit for mechanistic interpretability.

Cunningham et al. (2023). "Sparse Autoencoders Find Highly Interpretable
    Features in Language Models." https://arxiv.org/abs/2309.08600
    SAE training methodology, feature interpretability evaluations.

Elhage et al. (2022). "Toy Models of Superposition."
    Transformer Circuits Thread.
    https://transformer-circuits.pub/2022/toy_model/index.html
    Superposition hypothesis: features are compressed into a lower-dimensional
    residual stream; SAEs recover the ground-truth feature basis.

Algorithm
---------
Given a circuit identified by Glassbox (attention heads + MLP layers), and a
Sparse Autoencoder trained on the residual stream at layer l:

  1. Run clean forward pass, cache residual stream at each layer.
  2. For each layer l, encode residual_post[l] through the SAE:
         f_acts = ReLU(W_enc @ (resid - b_dec) + b_enc)   [n_features]
     This gives the sparse feature activation vector.
  3. Score each active feature f by its logit-difference contribution:
         score(f) = f_acts[f]  ×  (W_dec[f] @ unembed_dir)
     where unembed_dir = W_U[:,target] - W_U[:,distractor].
     Positive → pushes model toward target token.
     Negative → pushes model toward distractor.
  4. Aggregate across layers weighted by each layer's logit shift.

APPROXIMATION NOTE (disclosed)
--------------------------------
The feature-level LD contribution applies W_dec @ unembed_dir without
passing through the final layer norm.  This is a first-order approximation
identical in spirit to the logit-lens direct-effect approximation.  Relative
feature rankings are preserved; absolute values are directional.

The SAE only covers the RESIDUAL STREAM, not individual head outputs.
To attribute per-head SAE features, decompose the head output
  head_out = z @ W_O  [d_model]
through the SAE.  This requires head-output SAEs (less available) or the
linearity approximation that head contributions are additive.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = ["SAEFeatureAttributor"]

logger = logging.getLogger(__name__)

_SAELENS_AVAILABLE: Optional[bool] = None


def _check_saelens() -> bool:
    global _SAELENS_AVAILABLE
    if _SAELENS_AVAILABLE is None:
        try:
            import sae_lens  # noqa: F401
            _SAELENS_AVAILABLE = True
        except ImportError:
            _SAELENS_AVAILABLE = False
    return _SAELENS_AVAILABLE


# ---------------------------------------------------------------------------
# Supported SAE releases (model_name -> release id in sae-lens hub)
# Add more as Joseph Bloom releases them.
# ---------------------------------------------------------------------------
_SAE_RELEASES: Dict[str, str] = {
    "gpt2":        "gpt2-small-res-jb",
    "gpt2-small":  "gpt2-small-res-jb",
}

# Map layer index to hook-point string in the sae-lens release
_HOOK_TEMPLATE = "blocks.{layer}.hook_resid_post"


class _CustomSAE:
    """
    Lightweight SAE wrapper for custom checkpoints (no sae-lens dependency).

    Mirrors the encode/decode interface used by sae-lens SAE objects so that
    the rest of SAEFeatureAttributor can treat both modes identically.

    Attributes
    ----------
    W_enc : Tensor (n_features, d_model)
    b_enc : Tensor (n_features,)
    W_dec : Tensor (d_model, n_features)
    b_dec : Tensor (d_model,)
    """

    def __init__(
        self,
        W_enc: "torch.Tensor",
        b_enc: "torch.Tensor",
        W_dec: "torch.Tensor",
        b_dec: "torch.Tensor",
    ) -> None:
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Encode residual stream → feature activations.

        x : (..., d_model)
        returns: (..., n_features)  — ReLU(x @ W_enc.T + b_enc)
        """
        return torch.relu(x @ self.W_enc.T + self.b_enc)

    def decode(self, f: "torch.Tensor") -> "torch.Tensor":
        """
        Decode feature activations → reconstructed residual stream.

        f : (..., n_features)
        returns: (..., d_model)
        """
        return f @ self.W_dec.T + self.b_dec

    @property
    def W_dec_tensor(self) -> "torch.Tensor":
        """W_dec as (n_features, d_model) for dot-product scoring."""
        return self.W_dec.T


class SAEFeatureAttributor:
    """
    Attributes transformer circuit components to sparse autoencoder features.

    This class decomposes residual stream activations at each layer into
    their SAE feature basis, then scores each feature by its contribution
    to the target vs. distractor logit difference.

    Supports two SAE loading modes:

    **Mode 1 — Hub release (default):** Downloads a pre-trained SAE from the
    sae-lens hub. Requires ``pip install sae-lens``. Works out-of-the-box for
    GPT-2 small (gpt2, gpt2-small). Pass ``sae_release="..."`` for other models.

    **Mode 2 — Custom checkpoint (v3.1.0):** Load your own trained SAE weights
    from a ``.pt`` file. Pass ``sae_path="/path/to/sae.pt"``. The file must
    contain a dict with keys::

        encoder_weight  : (n_features, d_model)   — W_enc
        encoder_bias    : (n_features,)            — b_enc
        decoder_weight  : (d_model, n_features)    — W_dec
        decoder_bias    : (d_model,)               — b_dec

    This mode does NOT require sae-lens. Useful for:
      - Fine-tuned or custom model SAEs
      - SAEs trained with non-standard architectures
      - SAEs from non-public releases

    Parameters
    ----------
    model : HookedTransformer
        TransformerLens model instance.
    sae_release : str, optional
        SAE release name on the sae-lens hub (e.g. "gpt2-small-res-jb").
        Auto-detected from the model name if not specified.
        Mutually exclusive with ``sae_path``.
    sae_path : str or Path, optional
        Path to a custom SAE checkpoint ``.pt`` file.
        If supplied, ``sae_release`` is ignored and sae-lens is not required.
        A single checkpoint is applied to ALL queried layers; for per-layer
        checkpoints, pass a dict ``{layer: path}`` to ``sae_path``.
    device : str or torch.device, optional
        Device to run the SAE on.  Defaults to the model's device.

    Raises
    ------
    ImportError
        If Mode 1 is used and ``sae-lens`` is not installed.
    ValueError
        If neither a valid release nor a valid sae_path is provided.
    FileNotFoundError
        If ``sae_path`` does not exist on disk.

    Examples
    --------
    >>> # Mode 1 — hub release
    >>> sfa = SAEFeatureAttributor(model)
    >>>
    >>> # Mode 2 — custom checkpoint (single .pt for all layers)
    >>> sfa = SAEFeatureAttributor(model, sae_path="./my_sae_layer9.pt")
    >>>
    >>> # Mode 2 — per-layer checkpoints
    >>> sfa = SAEFeatureAttributor(model, sae_path={9: "./sae_l9.pt", 10: "./sae_l10.pt"})
    >>>
    >>> tokens = model.to_tokens("When Mary and John went to the store, John gave a drink to")
    >>> result = sfa.attribute(tokens, " Mary", " John", layers=[9, 10, 11])
    >>> print(result["top_features"][:5])
    """

    def __init__(
        self,
        model,
        sae_release: Optional[str] = None,
        sae_path = None,          # str | Path | Dict[int, str|Path] | None
        device: Optional[str] = None,
    ) -> None:
        import os
        from pathlib import Path

        self.model  = model
        self.device = device or str(next(model.parameters()).device)
        self._sae_cache: Dict[int, object] = {}

        # ── Mode 2: custom checkpoint ──────────────────────────────────────
        if sae_path is not None:
            # Normalise to Dict[int, Path]
            if isinstance(sae_path, dict):
                self._custom_paths: Optional[Dict] = {
                    int(k): Path(v) for k, v in sae_path.items()
                }
                self._custom_path_single: Optional["Path"] = None
            else:
                self._custom_paths = None
                self._custom_path_single = Path(sae_path)
                if not self._custom_path_single.exists():
                    raise FileNotFoundError(
                        f"SAE checkpoint not found: {self._custom_path_single}\n"
                        "Expected keys: encoder_weight, encoder_bias, "
                        "decoder_weight, decoder_bias"
                    )

            self.sae_release   = None
            self._use_custom   = True
            logger.info("SAEFeatureAttributor: using custom checkpoint(s) from %s", sae_path)
            return

        # ── Mode 1: hub release ────────────────────────────────────────────
        self._use_custom = False
        self._custom_paths = None
        self._custom_path_single = None

        if not _check_saelens():
            raise ImportError(
                "sae-lens is required for SAE feature attribution.\n"
                "Install it with:  pip install sae-lens\n"
                "See https://github.com/jbloomAus/SAELens\n"
                "Or pass sae_path='.../my_sae.pt' to use a custom checkpoint "
                "(no sae-lens required)."
            )

        model_name_lower = getattr(model.cfg, "model_name", "").lower()
        if sae_release is None:
            sae_release = _SAE_RELEASES.get(model_name_lower)
            if sae_release is None:
                raise ValueError(
                    f"No default SAE release found for model '{model_name_lower}'.\n"
                    f"Supported models: {list(_SAE_RELEASES.keys())}\n"
                    f"Pass sae_release='...' explicitly, or sae_path='...' for a "
                    f"custom checkpoint.\n"
                    f"See https://github.com/jbloomAus/SAELens for available releases."
                )
        self.sae_release = sae_release

    # -----------------------------------------------------------------------
    # INTERNAL — SAE loading (hub or custom)
    # -----------------------------------------------------------------------

    def _get_sae(self, layer: int):
        """Load (and cache) the SAE for a given layer."""
        if layer not in self._sae_cache:
            if self._use_custom:
                self._sae_cache[layer] = self._load_custom_sae(layer)
            else:
                from sae_lens import SAE
                hook_point = _HOOK_TEMPLATE.format(layer=layer)
                logger.info(
                    "Loading SAE: release=%s  hook_point=%s", self.sae_release, hook_point
                )
                sae, _cfg, _sparsity = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=hook_point,
                    device=self.device,
                )
                self._sae_cache[layer] = sae
        return self._sae_cache[layer]

    def _load_custom_sae(self, layer: int) -> "_CustomSAE":
        """Load a custom SAE checkpoint from disk and wrap it in _CustomSAE."""
        from pathlib import Path

        if self._custom_paths is not None:
            if layer not in self._custom_paths:
                raise ValueError(
                    f"Custom SAE checkpoint not provided for layer {layer}.\n"
                    f"Available layers: {list(self._custom_paths.keys())}"
                )
            path = self._custom_paths[layer]
        else:
            path = self._custom_path_single  # type: ignore[assignment]

        logger.info("Loading custom SAE from %s for layer %d", path, layer)
        ckpt = torch.load(path, map_location=self.device)

        required = {"encoder_weight", "encoder_bias", "decoder_weight", "decoder_bias"}
        missing  = required - set(ckpt.keys())
        if missing:
            raise ValueError(
                f"Custom SAE checkpoint at '{path}' is missing keys: {missing}\n"
                f"Required: {required}\nFound: {set(ckpt.keys())}"
            )

        return _CustomSAE(
            W_enc = ckpt["encoder_weight"].to(self.device).float(),
            b_enc = ckpt["encoder_bias"].to(self.device).float(),
            W_dec = ckpt["decoder_weight"].to(self.device).float(),
            b_dec = ckpt["decoder_bias"].to(self.device).float(),
        )

    # -----------------------------------------------------------------------
    # CORE API
    # -----------------------------------------------------------------------

    def attribute(
        self,
        tokens:           torch.Tensor,
        target_token:     str,
        distractor_token: str,
        layers:           Optional[List[int]] = None,
        top_k_per_layer:  int = 20,
        top_k_global:     int = 50,
    ) -> Dict:
        """
        Decompose the residual stream at each specified layer into SAE features
        and score each feature by its logit-difference contribution.

        Parameters
        ----------
        tokens : torch.Tensor  [1, seq_len]
            Tokenised prompt.
        target_token : str
            Correct next-token string (e.g. " Mary").
        distractor_token : str
            Distractor token string (e.g. " John").
        layers : List[int], optional
            Which layers to analyse.  Defaults to the last 4 layers.
        top_k_per_layer : int
            Return this many top features per layer.
        top_k_global : int
            Return this many top features globally (ranked by |score|).

        Returns
        -------
        dict with keys:
          "layer_features"  : { layer: List[FeatureRecord] }
              Each FeatureRecord = {feature_id, activation, ld_contribution,
                                    direction: "target"|"distractor"|"neutral"}
          "top_features"    : List[FeatureRecord]  — global top_k_global
          "layer_logit_diffs" : { layer: float }   — LD at each layer (logit lens)
          "n_active_per_layer": { layer: int }      — non-zero features
          "sae_release"     : str
          "layers_analysed" : List[int]
          "target_token"    : str
          "distractor_token": str
        """
        n_layers = self.model.cfg.n_layers
        if layers is None:
            # Default: final 4 layers (where name-mover heads typically live)
            layers = list(range(max(0, n_layers - 4), n_layers))

        # Token IDs ----------------------------------------------------------------
        target_id     = self.model.to_single_token(target_token)
        distractor_id = self.model.to_single_token(distractor_token)

        W_U         = self.model.W_U.detach().float()                    # [d_model, d_vocab]
        unembed_dir = (W_U[:, target_id] - W_U[:, distractor_id]).to(self.device)  # [d_model]

        # Single forward pass — cache residual_post at requested layers ---------
        hook_names = {_HOOK_TEMPLATE.format(layer=l) for l in layers}
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda name: name in hook_names,
            )

        # ── SAE decomposition per layer ─────────────────────────────────────
        layer_features:    Dict[int, List[Dict]] = {}
        layer_logit_diffs: Dict[int, float]      = {}
        n_active:          Dict[int, int]        = {}
        all_features:      List[Dict]            = []

        for layer in layers:
            sae = self._get_sae(layer)

            # Residual stream at last token position, shape [d_model]
            resid = cache[_HOOK_TEMPLATE.format(layer=layer)][0, -1].float().to(self.device)

            # Encode through SAE  → feature_acts [n_features], sparse
            with torch.no_grad():
                feature_acts = sae.encode(resid.unsqueeze(0)).squeeze(0)  # [n_features]

            # LD at this layer via logit lens (resid → ln_final → W_U)
            with torch.no_grad():
                r_normed    = self.model.ln_final(resid.cpu().unsqueeze(0)).squeeze(0)
                logits      = (r_normed.float() @ W_U.cpu())
                layer_ld    = float(logits[target_id].item() - logits[distractor_id].item())
            layer_logit_diffs[layer] = layer_ld

            # Score each active feature -------------------------------------------
            # score(f) = f_acts[f] × (W_dec[f] @ unembed_dir)
            # W_dec shape: (n_features, d_model) for both hub SAEs and _CustomSAE
            if isinstance(sae, _CustomSAE):
                # _CustomSAE stores W_dec as (d_model, n_features); .T → (n_features, d_model)
                W_dec = sae.W_dec.T.detach().float()
            else:
                W_dec = sae.W_dec.detach().float()   # sae-lens: already (n_features, d_model)
            unembed_dir_dev = unembed_dir.to(W_dec.device)
            feature_ld_contribs = feature_acts * (W_dec @ unembed_dir_dev)  # [n_features]

            # Keep only active features (activation > 0)
            active_mask    = feature_acts > 0
            n_active[layer] = int(active_mask.sum().item())

            active_ids     = active_mask.nonzero(as_tuple=True)[0].tolist()
            active_acts    = feature_acts[active_mask].tolist()
            active_contribs = feature_ld_contribs[active_mask].tolist()

            records = []
            for feat_id, act, contrib in zip(active_ids, active_acts, active_contribs):
                direction = (
                    "target"      if contrib >  0.01 else
                    "distractor"  if contrib < -0.01 else
                    "neutral"
                )
                records.append({
                    "layer":           layer,
                    "feature_id":      feat_id,
                    "activation":      float(act),
                    "ld_contribution": float(contrib),
                    "direction":       direction,
                    "neuronpedia_url": (
                        f"https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feat_id}"
                        if (self.sae_release and "gpt2-small" in self.sae_release) else None
                    ),
                })

            # Sort by |contribution| descending, keep top_k_per_layer
            records.sort(key=lambda r: abs(r["ld_contribution"]), reverse=True)
            layer_features[layer] = records[:top_k_per_layer]
            all_features.extend(records)

            logger.info(
                "Layer %d: n_active=%d  LD=%.3f  top feature contrib=%.4f",
                layer, n_active[layer], layer_ld,
                records[0]["ld_contribution"] if records else 0.0,
            )

        # Global top-k
        all_features.sort(key=lambda r: abs(r["ld_contribution"]), reverse=True)

        return {
            "layer_features":     layer_features,
            "top_features":       all_features[:top_k_global],
            "layer_logit_diffs":  layer_logit_diffs,
            "n_active_per_layer": n_active,
            "sae_release":        self.sae_release,
            "layers_analysed":    layers,
            "target_token":       target_token,
            "distractor_token":   distractor_token,
        }

    def attribute_circuit_heads(
        self,
        circuit:          List[Tuple[int, int]],
        tokens:           torch.Tensor,
        target_token:     str,
        distractor_token: str,
        top_k_per_head:   int = 10,
    ) -> Dict:
        """
        SAE attribution scoped to circuit heads.

        For each head (layer, head_idx) in the circuit, decomposes that head's
        contribution to the residual stream through the SAE at its layer, then
        scores each active feature by its LD contribution.

        Formula (head-level SAE attribution, linear approximation)
        -----------------------------------------------------------
        head_out(l,h) = z[l,h,-1] @ W_O[l,h]   [d_model]

        NOTE: This uses the head output vector directly, NOT the full residual
        stream.  We pass head_out through the SAE encoder:
            f_acts(l,h) = ReLU(W_enc @ head_out + b_enc)
        and score:
            score(f) = f_acts(l,h)[f] × (W_dec[f] @ unembed_dir)

        This is a LINEAR APPROXIMATION because:
          (a) The SAE was trained on the full residual stream (sum of all heads
              + MLP + embedding), not on individual head outputs.
          (b) Applying the SAE encoder to a single head output ignores the
              operating point of the other components.
        The approximation is tightest when the target head has large output
        magnitude relative to the other components.

        Use attribute() for exact residual-stream SAE decomposition.
        Use this method to compare which heads most activate which features.

        Parameters
        ----------
        circuit : List[Tuple[int, int]]
            List of (layer, head) tuples from analyze() or minimum_faithful_circuit().
        tokens : torch.Tensor  [1, seq_len]
        target_token, distractor_token : str
        top_k_per_head : int

        Returns
        -------
        dict with keys:
          "head_features"  : { "(l,h)": List[FeatureRecord] }
          "top_features"   : List[FeatureRecord]  — global top
          "sae_release"    : str
        """
        target_id     = self.model.to_single_token(target_token)
        distractor_id = self.model.to_single_token(distractor_token)
        W_U           = self.model.W_U.detach().float()
        unembed_dir   = (W_U[:, target_id] - W_U[:, distractor_id]).to(self.device)

        # Cache z (head output pre-W_O) for all requested layers
        unique_layers = list({l for l, _ in circuit})
        hook_names    = {f"blocks.{l}.attn.hook_z" for l in unique_layers}
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens, names_filter=lambda name: name in hook_names,
            )

        head_features: Dict[str, List[Dict]] = {}
        all_records:   List[Dict]            = []

        for layer, head_idx in circuit:
            sae   = self._get_sae(layer)
            z     = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head_idx].float()  # [d_head]
            W_O   = self.model.blocks[layer].attn.W_O.detach().float()             # [n_heads, d_head, d_model]
            head_out = z @ W_O[head_idx]                                           # [d_model]
            head_out_dev = head_out.to(self.device)

            with torch.no_grad():
                f_acts = sae.encode(head_out_dev.unsqueeze(0)).squeeze(0)          # [n_features]

            # BUG FIX 1: _CustomSAE stores W_dec as (d_model, n_features); hub SAEs
            # store it as (n_features, d_model).  We always need (n_features, d_model)
            # so that W_dec @ unembed_dir[d_model] → (n_features,) matches f_acts shape.
            if isinstance(sae, _CustomSAE):
                W_dec = sae.W_dec.T.detach().float()   # (d_model, n_features).T → (n_features, d_model)
            else:
                W_dec = sae.W_dec.detach().float()     # sae-lens: already (n_features, d_model)
            ud_dev  = unembed_dir.to(W_dec.device)
            contribs = f_acts * (W_dec @ ud_dev)                                   # [n_features]

            active_mask = f_acts > 0
            records = []
            for fid, act, contrib in zip(
                active_mask.nonzero(as_tuple=True)[0].tolist(),
                f_acts[active_mask].tolist(),
                contribs[active_mask].tolist(),
            ):
                records.append({
                    "layer":           layer,
                    "head":            head_idx,
                    "feature_id":      fid,
                    "activation":      float(act),
                    "ld_contribution": float(contrib),
                    "direction":       (
                        "target"     if contrib >  0.01 else
                        "distractor" if contrib < -0.01 else
                        "neutral"
                    ),
                    # BUG FIX 2: self.sae_release is None in custom-SAE mode;
                    # guard against TypeError from "x in None".
                    "neuronpedia_url": (
                        f"https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{fid}"
                        if (self.sae_release and "gpt2-small" in self.sae_release) else None
                    ),
                })

            records.sort(key=lambda r: abs(r["ld_contribution"]), reverse=True)
            head_features[str((layer, head_idx))] = records[:top_k_per_head]
            all_records.extend(records)

        all_records.sort(key=lambda r: abs(r["ld_contribution"]), reverse=True)
        return {
            "head_features": head_features,
            "top_features":  all_records[:50],
            "sae_release":   self.sae_release,
            "note": (
                "head_features uses a linear approximation (SAE applied to isolated "
                "head output, not full residual stream).  Use attribute() for exact "
                "residual-stream decomposition."
            ),
        }
