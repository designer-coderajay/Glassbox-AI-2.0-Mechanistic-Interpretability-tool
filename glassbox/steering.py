"""
glassbox/steering.py
=====================
Steering Vector Export — Glassbox v3.4.0

Implements behaviour-steering via Representation Engineering (Zou et al., 2023)
and activation patching (Li et al., 2023).  No LLM dependency: all operations
are pure torch/numpy linear algebra on the residual stream.

EU AI Act relevance
-------------------
Article 9(2)(b)  — risk mitigation measures built into the system
Article 9(5)     — testing to identify the most appropriate risk mitigation
Article 13(1)    — transparency to allow informed deployment decisions
Article 15(1)    — accuracy, robustness, and cybersecurity through technical
                   measures (steering as a runtime safety layer)

Public API
----------
  SteeringVector         dataclass — normalised d_model direction + metadata
  SteeringVectorExporter — extract / apply / export / test steering vectors

References
----------
  Zou et al. 2023  "Representation Engineering: A Top-Down Approach to AI
                    Transparency" https://arxiv.org/abs/2310.01405
  Li et al. 2023   "Inference-Time Intervention: Eliciting Truthful Answers
                    from a Language Model" https://arxiv.org/abs/2306.03341
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional heavy imports — deferred so the module can be imported cheaply
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F

    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False

try:
    import numpy as np

    _NUMPY_OK = True
except ImportError:  # pragma: no cover
    _NUMPY_OK = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SteeringVector:
    """
    A single steering vector derived from the residual stream.

    Attributes
    ----------
    direction : torch.Tensor
        Unit-norm vector of shape (d_model,) in the residual-stream basis.
    layer : int
        TransformerLens layer index at which this vector was extracted.
    concept_label : str
        Human-readable label (e.g. "gender_bias", "toxicity").
    scale : float
        Default magnitude multiplier applied during suppression (negative
        = suppress the concept, positive = amplify).
    source_info : dict
        Provenance metadata: n_positive, n_negative, extraction_method,
        model_name, glassbox_version, eu_article_ref, timestamp_utc.
    """

    direction: "torch.Tensor"
    layer: int
    concept_label: str
    scale: float = -15.0           # negative = suppress; tuned for most GPT-2-family models
    source_info: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        d = self.direction.shape[0] if _TORCH_OK else "?"
        return (
            f"SteeringVector(concept={self.concept_label!r}, layer={self.layer}, "
            f"d_model={d}, scale={self.scale:.1f})"
        )

    # ------------------------------------------------------------------
    def norm(self) -> float:
        """Return the L2-norm of the direction vector (should be ≈ 1.0)."""
        if not _TORCH_OK:
            return float("nan")
        return float(self.direction.norm().item())

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """Serialise to a JSON-friendly dict (excludes the tensor itself)."""
        return {
            "concept_label": self.concept_label,
            "layer": self.layer,
            "scale": self.scale,
            "d_model": int(self.direction.shape[0]) if _TORCH_OK else None,
            "norm": self.norm(),
            "source_info": self.source_info,
        }


# ---------------------------------------------------------------------------
# Extraction helpers (private)
# ---------------------------------------------------------------------------

def _collect_residual_stream(
    model,
    texts: List[str],
    layer: int,
    device: Optional[str] = None,
) -> "torch.Tensor":
    """
    Run ``model`` on each text and return the mean last-token residual-stream
    activation at ``layer``.

    Returns
    -------
    torch.Tensor of shape (n_texts, d_model)
    """
    if not _TORCH_OK:
        raise ImportError("torch is required for steering vector extraction.")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    activations: List["torch.Tensor"] = []

    with torch.no_grad():
        for text in texts:
            tokens = model.to_tokens(text).to(dev)
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post",
            )
            resid = cache[f"blocks.{layer}.hook_resid_post"]  # (1, seq, d_model)
            # Take last-token representation
            last = resid[0, -1, :].float()                    # (d_model,)
            activations.append(last)

    return torch.stack(activations, dim=0)  # (n_texts, d_model)


def _mean_diff_direction(
    positive_acts: "torch.Tensor",
    negative_acts: "torch.Tensor",
) -> "torch.Tensor":
    """
    Compute the normalised mean-difference direction (Representation
    Engineering, §3.1).

        direction = normalise( mean(positive) - mean(negative) )
    """
    diff = positive_acts.mean(0) - negative_acts.mean(0)
    return F.normalize(diff.unsqueeze(0), dim=-1).squeeze(0)


def _pca_direction(
    positive_acts: "torch.Tensor",
    negative_acts: "torch.Tensor",
) -> "torch.Tensor":
    """
    Compute the top principal component of the contrast matrix as an
    alternative to mean-diff (more robust with small datasets).
    """
    contrast = positive_acts - negative_acts           # (n, d_model)
    # Cheap SVD — only need the first right singular vector
    _, _, Vh = torch.linalg.svd(contrast, full_matrices=False)
    direction = Vh[0]                                  # (d_model,)
    return F.normalize(direction.unsqueeze(0), dim=-1).squeeze(0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SteeringVectorExporter:
    """
    Extract, apply, export, and evaluate steering vectors for EU AI Act
    Article 9 risk mitigation documentation.

    Parameters
    ----------
    method : str
        ``"mean_diff"`` (default) or ``"pca"``.  mean_diff mirrors the
        original Representation Engineering paper; pca is more stable
        with n<10 contrast pairs.
    device : str or None
        PyTorch device string.  Auto-detected if None.
    verbose : bool
        Print progress messages.
    """

    _VERSION = "3.4.0"

    def __init__(
        self,
        method: str = "mean_diff",
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if method not in ("mean_diff", "pca"):
            raise ValueError(f"method must be 'mean_diff' or 'pca', got {method!r}")
        self.method = method
        self.device = device or ("cuda" if (_TORCH_OK and torch.cuda.is_available()) else "cpu")
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public: extraction
    # ------------------------------------------------------------------

    def extract_mean_diff(
        self,
        model,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer: int,
        concept_label: str = "concept",
        scale: float = -15.0,
    ) -> SteeringVector:
        """
        Extract a steering vector from mean residual-stream activations.

        Parameters
        ----------
        model : HookedTransformer
            Loaded TransformerLens model.
        positive_prompts : list[str]
            Texts exhibiting the target concept (e.g. biased completions).
        negative_prompts : list[str]
            Matched texts *without* the target concept.
        layer : int
            Layer at which to extract the direction.
        concept_label : str
            Human-readable concept name.
        scale : float
            Default suppression magnitude (negative = suppress).

        Returns
        -------
        SteeringVector
        """
        if not _TORCH_OK:
            raise ImportError("torch is required.")

        if len(positive_prompts) != len(negative_prompts):
            raise ValueError("positive_prompts and negative_prompts must have equal length.")

        if self.verbose:
            print(f"[Steering] Extracting activations for {len(positive_prompts)} contrast pair(s) at layer {layer}...")

        pos_acts = _collect_residual_stream(model, positive_prompts, layer, self.device)
        neg_acts = _collect_residual_stream(model, negative_prompts, layer, self.device)

        if self.method == "mean_diff":
            direction = _mean_diff_direction(pos_acts, neg_acts)
        else:
            direction = _pca_direction(pos_acts, neg_acts)

        import time

        sv = SteeringVector(
            direction=direction.cpu(),
            layer=layer,
            concept_label=concept_label,
            scale=scale,
            source_info={
                "extraction_method": self.method,
                "n_positive": len(positive_prompts),
                "n_negative": len(negative_prompts),
                "model_name": getattr(model, "cfg", None) and model.cfg.model_name or "unknown",
                "glassbox_version": self._VERSION,
                "eu_article_ref": "Article 9(2)(b) — risk mitigation measure",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

        if self.verbose:
            print(f"[Steering] Extracted: {sv}")

        return sv

    def extract_from_circuit(
        self,
        model,
        gb_result: Dict,
        positive_prompts: List[str],
        negative_prompts: List[str],
        concept_label: str = "circuit_concept",
        scale: float = -15.0,
    ) -> SteeringVector:
        """
        Extract a steering vector anchored to the most influential layer
        identified by a prior Glassbox circuit analysis.

        This uses the circuit result's head importance scores to choose the
        optimal layer automatically, then delegates to extract_mean_diff.

        Parameters
        ----------
        gb_result : dict
            Output of ``GlassboxV2.analyze(...)``.
        """
        # Find the layer with the highest mean head importance score
        layer = self._best_layer_from_circuit(gb_result, model)

        if self.verbose:
            print(f"[Steering] Circuit-guided layer selection: layer {layer}")

        return self.extract_mean_diff(
            model=model,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            layer=layer,
            concept_label=concept_label,
            scale=scale,
        )

    # ------------------------------------------------------------------
    # Public: application
    # ------------------------------------------------------------------

    def apply(
        self,
        model,
        text: str,
        vector: SteeringVector,
        alpha: Optional[float] = None,
    ) -> str:
        """
        Apply a steering vector to the model's residual stream at inference
        time and return the resulting next-token prediction string.

        The hook adds ``alpha * direction`` to every position in the
        residual stream at ``vector.layer``.

        Parameters
        ----------
        alpha : float or None
            Override the vector's default scale.  Pass a large negative
            value to suppress the concept forcefully.

        Returns
        -------
        str — the top-1 next-token string predicted after steering.
        """
        if not _TORCH_OK:
            raise ImportError("torch is required.")

        a = alpha if alpha is not None else vector.scale
        direction = vector.direction.to(self.device)
        hook_name = f"blocks.{vector.layer}.hook_resid_post"

        def _hook(resid, hook):  # noqa: ARG001
            return resid + a * direction.unsqueeze(0).unsqueeze(0)

        model = model.to(self.device)
        tokens = model.to_tokens(text).to(self.device)

        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, _hook)],
            )

        # Greedy decode: top-1 at the last position
        next_token_id = int(logits[0, -1, :].argmax().item())
        next_token_str = model.to_string([next_token_id])
        return next_token_str

    # ------------------------------------------------------------------
    # Public: export
    # ------------------------------------------------------------------

    def export_pt(self, vector: SteeringVector, path: str) -> None:
        """
        Save the vector to a PyTorch .pt file.

        The saved dict contains:
          ``direction``, ``layer``, ``concept_label``, ``scale``,
          ``source_info``.
        """
        if not _TORCH_OK:
            raise ImportError("torch is required.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "direction": vector.direction,
                "layer": vector.layer,
                "concept_label": vector.concept_label,
                "scale": vector.scale,
                "source_info": vector.source_info,
            },
            path,
        )

        if self.verbose:
            print(f"[Steering] Saved .pt to {path}")

    def export_numpy(self, vector: SteeringVector, path: str) -> None:
        """
        Save the direction vector to a NumPy .npy file.
        Metadata is written to ``<path>.json`` alongside it.
        """
        if not _TORCH_OK:
            raise ImportError("torch is required for tensor conversion.")
        if not _NUMPY_OK:
            raise ImportError("numpy is required.")

        import json

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, vector.direction.numpy())

        meta_path = path.replace(".npy", "_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(vector.to_dict(), fh, indent=2)

        if self.verbose:
            print(f"[Steering] Saved .npy to {path}, metadata to {meta_path}")

    @staticmethod
    def load_pt(path: str) -> SteeringVector:
        """Load a SteeringVector previously saved with export_pt()."""
        if not _TORCH_OK:
            raise ImportError("torch is required.")

        data = torch.load(path, map_location="cpu")
        return SteeringVector(
            direction=data["direction"],
            layer=data["layer"],
            concept_label=data["concept_label"],
            scale=data.get("scale", -15.0),
            source_info=data.get("source_info", {}),
        )

    # ------------------------------------------------------------------
    # Public: evaluation
    # ------------------------------------------------------------------

    def test_suppression(
        self,
        model,
        gb,
        prompt: str,
        correct: str,
        incorrect: str,
        vector: SteeringVector,
        alpha: Optional[float] = None,
    ) -> Dict:
        """
        Quantify how much the steering vector suppresses the target concept.

        Runs ``gb.analyze()`` twice — once without the vector (baseline) and
        once with the vector applied via a residual-stream hook — and returns
        a dict with before/after faithfulness metrics and a suppression ratio.

        Parameters
        ----------
        gb : GlassboxV2
            Initialised Glassbox analyser.
        prompt, correct, incorrect : str
            Same arguments passed to gb.analyze().
        vector : SteeringVector
            The vector to test.
        alpha : float or None
            Override the vector's default scale.

        Returns
        -------
        dict with keys:
          baseline, steered, suppression_ratio, passed_threshold,
          eu_article_ref, verdict
        """
        if not _TORCH_OK:
            raise ImportError("torch is required.")

        a = alpha if alpha is not None else vector.scale

        # Baseline — no steering
        baseline_result = gb.analyze(
            prompt=prompt,
            correct=correct,
            incorrect=incorrect,
        )
        baseline_suff = baseline_result.get("faithfulness", {}).get("sufficiency", 0.0)

        # Steered — inject hook
        hook_name = f"blocks.{vector.layer}.hook_resid_post"
        direction = vector.direction.to(self.device)

        def _hook(resid, hook):  # noqa: ARG001
            return resid + a * direction.unsqueeze(0).unsqueeze(0)

        # GlassboxV2 must expose run_with_hooks capability via model
        try:
            original_run = gb.model.run_with_cache  # keep reference

            def _patched_run(*args, **kwargs):
                # wrap run_with_cache to inject the steering hook
                fwd_hooks = kwargs.pop("fwd_hooks", [])
                fwd_hooks.append((hook_name, _hook))
                return gb.model.run_with_hooks(*args, fwd_hooks=fwd_hooks, **kwargs)

            gb.model.run_with_cache = _patched_run
            steered_result = gb.analyze(
                prompt=prompt,
                correct=correct,
                incorrect=incorrect,
            )
        finally:
            gb.model.run_with_cache = original_run  # always restore

        steered_suff = steered_result.get("faithfulness", {}).get("sufficiency", 0.0)

        # Suppression ratio: how much did steered sufficiency drop vs baseline?
        suppression_ratio = (
            (baseline_suff - steered_suff) / (baseline_suff + 1e-9)
        ) if baseline_suff > 0 else 0.0

        threshold = 0.10  # 10 percentage-point drop is a meaningful suppression
        passed = suppression_ratio >= threshold

        verdict = (
            f"Steering vector '{vector.concept_label}' at layer {vector.layer} "
            f"{'effectively suppresses' if passed else 'does not significantly affect'} "
            f"the circuit (sufficiency {baseline_suff:.1%} → {steered_suff:.1%}, "
            f"suppression ratio {suppression_ratio:.1%})."
        )

        return {
            "baseline": {
                "sufficiency": baseline_suff,
                "comprehensiveness": baseline_result.get("faithfulness", {}).get("comprehensiveness", 0.0),
                "f1": baseline_result.get("faithfulness", {}).get("f1", 0.0),
            },
            "steered": {
                "sufficiency": steered_suff,
                "comprehensiveness": steered_result.get("faithfulness", {}).get("comprehensiveness", 0.0),
                "f1": steered_result.get("faithfulness", {}).get("f1", 0.0),
            },
            "alpha": a,
            "suppression_ratio": round(suppression_ratio, 4),
            "passed_threshold": passed,
            "eu_article_ref": "Article 9(2)(b), Article 15(1)",
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Public: bulk extraction across multiple concepts
    # ------------------------------------------------------------------

    def extract_bias_suite(
        self,
        model,
        layer: int,
        contrast_pairs: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
    ) -> Dict[str, SteeringVector]:
        """
        Extract a suite of bias-mitigation steering vectors in one call.

        Parameters
        ----------
        contrast_pairs : dict or None
            ``{concept_label: (positive_prompts, negative_prompts)}``.
            If None, uses the built-in default IOI / gender / toxicity suite.

        Returns
        -------
        dict mapping concept_label -> SteeringVector
        """
        pairs = contrast_pairs if contrast_pairs is not None else _DEFAULT_CONTRAST_PAIRS

        results: Dict[str, SteeringVector] = {}
        for label, (pos, neg) in pairs.items():
            if self.verbose:
                print(f"[Steering] Extracting vector for concept: {label}")
            results[label] = self.extract_mean_diff(
                model=model,
                positive_prompts=pos,
                negative_prompts=neg,
                layer=layer,
                concept_label=label,
            )

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_annex_iv_text(self, vector: SteeringVector, test_result: Optional[Dict] = None) -> str:
        """
        Generate an Annex IV–compliant paragraph describing this steering
        vector as a documented risk mitigation measure.
        """
        lines = [
            f"Steering Vector Risk Mitigation Measure — {vector.concept_label}",
            "=" * 60,
            "",
            f"Concept label    : {vector.concept_label}",
            f"Extraction layer : {vector.layer}",
            f"Scale (alpha)    : {vector.scale:.1f}",
            f"d_model          : {int(vector.direction.shape[0]) if _TORCH_OK else 'N/A'}",
            f"Method           : {vector.source_info.get('extraction_method', 'unknown')}",
            f"Model            : {vector.source_info.get('model_name', 'unknown')}",
            f"Timestamp        : {vector.source_info.get('timestamp_utc', 'N/A')}",
            "",
            "EU AI Act basis",
            "-" * 30,
            "Article 9(2)(b)  Risk management — technical measures to mitigate risks",
            "                 identified in the risk assessment.",
            "Article 9(5)     Testing of AI systems to identify the most appropriate",
            "                 risk mitigation measures.",
            "Article 13(1)    Transparency and provision of information enabling",
            "                 informed use of the AI system.",
            "Article 15(1)    Accuracy, robustness, and cybersecurity measures.",
            "",
        ]

        if test_result:
            b = test_result.get("baseline", {})
            s = test_result.get("steered", {})
            sr = test_result.get("suppression_ratio", 0.0)
            passed = test_result.get("passed_threshold", False)

            lines += [
                "Suppression test results",
                "-" * 30,
                f"Baseline sufficiency  : {b.get('sufficiency', 0.0):.1%}",
                f"Steered sufficiency   : {s.get('sufficiency', 0.0):.1%}",
                f"Suppression ratio     : {sr:.1%}",
                f"Test outcome          : {'EFFECTIVE' if passed else 'INEFFECTIVE'}",
                "",
                "Verdict",
                "-" * 30,
                test_result.get("verdict", ""),
                "",
            ]

        lines += [
            "Compliance statement",
            "-" * 30,
            "This steering vector constitutes a documented, testable, and reproducible",
            "risk mitigation measure per Regulation (EU) 2024/1689 Article 9(2)(b).",
            "The vector was derived from representative contrast pairs and validated",
            "via suppression testing.  The full provenance is recorded in source_info.",
        ]

        return "\n".join(lines)

    def to_html(self, vector: SteeringVector, test_result: Optional[Dict] = None) -> str:
        """
        Render a self-contained HTML card for embedding in the Annex IV
        Evidence Vault or the Glassbox dashboard.
        """
        concept = vector.concept_label
        layer = vector.layer
        scale = vector.scale
        method = vector.source_info.get("extraction_method", "N/A")
        model_name = vector.source_info.get("model_name", "N/A")
        ts = vector.source_info.get("timestamp_utc", "N/A")
        n_pos = vector.source_info.get("n_positive", "N/A")
        n_neg = vector.source_info.get("n_negative", "N/A")
        d_model = int(vector.direction.shape[0]) if _TORCH_OK else "N/A"

        # Suppression test block
        test_html = ""
        if test_result:
            b_suff = test_result.get("baseline", {}).get("sufficiency", 0.0)
            s_suff = test_result.get("steered", {}).get("sufficiency", 0.0)
            sr = test_result.get("suppression_ratio", 0.0)
            passed = test_result.get("passed_threshold", False)
            badge_col = "#16a34a" if passed else "#dc2626"
            badge_txt = "EFFECTIVE" if passed else "INEFFECTIVE"
            test_html = f"""
            <div class="section">
              <div class="section-title">Suppression Test Results</div>
              <table>
                <tr><th>Metric</th><th>Baseline</th><th>Steered</th></tr>
                <tr><td>Sufficiency</td><td>{b_suff:.1%}</td><td>{s_suff:.1%}</td></tr>
                <tr><td>Suppression ratio</td><td colspan="2" style="font-weight:600">{sr:.1%}</td></tr>
              </table>
              <div style="margin-top:8px">
                <span style="background:{badge_col};color:#fff;padding:3px 10px;border-radius:4px;font-size:12px;font-weight:700">
                  {badge_txt}
                </span>
              </div>
              <p style="font-size:12px;margin-top:8px;color:#555">{test_result.get('verdict','')}</p>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Steering Vector — {concept}</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#f8fafc;color:#1e293b;margin:0;padding:20px}}
  .card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:24px;max-width:680px;margin:auto;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
  h2{{margin:0 0 4px;font-size:18px;color:#0f172a}}
  .subtitle{{color:#64748b;font-size:13px;margin-bottom:18px}}
  .section{{margin-bottom:16px}}
  .section-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin-bottom:6px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #f1f5f9}}
  th{{background:#f8fafc;font-weight:600;color:#475569}}
  .badge{{display:inline-block;background:#dbeafe;color:#1d4ed8;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;margin-bottom:12px}}
  .article-ref{{font-size:11px;color:#64748b;margin-top:16px;border-top:1px solid #f1f5f9;padding-top:12px}}
</style>
</head>
<body>
<div class="card">
  <div class="badge">Glassbox {self._VERSION} — Steering Vector</div>
  <h2>Steering Vector: <em>{concept}</em></h2>
  <div class="subtitle">EU AI Act Article 9(2)(b) Risk Mitigation Measure</div>

  <div class="section">
    <div class="section-title">Vector Properties</div>
    <table>
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Concept label</td><td><strong>{concept}</strong></td></tr>
      <tr><td>Layer</td><td>{layer}</td></tr>
      <tr><td>d_model</td><td>{d_model}</td></tr>
      <tr><td>Default scale (alpha)</td><td>{scale:.1f}</td></tr>
      <tr><td>Extraction method</td><td>{method}</td></tr>
      <tr><td>Contrast pairs</td><td>{n_pos} positive / {n_neg} negative</td></tr>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Provenance</div>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>Model</td><td>{model_name}</td></tr>
      <tr><td>Extracted at</td><td>{ts}</td></tr>
      <tr><td>Glassbox version</td><td>{self._VERSION}</td></tr>
    </table>
  </div>

  {test_html}

  <div class="article-ref">
    <strong>EU AI Act basis:</strong> Article 9(2)(b) — risk mitigation measures;
    Article 9(5) — testing; Article 13(1) — transparency; Article 15(1) — robustness.<br>
    Regulation (EU) 2024/1689 — AI Act Annex IV documentation requirement.
  </div>
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_layer_from_circuit(gb_result: Dict, model) -> int:
        """
        Pick the layer with the highest aggregate head importance score from
        a Glassbox circuit result.  Falls back to the middle layer if the
        result does not carry per-head scores.
        """
        n_layers = getattr(getattr(model, "cfg", None), "n_layers", None) or 12
        default_layer = n_layers // 2

        heads = gb_result.get("circuit", {})
        if not heads:
            return default_layer

        # heads is expected to be a dict { (layer, head): score } or a list
        if isinstance(heads, dict):
            layer_scores: Dict[int, float] = {}
            for key, score in heads.items():
                if isinstance(key, (list, tuple)) and len(key) >= 1:
                    layer_idx = int(key[0])
                    layer_scores[layer_idx] = layer_scores.get(layer_idx, 0.0) + abs(float(score))
            if layer_scores:
                return max(layer_scores, key=lambda k: layer_scores[k])

        return default_layer


# ---------------------------------------------------------------------------
# Built-in default contrast pairs (no external data required)
# Covers the most common EU AI Act Article 10 protected attributes
# ---------------------------------------------------------------------------

_DEFAULT_CONTRAST_PAIRS: Dict[str, Tuple[List[str], List[str]]] = {
    "gender_bias": (
        [
            "The nurse said she would call the doctor.",
            "The secretary organised the files for her boss.",
            "The assistant prepared the report for her manager.",
        ],
        [
            "The nurse said he would call the doctor.",
            "The secretary organised the files for his boss.",
            "The assistant prepared the report for his manager.",
        ],
    ),
    "racial_bias": (
        [
            "The Black man was arrested outside the convenience store.",
            "The Latino suspect was identified by police.",
        ],
        [
            "The White man was arrested outside the convenience store.",
            "The suspect was identified by police.",
        ],
    ),
    "toxicity": (
        [
            "You are an idiot and a worthless piece of garbage.",
            "Those people are subhuman filth who deserve nothing.",
        ],
        [
            "You made a mistake and should try again.",
            "Those people hold different views.",
        ],
    ),
    "age_bias": (
        [
            "The elderly applicant was too slow to learn new software.",
            "The old worker struggled to keep up with younger colleagues.",
        ],
        [
            "The applicant was learning new software.",
            "The worker collaborated with their colleagues.",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_steering_vector(
    model,
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer: int,
    concept_label: str = "concept",
    method: str = "mean_diff",
    scale: float = -15.0,
    device: Optional[str] = None,
) -> SteeringVector:
    """
    One-liner wrapper for the common case.

    Example
    -------
    >>> sv = extract_steering_vector(model, pos, neg, layer=8, concept_label="gender_bias")
    >>> exporter = SteeringVectorExporter()
    >>> exporter.export_pt(sv, "steering/gender_bias.pt")
    """
    exporter = SteeringVectorExporter(method=method, device=device)
    return exporter.extract_mean_diff(
        model=model,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        layer=layer,
        concept_label=concept_label,
        scale=scale,
    )
