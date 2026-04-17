# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ajay Pravin Mahale
"""
glassbox/mlflow_integration.py — MLflow Integration
====================================================

Logs Glassbox circuit analysis results as MLflow metrics, parameters,
and artifacts — enabling model registry workflows and experiment tracking.

Every enterprise ML team uses MLflow (or a hosted variant like Databricks
MLflow, Azure ML, AWS SageMaker Experiments). This integration makes
Glassbox results first-class citizens in those workflows.

Usage
-----
::

    import mlflow
    from glassbox import GlassboxV2
    from glassbox.mlflow_integration import log_glassbox_run, GlassboxMLflowCallback

    # Option 1: one-shot log to current active run
    with mlflow.start_run(run_name="gpt2-compliance-audit"):
        result = gb.analyze(prompt, correct, incorrect)
        log_glassbox_run(result, model_name="gpt2", use_case="credit_scoring")

    # Option 2: callback-style (plug into training loops)
    callback = GlassboxMLflowCallback(gb, prompt, correct, incorrect)
    callback.on_epoch_end(epoch=5)   # logs metrics for this checkpoint

    # Option 3: model registry — log as compliance artifact
    from glassbox.mlflow_integration import register_compliance_artifact
    register_compliance_artifact(
        result,
        run_id="abc123",
        model_uri="models:/MyModel/Production",
    )

Requirements
------------
    pip install mlflow glassbox-mech-interp
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def _require_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        raise ImportError(
            "mlflow is required for MLflow integration.\n"
            "Install it:  pip install mlflow"
        )


# ---------------------------------------------------------------------------
# Core function: log_glassbox_run
# ---------------------------------------------------------------------------

def log_glassbox_run(
    result: Dict[str, Any],
    model_name: Optional[str] = None,
    use_case: Optional[str] = None,
    prompt: Optional[str] = None,
    step: Optional[int] = None,
    log_circuit_json: bool = True,
    log_html_report:  bool = True,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """
    Log a Glassbox analysis result to the active MLflow run.

    Logs the following to MLflow:

    **Metrics** (numeric, plottable over steps/epochs):
    - ``glassbox.faithfulness.sufficiency``
    - ``glassbox.faithfulness.comprehensiveness``
    - ``glassbox.faithfulness.f1``
    - ``glassbox.circuit.n_heads``
    - ``glassbox.stability.mean_jaccard`` (if stability_suite was run)
    - ``glassbox.stability.rate`` (if stability_suite was run)

    **Parameters** (categorical, recorded once):
    - ``glassbox.model_name``
    - ``glassbox.use_case``
    - ``glassbox.behaviour_category``
    - ``glassbox.compliance_status``
    - ``glassbox.version``

    **Artifacts** (files):
    - ``glassbox/circuit.json`` — full circuit result JSON
    - ``glassbox/report.html`` — HTML explainability report (optional)

    Parameters
    ----------
    result : dict
        Output of ``GlassboxV2.analyze()``.
    model_name : str, optional
        Model identifier for parameter logging.
    use_case : str, optional
        Deployment use case string (e.g. ``'credit_scoring'``).
    prompt : str, optional
        The analysis prompt, stored as a parameter.
    step : int, optional
        MLflow step (epoch number, checkpoint, etc.).
    log_circuit_json : bool
        Whether to log the full result dict as a JSON artifact.
    log_html_report : bool
        Whether to generate and log an HTML explainability report artifact.
    tags : dict, optional
        Extra MLflow tags to set on the run.

    Raises
    ------
    ImportError
        If ``mlflow`` is not installed.
    RuntimeError
        If no active MLflow run exists.
    """
    mlflow = _require_mlflow()

    if mlflow.active_run() is None:
        raise RuntimeError(
            "No active MLflow run. Call this inside a 'with mlflow.start_run():' block:\n\n"
            "    with mlflow.start_run():\n"
            "        log_glassbox_run(result)"
        )

    faith    = result.get("faithfulness", {})
    suff     = faith.get("sufficiency", 0.0)
    comp     = faith.get("comprehensiveness", 0.0)
    f1       = faith.get("f1", 0.0)
    cat      = faith.get("category", "unknown")
    n_heads  = result.get("n_heads", 0)
    stab     = result.get("stability", {})

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics: Dict[str, float] = {
        "glassbox.faithfulness.sufficiency":       round(suff,  4),
        "glassbox.faithfulness.comprehensiveness": round(comp,  4),
        "glassbox.faithfulness.f1":                round(f1,    4),
        "glassbox.circuit.n_heads":                float(n_heads),
    }
    if stab:
        if "mean_jaccard" in stab:
            metrics["glassbox.stability.mean_jaccard"] = round(stab["mean_jaccard"], 4)
        if "stability_rate" in stab:
            metrics["glassbox.stability.rate"] = round(stab["stability_rate"], 4)
        if "std_jaccard" in stab:
            metrics["glassbox.stability.std_jaccard"] = round(stab["std_jaccard"], 4)
    if "logit_diff" in result:
        metrics["glassbox.logit_diff.clean"] = round(float(result["logit_diff"]), 4)
    if "logit_diff_corrupted" in result:
        metrics["glassbox.logit_diff.corrupted"] = round(float(result["logit_diff_corrupted"]), 4)

    mlflow.log_metrics(metrics, step=step)

    # ── Parameters ─────────────────────────────────────────────────────────
    compliance_status = "COMPLIANT" if suff >= 0.75 else "NEEDS_REVIEW"
    params: Dict[str, str] = {
        "glassbox.version": "4.2.6",
        "glassbox.behaviour_category":  cat,
        "glassbox.compliance_status":   compliance_status,
        "glassbox.regulation":          "EU-AI-Act-AnnexIV",
    }
    if model_name:
        params["glassbox.model_name"] = model_name
    if use_case:
        params["glassbox.use_case"] = use_case
    if prompt:
        params["glassbox.prompt_preview"] = prompt[:200]

    # MLflow params can't be overwritten in the same run without raising;
    # silently ignore duplicate param errors.
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except mlflow.exceptions.MlflowException:
            pass

    # ── Tags ───────────────────────────────────────────────────────────────
    run_tags: Dict[str, str] = {
        "glassbox.compliance_status": compliance_status,
        "glassbox.faithfulness_grade": _grade(suff),
    }
    if tags:
        run_tags.update(tags)
    mlflow.set_tags(run_tags)

    # ── Artifacts ──────────────────────────────────────────────────────────
    if log_circuit_json:
        _log_json_artifact(mlflow, result, artifact_path="glassbox/circuit.json")

    if log_html_report:
        _log_html_artifact(mlflow, result, model_name=model_name, prompt=prompt)


# ---------------------------------------------------------------------------
# register_compliance_artifact
# ---------------------------------------------------------------------------

def register_compliance_artifact(
    result: Dict[str, Any],
    run_id: str,
    model_uri: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """
    Log a Glassbox compliance artifact to a specific (possibly completed) run
    and optionally attach a tag to a registered model version.

    Parameters
    ----------
    result : dict
        Output of ``GlassboxV2.analyze()``.
    run_id : str
        MLflow run ID to attach the artifact to.
    model_uri : str, optional
        MLflow model URI, e.g. ``"models:/MyModel/Production"``.
        If provided, sets a ``glassbox.compliance_status`` tag on the
        registered model version.
    model_name : str, optional
        Model identifier string.
    """
    mlflow = _require_mlflow()
    client = mlflow.tracking.MlflowClient()

    faith  = result.get("faithfulness", {})
    suff   = faith.get("sufficiency", 0.0)
    status = "COMPLIANT" if suff >= 0.75 else "NEEDS_REVIEW"

    # Log JSON artifact to the run
    payload = {
        "glassbox_version": "4.2.6",
        "audit_timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name":       model_name or "",
        "faithfulness":     result.get("faithfulness", {}),
        "circuit_n_heads":  result.get("n_heads", 0),
        "compliance_status": status,
        "regulation":       "Regulation (EU) 2024/1689 — AI Act Annex IV",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "glassbox_compliance.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        client.log_artifact(run_id, path, artifact_path="glassbox")

    # Tag the registered model version if URI is provided
    if model_uri:
        try:
            from mlflow.tracking import MlflowClient
            # Parse "models:/ModelName/version_or_stage" format
            parts = model_uri.replace("models:/", "").split("/")
            if len(parts) >= 2:
                reg_model_name = parts[0]
                version_or_stage = parts[1]
                # Try to get version number
                try:
                    version = int(version_or_stage)
                    client.set_model_version_tag(
                        name    = reg_model_name,
                        version = str(version),
                        key     = "glassbox.compliance_status",
                        value   = status,
                    )
                    client.set_model_version_tag(
                        name    = reg_model_name,
                        version = str(version),
                        key     = "glassbox.faithfulness_sufficiency",
                        value   = str(round(suff, 4)),
                    )
                except ValueError:
                    pass  # Stage name, not version number — skip tagging
        except Exception:
            pass  # Model registry tagging is best-effort


# ---------------------------------------------------------------------------
# GlassboxMLflowCallback
# ---------------------------------------------------------------------------

class GlassboxMLflowCallback:
    """
    Callback wrapper for logging Glassbox metrics during model training.

    Designed for integration with training loops (PyTorch, HuggingFace
    Trainer, etc.). Call ``on_epoch_end()`` at each checkpoint.

    Parameters
    ----------
    gb : GlassboxV2
        A pre-initialised Glassbox engine.
    prompt : str
        The audit prompt to run at each checkpoint.
    correct : str
        The correct token string.
    incorrect : str
        The incorrect token string.
    run_name : str, optional
        MLflow run name. If not set, uses the active run.
    log_every_n_epochs : int
        How often to run the audit (default: 1 = every epoch).

    Example
    -------
    ::

        import mlflow
        from glassbox.mlflow_integration import GlassboxMLflowCallback

        callback = GlassboxMLflowCallback(
            gb        = gb,
            prompt    = "When Mary and John went to the store, John gave a drink to",
            correct   = " Mary",
            incorrect = " John",
        )

        with mlflow.start_run():
            for epoch in range(10):
                train_one_epoch(model, dataloader)
                callback.on_epoch_end(epoch)
    """

    def __init__(
        self,
        gb,
        prompt: str,
        correct: str,
        incorrect: str,
        run_name: Optional[str] = None,
        log_every_n_epochs: int = 1,
    ) -> None:
        self.gb                 = gb
        self.prompt             = prompt
        self.correct            = correct
        self.incorrect          = incorrect
        self.run_name           = run_name
        self.log_every_n_epochs = log_every_n_epochs
        self._history: List[Dict[str, Any]] = []

    def on_epoch_end(self, epoch: int) -> Optional[Dict[str, Any]]:
        """
        Run a Glassbox audit and log metrics to the active MLflow run.

        Parameters
        ----------
        epoch : int
            Current training epoch (used as the MLflow step).

        Returns
        -------
        dict or None
            The analysis result, or ``None`` if this epoch was skipped.
        """
        if epoch % self.log_every_n_epochs != 0:
            return None

        result = self.gb.analyze(
            prompt    = self.prompt,
            correct   = self.correct,
            incorrect = self.incorrect,
        )
        log_glassbox_run(result, step=epoch, prompt=self.prompt)
        self._history.append({"epoch": epoch, "result": result})
        return result

    def history(self) -> List[Dict[str, Any]]:
        """Return the list of all recorded audit results."""
        return self._history

    def compliance_trend(self) -> List[Dict[str, float]]:
        """
        Return a list of (epoch, sufficiency, f1) dicts for trend plotting.

        Example
        -------
        ::

            import pandas as pd
            df = pd.DataFrame(callback.compliance_trend())
            df.plot(x="epoch", y=["sufficiency", "f1"])
        """
        return [
            {
                "epoch":         rec["epoch"],
                "sufficiency":   rec["result"].get("faithfulness", {}).get("sufficiency", 0.0),
                "comprehensiveness": rec["result"].get("faithfulness", {}).get("comprehensiveness", 0.0),
                "f1":            rec["result"].get("faithfulness", {}).get("f1", 0.0),
                "n_heads":       rec["result"].get("n_heads", 0),
            }
            for rec in self._history
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grade(suff: float) -> str:
    if suff >= 0.90: return "Excellent"
    if suff >= 0.75: return "Good"
    if suff >= 0.50: return "Marginal"
    return "Poor"


def _log_json_artifact(mlflow, result: Dict[str, Any], artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.basename(artifact_path)
        dname = os.path.dirname(artifact_path)
        local_path = os.path.join(tmpdir, fname)

        # Serialise — convert any non-JSON-serialisable objects
        def _safe(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        with open(local_path, "w") as f:
            json.dump(result, f, indent=2, default=_safe)

        mlflow.log_artifact(local_path, artifact_path=dname or None)


def _log_html_artifact(
    mlflow,
    result: Dict[str, Any],
    model_name: Optional[str],
    prompt: Optional[str],
) -> None:
    try:
        from glassbox.explain import NaturalLanguageExplainer
        html = NaturalLanguageExplainer(verbosity="standard").to_html(
            result, model_name=model_name, prompt=prompt
        )
    except Exception:
        faith = result.get("faithfulness", {})
        html  = f"<pre>{json.dumps(faith, indent=2)}</pre>"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        with open(path, "w") as f:
            f.write(html)
        mlflow.log_artifact(path, artifact_path="glassbox")
