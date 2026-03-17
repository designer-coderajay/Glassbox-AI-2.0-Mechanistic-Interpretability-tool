"""
api/main.py — Glassbox REST API
=================================

POST /v1/audit/analyze    — White-box circuit analysis + Annex IV compliance report
POST /v1/audit/black-box  — Black-box behavioural audit (any model via API)
GET  /v1/audit/report/:id — Retrieve a previously generated report
GET  /health              — Health check
GET  /                    — API info

This is the API that enterprise compliance buyers will call.
Every response is a structured JSON compliant with EU AI Act Annex IV.

Usage
-----
    pip install fastapi uvicorn
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    # Analyze a model decision
    curl -X POST http://localhost:8000/v1/audit/analyze \\
         -H "Content-Type: application/json" \\
         -d '{
           "model_name": "gpt2",
           "prompt": "When Mary and John went to the store, John gave a drink to",
           "correct_token": " Mary",
           "incorrect_token": " John",
           "provider_name": "Acme Corp",
           "provider_address": "1 Main St, Amsterdam",
           "system_purpose": "Customer service NLP",
           "deployment_context": "financial_services",
           "generate_pdf": true
         }'

Dependencies
------------
    pip install fastapi uvicorn transformer-lens glassbox-mech-interp reportlab
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    # Stub classes for type hints when fastapi not installed
    class BaseModel:   # type: ignore
        pass
    class Field:       # type: ignore
        def __call__(self, *a, **kw): pass

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class WhiteBoxRequest(BaseModel):
    """
    Request body for POST /v1/audit/analyze.
    White-box analysis — requires a TransformerLens-supported model name.
    """
    # Analysis parameters
    model_name:       str   = Field(..., example="gpt2", description="HuggingFace model name, must be TransformerLens-compatible")
    prompt:           str   = Field(..., example="When Mary and John went to the store, John gave a drink to")
    correct_token:    str   = Field(..., example=" Mary")
    incorrect_token:  str   = Field(..., example=" John")
    method:           str   = Field("taylor", example="taylor", description="Attribution method: 'taylor' or 'integrated_gradients'")
    include_logit_lens: bool = Field(False)

    # Annex IV report parameters
    provider_name:    str   = Field(..., example="Acme Bank NV")
    provider_address: str   = Field(..., example="1 Fintech Street, Amsterdam 1011AB")
    system_purpose:   str   = Field(..., example="Credit risk assessment for loan applications")
    deployment_context: str = Field("other_high_risk", example="financial_services")
    use_case:         str   = Field("AI decision analysis")
    generate_pdf:     bool  = Field(True)


class BlackBoxRequest(BaseModel):
    """
    Request body for POST /v1/audit/black-box.
    Works on ANY model via API — no weights needed.
    """
    # Target model configuration
    target_provider: str = Field(..., example="openai", description="'openai', 'anthropic', 'together', 'groq'")
    target_model:    str = Field(..., example="gpt-4")
    api_key:         str = Field(..., description="API key for the target model (not stored)")

    # Audit parameters
    decision_prompt:   str   = Field(..., example="The loan applicant has a credit score of 620. The application should be")
    expected_positive: str   = Field(..., example="approved")
    expected_negative: str   = Field(..., example="denied")
    context_variables: Optional[Dict[str, Any]] = Field(None, example={"credit_score": 620, "loan_amount": 25000})
    n_rephrases:       int   = Field(3, ge=0, le=10)
    n_sensitivity_steps: int = Field(5, ge=2, le=10)

    # Annex IV report parameters
    provider_name:    str = Field(..., example="Acme Bank NV")
    provider_address: str = Field(..., example="1 Fintech Street, Amsterdam 1011AB")
    system_purpose:   str = Field(..., example="Credit risk assessment for loan applications")
    deployment_context: str = Field("financial_services")
    use_case:         str = Field("AI decision analysis")
    generate_pdf:     bool = Field(True)


class AuditResponse(BaseModel):
    """Standard response from any /v1/audit/* endpoint."""
    report_id:          str
    status:             str
    compliance_status:  str
    explainability_grade: str
    faithfulness:       Dict[str, Any]
    n_circuit_components: int
    analysis_mode:      str  # "white_box" or "black_box"
    json_report_url:    str
    pdf_report_url:     Optional[str]
    full_report:        Dict[str, Any]
    elapsed_seconds:    float


# ---------------------------------------------------------------------------
# In-memory report store (replace with Redis/DB in production)
# ---------------------------------------------------------------------------
_REPORT_STORE: Dict[str, Dict[str, Any]] = {}
_PDF_STORE:    Dict[str, Path]          = {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app() -> "FastAPI":
    if not _FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Glassbox AI Compliance API",
        description=(
            "EU AI Act Annex IV compliance report generator. "
            "White-box mechanistic interpretability analysis + "
            "black-box behavioural audit. "
            "Regulation (EU) 2024/1689 — enforced August 2026."
        ),
        version=_get_version(),
        contact={"name": "Ajay Pravin Mahale", "email": "mahale.ajay01@gmail.com"},
        license_info={"name": "Apache 2.0"},
    )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "glassbox_version": _get_version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # ------------------------------------------------------------------
    # API info
    # ------------------------------------------------------------------
    @app.get("/")
    def root():
        return {
            "name":       "Glassbox AI Compliance API",
            "version": _get_version(),
            "regulation": "Regulation (EU) 2024/1689 — AI Act",
            "endpoints": {
                "white_box_analysis": "POST /v1/audit/analyze",
                "black_box_audit":    "POST /v1/audit/black-box",
                "retrieve_report":    "GET /v1/audit/report/{report_id}",
                "download_pdf":       "GET /v1/audit/pdf/{report_id}",
                "health":             "GET /health",
            },
            "docs":        "/docs",
            "openapi_url": "/openapi.json",
        }

    # ------------------------------------------------------------------
    # Dashboard UI
    # ------------------------------------------------------------------
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        """Serve the compliance dashboard UI."""
        dashboard_path = Path(__file__).parent.parent / "dashboard" / "compliance_dashboard.html"
        if dashboard_path.exists():
            return HTMLResponse(content=dashboard_path.read_text(), status_code=200)
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

    # ------------------------------------------------------------------
    # White-box analysis
    # ------------------------------------------------------------------
    @app.post("/v1/audit/analyze", response_model=AuditResponse)
    def analyze_white_box(req: WhiteBoxRequest):
        """
        Full white-box mechanistic interpretability analysis + Annex IV report.

        Loads the model via TransformerLens, runs attribution patching + MFC,
        and generates a complete EU AI Act Annex IV compliance report.

        Model must be supported by TransformerLens (GPT-2, GPT-Neo, etc.)
        For proprietary models (GPT-4, Claude), use /v1/audit/black-box instead.
        """
        start = time.time()
        report_id = uuid.uuid4().hex[:8].upper()

        try:
            import torch
            from transformer_lens import HookedTransformer
            from glassbox import GlassboxV2
            from glassbox.compliance import AnnexIVReport, DeploymentContext

            logger.info("[%s] Loading model: %s", report_id, req.model_name)
            model = HookedTransformer.from_pretrained(req.model_name)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            gb = GlassboxV2(model)
            logger.info("[%s] Running analysis...", report_id)
            result = gb.analyze(
                prompt            = req.prompt,
                correct           = req.correct_token,
                incorrect         = req.incorrect_token,
                method            = req.method,
                include_logit_lens= req.include_logit_lens,
            )

        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model loading failed. Ensure transformer-lens is installed: {e}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)[:300]}")

        try:
            ctx = _parse_context(req.deployment_context)
            annex = AnnexIVReport(
                model_name         = req.model_name,
                system_purpose     = req.system_purpose,
                provider_name      = req.provider_name,
                provider_address   = req.provider_address,
                deployment_context = ctx,
            )
            annex.add_analysis(result, use_case=req.use_case)

            # Store JSON
            json_report = annex.to_json()
            _REPORT_STORE[report_id] = {
                "json": json_report,
                "mode": "white_box",
                "created_at": time.time(),
            }

            # Generate PDF
            pdf_url = None
            if req.generate_pdf:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    pdf_path = Path(tmp.name)
                annex.to_pdf(str(pdf_path))
                _PDF_STORE[report_id] = pdf_path
                pdf_url = f"/v1/audit/pdf/{report_id}"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)[:300]}")

        import json
        report_data = json.loads(json_report)
        s3 = report_data["sections"]["3_monitoring_control"]

        return AuditResponse(
            report_id              = report_id,
            status                 = "completed",
            compliance_status      = report_data.get("compliance_status", ""),
            explainability_grade   = s3.get("explainability_grade", ""),
            faithfulness           = result["faithfulness"],
            n_circuit_components   = result["n_heads"],
            analysis_mode          = "white_box",
            json_report_url        = f"/v1/audit/report/{report_id}",
            pdf_report_url         = pdf_url,
            full_report            = report_data,
            elapsed_seconds        = round(time.time() - start, 2),
        )

    # ------------------------------------------------------------------
    # Black-box audit
    # ------------------------------------------------------------------
    @app.post("/v1/audit/black-box", response_model=AuditResponse)
    def analyze_black_box(req: BlackBoxRequest):
        """
        Black-box behavioural audit for ANY model via API.

        Works on GPT-4, Claude, Llama, Gemini, or any proprietary LLM.
        Uses counterfactual probing + sensitivity analysis + consistency testing
        to produce EU AI Act Annex IV-compatible explainability metrics.

        The API key is used only for this request and is not stored.
        """
        start = time.time()
        report_id = uuid.uuid4().hex[:8].upper()

        try:
            from glassbox.audit import BlackBoxAuditor, ModelProvider
            from glassbox.compliance import AnnexIVReport

            provider = ModelProvider(req.target_provider)
            auditor  = BlackBoxAuditor(
                model_provider = provider,
                model_name     = req.target_model,
                api_key        = req.api_key,
            )

            logger.info("[%s] Black-box audit: %s/%s", report_id, req.target_provider, req.target_model)
            result = auditor.audit(
                decision_prompt    = req.decision_prompt,
                expected_positive  = req.expected_positive,
                expected_negative  = req.expected_negative,
                context_variables  = req.context_variables,
                n_rephrases        = req.n_rephrases,
                n_sensitivity_steps= req.n_sensitivity_steps,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {e}")
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=f"Target model API error: {str(e)[:300]}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audit error: {str(e)[:300]}")

        try:
            ctx   = _parse_context(req.deployment_context)
            annex = AnnexIVReport(
                model_name         = req.target_model,
                system_purpose     = req.system_purpose,
                provider_name      = req.provider_name,
                provider_address   = req.provider_address,
                deployment_context = ctx,
            )
            annex.add_analysis(result, use_case=req.use_case)

            json_report = annex.to_json()
            _REPORT_STORE[report_id] = {
                "json": json_report,
                "mode": "black_box",
                "created_at": time.time(),
            }

            pdf_url = None
            if req.generate_pdf:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    pdf_path = Path(tmp.name)
                annex.to_pdf(str(pdf_path))
                _PDF_STORE[report_id] = pdf_path
                pdf_url = f"/v1/audit/pdf/{report_id}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)[:300]}")

        import json
        report_data = json.loads(json_report)
        s3 = report_data["sections"]["3_monitoring_control"]

        return AuditResponse(
            report_id              = report_id,
            status                 = "completed",
            compliance_status      = report_data.get("compliance_status", ""),
            explainability_grade   = s3.get("explainability_grade", ""),
            faithfulness           = result["faithfulness"],
            n_circuit_components   = result["n_heads"],
            analysis_mode          = "black_box",
            json_report_url        = f"/v1/audit/report/{report_id}",
            pdf_report_url         = pdf_url,
            full_report            = report_data,
            elapsed_seconds        = round(time.time() - start, 2),
        )

    # ------------------------------------------------------------------
    # Retrieve report
    # ------------------------------------------------------------------
    @app.get("/v1/audit/report/{report_id}")
    def get_report(report_id: str):
        """Retrieve a previously generated JSON report by ID."""
        report_id = report_id.upper()
        if report_id not in _REPORT_STORE:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
        import json
        return JSONResponse(json.loads(_REPORT_STORE[report_id]["json"]))

    # ------------------------------------------------------------------
    # Download PDF
    # ------------------------------------------------------------------
    @app.get("/v1/audit/pdf/{report_id}")
    def download_pdf(report_id: str):
        """Download the Annex IV PDF report for a given report ID."""
        report_id = report_id.upper()
        if report_id not in _PDF_STORE:
            raise HTTPException(status_code=404, detail=f"PDF for report {report_id} not found.")
        pdf_path = _PDF_STORE[report_id]
        if not pdf_path.exists():
            raise HTTPException(status_code=410, detail="PDF file no longer available.")
        return FileResponse(
            str(pdf_path),
            media_type   = "application/pdf",
            filename     = f"annex_iv_{report_id}.pdf",
        )

    # ------------------------------------------------------------------
    # List reports
    # ------------------------------------------------------------------
    @app.get("/v1/audit/reports")
    def list_reports():
        """List all report IDs stored in this session."""
        return {
            "reports": [
                {
                    "report_id":  rid,
                    "mode":       data["mode"],
                    "created_at": data["created_at"],
                    "has_pdf":    rid in _PDF_STORE,
                }
                for rid, data in _REPORT_STORE.items()
            ],
            "total": len(_REPORT_STORE),
        }

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_context(ctx_str: str):
    from glassbox.compliance import DeploymentContext
    try:
        return DeploymentContext(ctx_str)
    except ValueError:
        return DeploymentContext.OTHER_HIGH_RISK


def _get_version() -> str:
    try:
        from glassbox import __version__
        return __version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# WSGI app instance
# ---------------------------------------------------------------------------
if _FASTAPI_AVAILABLE:
    app = create_app()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("Install uvicorn: pip install uvicorn")
        raise

    uvicorn.run(
        "api.main:app",
        host    = os.environ.get("HOST", "0.0.0.0"),
        port    = int(os.environ.get("PORT", "8000")),
        reload  = os.environ.get("ENV", "production") == "development",
        log_level = "info",
    )
