# Changelog

All notable changes to Glassbox are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.7.0] — 2026-04-03

### Added
- **`glassbox/corruption.py`** — `MultiCorruptionPipeline` with 4 corruption strategies (`CorruptionStrategy` enum):
  - `NAME_SWAP`: Bidirectional IO⇔S name swap (Wang et al. 2022 standard)
  - `RANDOM_TOKEN`: Replace IO/S tokens with `Uniform(V)` random vocabulary token
  - `GAUSSIAN_NOISE`: Add `N(0, σ²·I)` noise to token embeddings (σ = std of clean embeddings)
  - `MEAN_ABLATION`: Replace last-position residual stream with dataset mean (zero ablation fallback)
  - `RobustnessReport`: aggregated across all corruptions; flags `perturbation_sensitive` when `max_k |S_k − S̄| ≥ 0.10`
  - `CorruptionResult` dataclass: per-strategy S/Comp/F1/LD metrics
- **`glassbox/validation.py`** — Statistical validation gates:
  - `SampleSizeGate`: raises `SampleSizeError` (n<20, hard block) or `SampleSizeWarning` (n<50, soft warn)
  - `HeldOutValidator`: 50/50 train/test split; flags `overfit` when `|F1_train − F1_test| ≥ 0.10`

### Changed
- Version `3.6.0` → `3.7.0`

### Mathematical Foundation
- Robustness criterion: ∀k : |S_k(C) − S̄| < δ = 0.10
- Power analysis: n_min = ((z_{α/2} + z_β) / atanh(ρ_min))² + 3; n≥50 → 80% power at |ρ|≥0.25
- Generalisation gap: gap = |F1_train − F1_test| < δ_gen = 0.10

---

## [3.6.0] — 2026-04-02

### Added
- **Full-stack interactive website**: Live circuit analyzer embedded directly in landing page — paste a prompt, pick a model, get real-time attribution heatmap + faithfulness metrics + compliance grade in-browser. Vanilla JS, no build step, graceful fallback to demo data when backend unavailable
- **WebSocket streaming** (`/ws/{job_id}`): Real-time analysis progress (stage indicators, percent bars, live messages) instead of blocking long-poll requests
- **CORS + rate limiting** in FastAPI: `CORSMiddleware` for Vercel/localhost; 20 req/min rate limiter middleware so the API is production-safe without nginx
- **Vercel API routing** (`api/index.py` + `vercel.json` rewrites): `/api/*` paths now proxy to the FastAPI serverless function — one domain, no separate backend needed for light loads

### Fixed
- **MCP server critical import** (`mcp/server.py`): `GlassboxAnalyzer` → `GlassboxV2` (was `ImportError` at runtime on every circuit discovery call)
- **MCP server `analyze()` signature** (`mcp/server.py`): `corrupted_prompt` → `correct`/`incorrect` (was `TypeError` on every invocation)
- **Async GPU blocking** (`mcp/server.py`): Wrapped all blocking TransformerLens calls in `asyncio.to_thread()` so MCP server event loop no longer stalls during model inference
- **`analyze()` input validation** (`glassbox/core.py`): Empty prompt/correct/incorrect now raises `ValueError` immediately instead of producing silent garbage output
- **Non-deterministic circuit sort** (`glassbox/core.py`): Secondary sort key `(layer, head)` added — compliance reports now produce identical circuit orderings across runs
- **`__version__` sync** (`glassbox/__init__.py`): Was `3.4.0`, now tracks `pyproject.toml` correctly

### Changed
- Version `3.5.0` → `3.6.0`
- README title and Docker image tags updated to `3.6.0`
- `.claude/CLAUDE.md`: HuggingFace Space URL corrected (`affaan/glassbox` → `designer-coderajay/...`), website URL corrected
- `pyproject.toml`: `notify = []` annotated — webhook-based, no pip deps needed

---

## [3.5.0] — 2026-04-01

### Added
- **Claude Code plugin** (`.claude/`): Full project brain, 6 specialized agents (interpretability-researcher on Opus, compliance-generator, python-reviewer, pytorch-build-resolver, code-reviewer, doc-updater), 6 skills (mechanistic-interpretability v2.0 with SAEs + steering vectors, eu-ai-act-compliance with GPAI Articles 51–55, python-testing with Hypothesis + syrupy, pytorch-transformerlens with Pythia multi-model, security-review with pip-audit, circuit-discovery), and 5 slash commands (`/circuit`, `/compliance`, `/review`, `/audit`, `/test`)
- **FastMCP server** (`mcp/`): Model Context Protocol server with 5 tools — `glassbox_circuit_discovery`, `glassbox_faithfulness_metrics`, `glassbox_compliance_report` (full 9-section Annex IV JSON), `glassbox_attention_patterns`, `glassbox_logit_lens`. Pydantic v2 input validation, model allowlist, graceful degradation when library not installed
- **Brand asset** (`assets/glassbox_brand.png`): 1400×800 circuit-trace design with attribution heatmap, L9H9 gold highlight (attribution=0.584), faithfulness bars and compliance card

### Fixed
- `glassbox/__init__.py` `__version__` corrected to `3.5.0`
- MCP server class reference corrected from `GlassboxAnalyzer` → `GlassboxV2`
- MCP server `analyze()` parameter names corrected (`correct`/`incorrect` not `corrupted_prompt`)
- Non-deterministic circuit sort — added secondary sort key `(layer, head)` for reproducible compliance reports
- Input validation added to `analyze()` — empty strings now raise `ValueError` immediately

### Changed
- `deploy_hf.yml` workflow: added `workflow_dispatch` trigger and `glassbox/**` path filter so library changes auto-sync to HuggingFace Space
- HuggingFace Space requirement bumped to `glassbox-mech-interp>=3.5.0`

---

## [Unreleased] — HuggingFace Space UI Fixes — 2026-03-22

### Fixed
- **HF Space About tab blank** — root cause was a CSS selector (`.etali4b10, .svelte-po8fcl { display:none !important }`) that matched the Gradio 4.43.0 Markdown component wrapper, silently hiding every `gr.Markdown()` and `gr.HTML()` block in the app. Full GB_CSS rewrite replacing 430 lines with 270 lines of clean, version-stable CSS using semantic selectors and `data-testid` attributes only.
- **Compliance Report "Error generating compliance report: D"** — `AnnexIVReport` library class raises an internal exception with message `"D"` on HF Space (version mismatch). Restructured `run_compliance_report` to derive grade A/B/C/D directly from the raw `gb.analyze()` result without depending on `AnnexIVReport`. `AnnexIVReport` is now tried non-blocking for the model card only; a fallback model card is generated if it throws.
- **Compliance Report output invisible** — `cr_report` component was `gr.HTML()` but the function returns mixed Markdown+HTML. Changed to `gr.Markdown(sanitize_html=False)` which renders both Markdown tables/headings and inline HTML div blocks.
- **About tab blank (previous attempts)** — replaced `gr.HTML(ABOUT_HTML)` (unreliable in HF iframe) with `gr.Markdown(ABOUT_MD)` using standard Markdown content. Gradio's Markdown component is unconditionally reliable.
- **HF Space URLs in README** — all 7 occurrences of the wrong Space ID (`Glassbox-ai`) corrected to `Glassbox-AI-2.0-Mechanistic-Interpretability-tool`.
- **pyproject.toml project URLs** — `Homepage`, `Dashboard`, `Documentation` updated from stale Render.com URL to active Vercel site, HF Space, and GitHub README.
- **GitHub Actions workflow** — fixed deploy target from wrong Space name to `Glassbox-AI-2.0-Mechanistic-Interpretability-tool`; added force-push and dynamic commit messages.

---

## [3.4.0] — 2026-03-21

### Added
- **MultiAgentAudit** (`glassbox/multi_agent_audit.py`): First open-source tool to trace bias contamination and semantic drift across multi-agent chains. `MultiAgentAudit().audit_chain([AgentCall(...)])` returns a `ChainAuditReport` with per-agent liability scoring, most-liable-agent identification, and Annex IV Article 9 narrative. Scores bias across 8 EU AI Act Article 10(5) protected categories. `to_html()` generates a self-contained liability dashboard. Maps to Article 9, Article 10(2)(f), Article 10(5), Article 13(1). Exported from top-level: `from glassbox import MultiAgentAudit, AgentCall`.
- **SteeringVectorExporter** (`glassbox/steering.py`): Extract and export steering vectors from the residual stream using Representation Engineering (Zou et al. 2023). `extract_mean_diff()`, `extract_pca()`, and `extract_bias_suite()` methods. `apply()` applies a vector as a runtime hook. `test_suppression()` computes before/after faithfulness comparison and suppression ratio. `export_pt()` / `export_numpy()` for regulatory submission artefacts. Maps to Article 9(2)(b), Article 9(5), Article 15(1). Exported: `SteeringVectorExporter, SteeringVector`.
- **AnnexIVEvidenceVault** (`glassbox/annex_iv_vault.py`): Assembles all interpretability findings (circuit analysis, bias tests, steering vectors, multi-agent audits, SAE features, stability scores) into a single machine-readable, regulation-mapped Annex IV evidence vault. `build_annex_iv_vault()` top-level function. Outputs JSON and self-contained HTML suitable for regulatory submission. Covers Annex IV §1–§7, maps to Articles 9, 10, 11, 13, 15, 72.
- **HuggingFace Space v3.4** interactive dashboard deployed at `designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool` with five tabs: Circuit Analysis, Logit Lens, Attention Patterns, Compliance Report, About.

### Changed
- Version bumped to 3.4.0 in `pyproject.toml`.
- README: Added v3.4.0 "What's New" section with full code examples for all three new modules. Updated live services table. Live demo link updated to correct HF Space.

---

## [3.3.0] — 2026-03-20

### Added
- **NaturalLanguageExplainer** (`glassbox/explain.py`): Rule-based, deterministic converter from raw circuit analysis results to structured plain-English compliance summaries. Zero LLM dependency. `verbosity` levels: `"brief"`, `"standard"`, `"detailed"`. `include_article_refs=True` adds EU AI Act article citations to every sentence. Methods: `explain()`, `explain_sections()`, `to_html()`. Exported: `NaturalLanguageExplainer`.
- **HuggingFace Hub integration** (`glassbox/hf_integration.py`): `load_from_hub()` loads any HookedTransformer-compatible model from HF Hub (29 architecture aliases). `HuggingFaceModelCard` pushes/reads compliance metadata sections to/from model card README.md. `push_compliance_section()` adds grade, F1, circuit summary, and article mapping. Exported: `load_from_hub, HuggingFaceModelCard`. Install: `pip install 'glassbox-mech-interp[hf]'`.
- **MLflow integration** (`glassbox/mlflow_integration.py`): `log_glassbox_run()` logs a full circuit analysis result as an MLflow run — grade, F1, sufficiency, comprehensiveness, circuit heads, and prompt metadata. `GlassboxMLflowCallback` for automatic logging during batch analysis. Install: `pip install 'glassbox-mech-interp[mlflow]'`.
- **Slack/Teams alerting** (`glassbox/notify.py`): `SlackAlerter` and `TeamsAlerter` send formatted alerts when CircuitDiff detects drift (`drift_threshold`) or compliance grade drops (`grade_threshold`). Webhook-based, no SDK required. `AlertConfig` for threshold management.
- **GitHub Action CI hook** (`glassbox/ci_hook.py`): `check_compliance_gate()` function suitable for use in GitHub Actions; exits with code 1 if compliance grade drops below configured threshold. Sample workflow in README.

### Changed
- `GlassboxV2.analyze()` now integrates `NaturalLanguageExplainer` — result dict includes `"explanation"` key with plain-English summary when `explain=True` (default False to preserve backward compatibility).
- README: Added v3.3.0 "What's New" section. Updated live services table.

---

## [3.2.0] — 2026-03-20

### Added
- **Black-box audit mode**: `BlackBoxAuditor` class in `glassbox/black_box.py`. Runs behavioural proxy metrics (token probability, output consistency, bias probes) on any model accessible via a callable — GPT-4, Claude, Gemini, or any proprietary API. No model weights required. `audit_api_model(model_fn, prompt, correct, incorrect)` returns faithfulness proxies and Annex IV draft. Exported: `BlackBoxAuditor`.
- **Stability suite** (`glassbox/stability.py`): `stability_suite(gb, prompt, correct, incorrect, n_bootstrap=100)` runs bootstrap F1 estimation, prompt perturbation robustness, and token-swap sensitivity. Returns `StabilityResult` with confidence intervals and `summary_stats()`. Exported: `stability_suite, StabilityResult`.
- **AnnexIVReport** (`glassbox/compliance.py`): `AnnexIVReport` class generates the full 9-section EU AI Act Annex IV technical documentation package. `to_json()`, `to_markdown()`, `to_model_card()` export formats. `add_analysis()` attaches circuit results. Exported: `AnnexIVReport`.
- **REST API** (`glassbox/api.py`): FastAPI-based REST endpoint. `/analyze`, `/compliance-report`, `/audit-log`, `/health` routes. `pip install 'glassbox-mech-interp[api]'`.
- **DeploymentContext enum** (`glassbox/compliance.py`): `FINANCIAL_SERVICES`, `HEALTHCARE`, `HR_EMPLOYMENT`, `EDUCATION`, `LEGAL`, `OTHER_HIGH_RISK` for context-aware risk classification and Annex IV narrative. Exported: `DeploymentContext`.

### Changed
- `GlassboxV2` now accepts `model_name` kwarg for Annex IV auto-population.
- README: Added black-box audit section. Updated live services table to v3.2.0.

---

## [3.1.0] — 2026-03-20

### Added
- **CircuitDiff** (`glassbox/circuit_diff.py`): Mechanistic diff between two model versions
  or checkpoints. `CircuitDiff(gb_a, gb_b).diff(prompt, correct, incorrect)` returns a
  `CircuitDiffResult` with added/removed/shared heads, Jaccard stability score, per-head
  attribution delta, and F1 delta. `batch_diff()` for multi-prompt stability analysis.
  `summary_stats()` aggregates stability mean/std and most-commonly-added/removed heads.
  `to_markdown()` generates PR-ready audit report. Maps to EU AI Act Article 72 (post-market
  monitoring) and Annex IV Section 6 (lifecycle change documentation).
  Exported from top-level: `from glassbox import CircuitDiff, CircuitDiffResult`.

- **Exact sufficiency in `bootstrap_metrics()`** (`glassbox/core.py`): New `_suff_exact()`
  method computes exact causal sufficiency via positive ablation — keeps only circuit heads
  active, corrupts all other heads, measures preserved logit difference. `bootstrap_metrics()`
  now takes `exact_suff=True` (default) to use this method. Return dict includes
  `meta.exact_suff` and `meta.suff_is_approx` fields. Reproducibility note documented in
  docstring: seed=42, GPT-2 small, Apple M2 Pro, PyTorch 2.2.0, TransformerLens 1.19.0.
  This resolves the discrepancy between Taylor approx (~80%) and exact (~100%) sufficiency.

- **Custom SAE upload** (`glassbox/sae_attribution.py`): `SAEFeatureAttributor` now accepts
  `sae_path` parameter. Pass a single `.pt` file path (applied to all layers) or a dict
  `{layer: path}` for per-layer checkpoints. Expected checkpoint keys: `encoder_weight`,
  `encoder_bias`, `decoder_weight`, `decoder_bias`. New `_CustomSAE` internal class mirrors
  the sae-lens encode/decode interface. sae-lens not required when using custom checkpoints.
  Enables SAE attribution for fine-tuned, custom, or non-public models.

- **OpenTelemetry tracing** (`glassbox/telemetry.py`): `setup_telemetry(service_name, endpoint)`
  initialises OTLP trace export. `instrument_glassbox(gb)` monkey-patches `analyze()` to emit
  a span per call with attributes: `glassbox.model`, `glassbox.grade`, `glassbox.f1`,
  `glassbox.circuit_heads`, `glassbox.duration_ms`. `trace_span()` context manager / decorator
  for custom instrumentation. Supports Jaeger, Honeycomb, Datadog OTLP, Grafana Tempo.
  Falls back to no-op silently if opentelemetry-sdk is not installed.
  Install: `pip install 'glassbox-mech-interp[telemetry]'`.
  Exported: `setup_telemetry`, `teardown_telemetry`, `trace_span`, `instrument_glassbox`,
  `is_telemetry_enabled`, `TelemetryConfig`.

### Changed
- `bootstrap_metrics()` default behaviour changed: sufficiency is now exact (`exact_suff=True`)
  rather than Taylor approximation. Pass `exact_suff=False` to restore prior behaviour.
- README: Added v3.1.0 "What's New" section. Updated live services table to v3.1.0.
  Added grade scale research-defined caveat. Added black-box behavioural proxy caveat.
  Added hosted API / Render free-tier disclaimer. Added scaling roadmap in roadmap section.

### Documentation
- Comprehensive legal hardening: Legal Notices & Regulatory Disclaimer (9 subsections),
  GDPR Project & Privacy Notice with Impressum (§5 TMG), trademark disclaimer.
  Legal NOTICE blocks in all compliance-facing modules (compliance.py, risk_register.py,
  bias.py, audit_log.py). CONTRIBUTING.md legal contribution guidelines.

---

## [3.0.0] — 2026-03-20

### Added
- **Bias Analysis Module** (`glassbox/bias.py`): `BiasAnalyzer` class with three EU AI Act
  Article 10(2)(f)-compliant tests. `counterfactual_fairness_test()` swaps demographic
  attributes in prompt templates to measure probability shift (parity gap). `demographic_parity_test()`
  computes positive outcome rates across groups and flags disparity above threshold.
  `token_bias_probe()` detects stereotypical associations between demographic and role tokens.
  All methods work offline (pre-computed logprobs dicts) or online (live `model_fn`).
  `BiasReport` aggregates results into an Annex IV Section 5 markdown report.
  Exported from top-level: `from glassbox import BiasAnalyzer, BiasReport`.
- **Webhooks** (`api/main.py`): Full webhook registration system. `POST /v1/webhooks`
  registers a callback URL with event filters (`job.completed`, `job.failed`) and optional
  HMAC-SHA256 signing secret. `GET /v1/webhooks` lists registered webhooks. `DELETE /v1/webhooks/{id}`
  and `PATCH /v1/webhooks/{id}` manage them. Payloads include `X-Glassbox-Event` and
  `X-Glassbox-Signature` headers. Delivery tracked per webhook (`delivery_count`,
  `last_delivery_status`).
- **Circuit SVG Export** (`dashboard/compliance_dashboard.html`): "Download SVG" button
  in the D3 circuit graph panel. Exports `glassbox-circuit.svg` with inlined styles
  and dark background — ready for paper figures.
- **Multi-Audit History Panel** (`dashboard/compliance_dashboard.html`): Toggleable
  "Audit History" panel with F1-over-time Chart.js line chart (grade C threshold line),
  grade distribution bar chart, and audit table. "Load from API" button fetches
  `GET /v1/audit/reports`. Demo data shows D→C→C→B→B grade trajectory.
- **Risk Register** (`glassbox/risk_register.py`): `RiskRegister` class persists compliance
  risks across audit sessions to a JSON file. `ingest_annex_report()` auto-extracts risks from
  any `AnnexIVReport`. Deduplication by description+model, occurrence counting, severity
  ordering, status tracking (`open | mitigated | accepted | escalated`), `trend_summary()`
  for dashboards, `to_markdown()` for PR comments and report embedding. Maps to EU AI Act
  Article 9 (risk management system) and Annex IV Section 5.
  Exported from top-level: `from glassbox import RiskRegister, RiskEntry`.
- **Test suites** (`tests/test_audit_log.py`, `tests/test_widget.py`): 76 passing tests.
  Full offline coverage of AuditLog hash chain, CircuitWidget/HeatmapWidget HTML rendering.

### Changed
- `glassbox/__init__.py`: Version 3.0.0. `BiasAnalyzer`, `BiasReport`, result dataclasses
  added to public API and `__all__`.
- `pyproject.toml`: Version 3.0.0. Description updated.
- `api/main.py`: `_WEBHOOK_STORE` added. `_fire_webhooks()` wired into async job completion.

---

## [2.9.0] — 2026-03-20

### Added
- **Tamper-evident Audit Log** (`glassbox/audit_log.py`): `AuditLog` class persists every
  compliance audit to an append-only JSONL file. Each record carries a SHA-256 hash of the
  previous entry, forming a hash chain that `verify_chain()` can validate. Supports
  `export_csv()` and `export_json()` for regulator hand-off. `summary()` returns grade
  distribution, compliance rate, and average F1 over all stored audits.
  Now exported from `glassbox` top-level: `from glassbox import AuditLog`.
- **GitHub Actions Composite Action** (`action.yml`): `glassbox-audit@v1` drops EU AI Act
  compliance gates into any CI/CD pipeline. Inputs: `model_name`, `prompt`, `correct_token`,
  `incorrect_token`, `fail_below_grade` (default: C), `deployment_context`, `method`.
  Outputs: `grade`, `f1_score`, `sufficiency`, `comprehensiveness`, `compliance_status`,
  `report_id`. Exits 1 and emits `::error::` annotation when grade falls below threshold.
  Uses composite run steps — no Docker image needed.
- **TypeScript SDK** (`sdk/glassbox.ts`): Zero-dependency fetch-based client for the
  Glassbox REST API. Works in Node.js ≥18, Deno, Bun, and browsers. Typed request/response
  interfaces (`WhiteBoxRequest`, `BlackBoxRequest`, `AuditReport`, `AsyncJobResponse`,
  `AttentionPatternsResponse`). `GlassboxClient` class with `auditWhiteBox()`,
  `auditBlackBox()`, `startBlackBoxJob()`, `waitForJob()` (polling helper), and
  `attentionPatterns()`. `GlassboxError` carries `statusCode` + `detail`. Default export +
  named `createClient()` factory for CJS/ESM compatibility.
- **Jupyter Notebook Widgets** (`glassbox/widget.py`): `CircuitWidget` wraps `GlassboxV2`
  for one-line notebook usage. `CircuitWidget.from_prompt(gb, prompt, correct, incorrect)`
  runs the analysis and renders an inline attribution heatmap via `_repr_html_()`.
  `HeatmapWidget` accepts any pre-computed result dict (from the Python SDK or REST API).
  Both classes gracefully degrade when ipywidgets is absent. Install with:
  `pip install 'glassbox-mech-interp[jupyter]'`.
  Now exported from `glassbox` top-level: `from glassbox import CircuitWidget, HeatmapWidget`.
- **Attention Patterns API endpoint** (`api/main.py`): `POST /v1/attention-patterns` accepts
  `model_name`, `prompt`, and an optional `heads` list (e.g. `["L9H9", "L9H6"]`). Returns
  raw attention matrices, per-head entropy, last-token attention vector, and head-type
  classifications. Returns HTTP 503 on free-tier RAM exhaustion with self-hosting instructions.

### Changed
- `glassbox/__init__.py`: `AuditLog`, `AuditRecord`, `CircuitWidget`, `HeatmapWidget` added
  to public API and `__all__`. Version bumped to 2.9.0.
- `pyproject.toml`: Version bumped to 2.9.0. Description updated to reflect new features.
- Dashboard (`compliance_dashboard.html`): Full Linear/Vercel/Stripe-grade UI redesign.
  Dark-first design system with CSS custom properties, Inter + JetBrains Mono fonts,
  backdrop-blur nav, noise-texture overlay, 4-tier colour scale. All JS wiring preserved.

---

## [2.8.0] — 2026-03-17

### Added
- **Model card generator** (`glassbox/compliance.py`): `AnnexIVReport.to_model_card()` generates
  a HuggingFace-compatible `MODEL_CARD.md` with YAML frontmatter (tags: eu-ai-act, annex-iv,
  compliance, mechanistic-interpretability), compliance status table, faithfulness metrics,
  risk flags, EU AI Act article references, and citation block.
  `save_model_card(path)` convenience method writes it to disk.
- **D3 circuit graph** (`dashboard/compliance_dashboard.html`): Interactive force-directed
  graph visualising the minimum faithful circuit. Nodes = attention heads, size proportional
  to attribution score, colour mapped by layer, gold border on high-attribution heads.
  Draggable, hover tooltips, layer-adjacency edges. Uses D3.js v7.
- **Attribution heatmap** (`dashboard/compliance_dashboard.html`): 12×12 grid of attention
  head attribution scores. Colour intensity maps to score magnitude; circuit members highlighted
  with a gold border. Demo data uses real IOI/GPT-2 results (L9H9, L9H6, L10H0, L3H0…).
- **Async job endpoint** (`api/main.py`): `POST /v1/audit/black-box/async` returns immediately
  with a `job_id`. Audit runs as a FastAPI `BackgroundTask`. Poll status via
  `GET /v1/jobs/{job_id}` (states: queued → running → completed/failed).
  `GET /v1/jobs` lists all session jobs. Accepts same `X-Provider-Api-Key` header.

### Changed
- Dashboard default API URL updated to the live Render endpoint.
- API privacy notice added to dashboard: key sent as header only, never logged, never stored.

## [2.7.0] — 2026-03-17

### Added
- **EU AI Act Compliance module** (`glassbox/compliance.py`): `AnnexIVReport` class generates
  all 9 Annex IV sections as PDF + JSON. Maps faithfulness metrics to Article 13.
  Explainability grades A–D with exact thresholds. 26/26 tests passing.
- **Black-box auditor** (`glassbox/audit.py`): `BlackBoxAuditor` audits any model via API
  (OpenAI, Anthropic, Together, Groq, Azure, custom endpoint) — no model weights needed.
  Uses counterfactual probing, sensitivity sweeps, consistency testing. Zero extra dependencies.
- **REST API** (`api/main.py`): FastAPI app with `POST /v1/audit/analyze` (white-box),
  `POST /v1/audit/black-box`, `GET /v1/audit/report/{id}`, `GET /v1/audit/pdf/{id}`,
  `GET /dashboard` (serves compliance UI), `GET /docs` (Swagger UI).
- **Compliance dashboard** (`dashboard/compliance_dashboard.html`): Full web UI for compliance
  officers. Demo mode with real IOI/GPT-2 data — works with zero backend.
- **Dockerfile** + **render.yaml**: production-ready container, one-click Render deploy.
- **Live deployment**: API live at `https://glassbox-ai-2-0-mechanistic.onrender.com`.

### Security
- API key moved from request body to `X-Provider-Api-Key` header — never logged or stored.
- `_StripKeyFilter` log handler scrubs any accidental key-shaped strings from all log output.
- `SECURITY.md` added with full key handling documentation, GDPR note, self-hosting guide.

### Fixed
- `api/main.py`: version string was hardcoded `2.6.0`; now reads from `glassbox.__version__`.
- README restructured: TOC added, section order fixed, REST API section added, Dashboard
  section updated to reflect live URL, benchmark numbers marked as preliminary.
- All EU AI Act article references verified against final Regulation (EU) 2024/1689 text.
- Penalty claim updated to include Article 99(4) citation and EUR-Lex link.

---

## [2.6.0] — 2026-03-17

### Fixed

- **Version sync** — `glassbox/__init__.py` `__version__` was hardcoded as `"2.3.0"` while
  `pyproject.toml` was at `2.5.2`. Both are now `2.6.0` and will track together going forward.
- **`publish.yml` cleanup** — removed all diagnostic Macaroon-decoding code introduced while
  debugging PyPI OIDC 403 errors (root cause was a Pending Publisher UUID mismatch, now fixed
  at the PyPI project settings level). Workflow now uses the official `pypa/gh-action-pypi-publish@release/v1`
  action — the simplest and most reliable path.
- **`CITATION.cff`** — title and abstract still referenced `Glassbox 2.3`; updated to `2.6`.
  Both `version:` fields updated from `2.5.2` to `2.6.0`.
- **`cli.py`** — CLI banner and argparse description both hardcoded `2.3`; updated to `2.6`.
- **`requirements.txt`** — removed `scipy` (not imported anywhere; Kendall τ-b is implemented
  without it), `streamlit` and `plotly` (dashboard-only, not part of the core package),
  and `pytest` (dev-only). File now mirrors `pyproject.toml` core and dev deps.
- **`deploy_hf.yml`** — heredoc for HuggingFace Space `requirements.txt` was indented 10 spaces
  inside the YAML `run:` block; those spaces were being written verbatim into the file, making
  every package name invalid for `pip install`. Replaced with explicit `echo` statements.
  Updated `glassbox-mech-interp>=2.1.0` → `>=2.6.0`.
- **`README.md`** — feature table header updated from `v2.3.0` to `v2.6.0`.
- **`dist/` cleanup** — removed stale `v2.2.0` wheel and sdist that were committed to git
  despite `dist/` being listed in `.gitignore`.

---

## [2.3.0] — 2025-07-01

### Added

**SAE Feature Attribution** (`glassbox/sae_attribution.py`) — new module.
Bridges circuit-level (attribution patching, EAP) and feature-level
(SAEs, superposition) interpretability. Two methods:
- `SAEFeatureAttributor.attribute()` — decomposes residual stream at each
  layer into sparse feature activations and scores each feature by its
  logit-difference contribution. Links directly to Neuronpedia for each
  active feature.
- `SAEFeatureAttributor.attribute_circuit_heads()` — head-scoped SAE
  attribution: which sparse features are activated by each circuit head?
  (Linear approximation; see docstring.)
Requires: `pip install sae-lens` (optional dep). Supports GPT-2 small
via Joseph Bloom's pretrained residual-stream SAEs.
References: Bloom et al. (2024), Bricken et al. (2023), Cunningham et al. (2023).

**Head Composition Analysis** (`glassbox/composition.py`) — new module.
Computes Q/K/V composition scores between attention head pairs (Elhage et al. 2021, §3.2).
- `HeadCompositionAnalyzer.q_composition_score(sl, sh, rl, rh)` — Q-composition.
- `HeadCompositionAnalyzer.k_composition_score(...)` — K-composition.
- `HeadCompositionAnalyzer.v_composition_score(...)` — V-composition.
- `HeadCompositionAnalyzer.composition_matrix(senders, receivers, kind)` — full matrix.
- `HeadCompositionAnalyzer.full_circuit_composition(circuit, kind, min_score)` — all pairwise scores within a circuit.
- `HeadCompositionAnalyzer.all_composition_scores(circuit)` — Q+K+V in one call.
No extra dependencies. Always available.

**Token Attribution** (`GlassboxV2.token_attribution()`) — added to `core.py`.
Per-input-token attribution via gradient × embedding (Simonyan et al. 2014).
Scores each token by its signed contribution to logit(target) - logit(distractor).
Returns `token_ids`, `token_strs`, `attributions`, `abs_attributions`, `top_tokens`.
Cost: 1 forward + 1 backward pass.

**Attention Pattern Analysis** (`GlassboxV2.attention_patterns()`) — added to `core.py`.
Returns full attention matrices, per-head entropy, last-token attention row, and
heuristic head-type classification: `induction_candidate`, `previous_token`,
`focused`, `uniform`, `self_attn`, `mixed`.
Cost: 1 forward pass.

**Expanded test suite** — 6 new test classes in `tests/test_engine.py`:
- `TestLogitLens` (8 tests) — logit_lens() correctness and mathematical consistency.
- `TestEdgeAttributionPatching` (8 tests) — EAP structure, score finiteness, positivity.
- `TestAttributionStability` (6 tests) — stability scores bounds, Kendall τ-b range.
- `TestTokenAttribution` (7 tests) — token attribution structure, sorting, finiteness.
- `TestAttentionPatterns` (8 tests) — patterns shape, row sums, entropy, head types.
- `TestHeadCompositionAnalyzer` (11 tests) — score bounds, causal validity, matrix shape.

### Changed
- `glassbox/__init__.py` — exports `SAEFeatureAttributor` and `HeadCompositionAnalyzer`.
- `pyproject.toml` — version 2.3.0; added `sae` optional dep group; added full
  classifiers, `arXiv Paper` and `Changelog` URLs, `ruff` and `mypy` config sections.
- `README.md` — complete rewrite. Added feature comparison table vs. TransformerLens /
  Baukit / Pyvene, full API reference, SAE and composition code examples, updated
  benchmarks section, complete citation block.
- `core.py` module docstring — added Simonyan et al. 2014, Olsson et al. 2022,
  Bloom et al. 2024 references; updated complexity table with new methods.
- `core.py` `GlassboxV2` class docstring — added all new method signatures.

---

## [2.2.0] — 2025-05-15

### Added

**Logit Lens** (`GlassboxV2.logit_lens()`) — implements nostalgebraist (2020) extended
with per-head direct effects (Elhage et al. 2021, §2.3).
- Projects residual stream at each layer through ln_final + unembed to show how
  predictions crystallise layer by layer.
- Per-head direct effects via virtual weights: `direct(l,h) = (W_O[l,h] @ z[l,h,-1]) · unembed_dir`.
- Optional inclusion in `analyze()` via `include_logit_lens=True`.
- 1 forward pass.

**Edge Attribution Patching** (`GlassboxV2.edge_attribution_patching()`) — implements
Syed et al. (2024). Scores every directed edge (sender → receiver) in the computation
graph. Formula: `EAP(u→v) = (∂metric/∂resid_pre_v) · Δh_u`. O(3) cost.
- Strictly more informative than node-level AP: reveals which connections carry the signal.
- Gradient captured via `act.register_hook()` to avoid breaking the computation graph.

**Attribution Stability** (`GlassboxV2.attribution_stability()`) — novel metric.
- Runs attribution over K random corruptions (25% token replacement).
- Per-head stability: `S(l,h) = 1 − std/(|mean| + ε)`.
- Global rank consistency: vectorised Kendall τ-b (Kendall 1938) across all C(K,2) pairs.
- No scipy dependency.

**`analyze()` updated** — `include_logit_lens: bool = False` parameter added.

### Changed
- `__version__` bumped to 2.2.0.
- Module docstring updated with new references (Dar et al. 2023, Syed et al. 2024, Kendall 1938).
- Complexity table updated.

### Infrastructure
- `.github/workflows/deploy_hf.yml` — GitHub Actions auto-sync to HuggingFace Space.
- `.github/workflows/publish.yml` — OIDC Trusted Publisher (no API tokens needed).
- PyPI package published at version 2.2.0.

---

## [2.1.0] — 2025-03-10

### Added

**MLP Attribution** (`GlassboxV2.mlp_attribution()`) — per-layer MLP contribution
via `hook_mlp_out`. Completes the circuit picture beyond attention heads. 3 passes.

**Integrated Gradients** — `attribution_patching(method="integrated_gradients")`.
Path-integral attribution (Sundararajan et al. 2017). Costs 2+n_steps passes.
Set `method="integrated_gradients"` in `analyze()` to propagate through.

**Bootstrap 95% CI** (`GlassboxV2.bootstrap_metrics()`) — nonparametric bootstrap
over N prompt triples. Returns mean, std, ci_lo, ci_hi for Suff/Comp/F1.

---

## [2.0.0] — 2025-01-20

### Added

- `GlassboxV2` class — full rewrite of the interpretability engine.
- Attribution patching (Taylor, O(3)) — Nanda et al. (2023).
- Minimum faithful circuit discovery (greedy forward/backward pruning).
- Faithfulness metrics: sufficiency, comprehensiveness, F1 (ERASER framework).
- Functional Circuit Alignment Score (FCAS) — novel cross-model metric.
- Interactive Streamlit dashboard.
- PyPI package `glassbox-mech-interp`.
- CLI: `glassbox-ai analyze`.

### Removed
- `GlassboxEngine` (v1.x class) — replaced by `GlassboxV2`.
  Shim alias kept in `alignment.py` for back-compat.

---

## [1.0.0] — 2024-09-01

Initial release. Basic attribution patching for GPT-2 small on IOI task.
