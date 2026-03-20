# Glassbox — Department Coverage & Roadmap
**As of v2.9.0 — March 2026**

---

## The Frame

Frontier AI labs (Anthropic, DeepMind, Meta FAIR) organise mechanistic interpretability work
across four departments. Glassbox needs to be useful to all four — both as an external tool
they'd adopt, and as the internal operating model for Glassbox itself as it grows.

---

## Department 1: AI Research & Interpretability

**Who they are.** Mechanistic interpretability researchers. SAE specialists.
People publishing on circuits, superposition, feature geometry.

**What they need from a tool.**
- Circuit discovery that doesn't take 45 minutes to run
- Statistically rigorous faithfulness metrics with CIs
- Cross-model circuit alignment (is this circuit universal?)
- SAE feature decomposition with neuronpedia links
- Composable, scriptable API — not a UI

**Glassbox coverage today.** STRONG.
- GlassboxV2: attribution patching O(3), MFC discovery, EAP, logit lens ✅
- Bootstrap 95% CIs on faithfulness ✅
- FCAS (cross-model circuit alignment) — novel metric ✅
- SAEFeatureAttributor with neuronpedia integration ✅
- Full Python API, CLI, pip installable ✅
- `gb.batch_analyze(prompts, n_workers=4)` → parallel batch (core.py line 1647) ✅
- **[v2.9.0] Jupyter widget** (`CircuitWidget`, `HeatmapWidget`) with live attribution heatmap ✅
- **[v2.9.0] `POST /v1/attention-patterns`** — per-head attention matrix API ✅

**Remaining gaps.**
- Paper-ready figure export: "Download circuit as SVG" button (dashboard, next)
- Circuit diff: compare circuits across model versions (v3.0)
- Custom SAE upload to `SAEFeatureAttributor` (v3.0)

---

## Department 2: AI Engineering & Infrastructure

**Who they are.** ML engineers building production AI systems. Platform teams.
People who need to integrate compliance checks into CI/CD pipelines.

**What they need from a tool.**
- REST API (JSON in, JSON out, deterministic) ✅
- Docker container they can run internally ✅
- Async endpoints for non-blocking audit jobs ✅
- SDKs: Python (have it), TypeScript ✅
- GitHub Actions integration ✅

**Glassbox coverage today.** STRONG.
- FastAPI REST API: all endpoints documented in Swagger ✅
- Docker container: `docker run -p 8000:8000 glassbox` ✅
- Live at Render for testing ✅
- Python package on PyPI ✅
- **[v2.9.0] `POST /v1/audit/black-box/async`** → job_id, poll `/v1/jobs/{id}` ✅
- **[v2.9.0] GitHub Action** `glassbox-audit@v1` (action.yml) — drop-in CI compliance gate ✅
- **[v2.9.0] TypeScript SDK** (`sdk/glassbox.ts`) — Node.js/Deno/Bun/browser ✅

**Remaining gaps.**
- Webhook support: POST callback when async job completes (v3.0)
- OpenTelemetry tracing for self-hosted deployments (v3.0)

---

## Department 3: Product & User Experience (Data Visualization)

**Who they are.** Product managers, UI/UX designers, data visualisation specialists.
People who translate interpretability results into decisions. Non-technical stakeholders.

**What they need from a tool.**
- Visual circuit graphs — not just numbers
- Attention heatmaps
- Interactive exploration: click a head, see what it attends to
- Shareable reports (PDF already done)
- Embeddable widgets for internal tooling

**Glassbox coverage today.** STRONG.
- Compliance dashboard: full Linear/Vercel/Stripe-grade redesign ✅
- Attribution heatmap: 12×12 grid, gold circuit highlights, hover tooltips ✅
- **[v2.9.0] D3 force-directed circuit graph** — draggable, hover tooltips ✅
- **[v2.9.0] Jupyter heatmap widget** (`CircuitWidget`) — inline notebook rendering ✅
- HF Space: Gradio with basic attention viz ⚠️ (needs refresh)

**Remaining gaps.**
- Attention pattern click-through: click a circuit head chip → full attention matrix overlay
- "Download circuit as SVG" button for the D3 graph
- HF Space refresh to match v2.9.0 UI quality

---

## Department 4: Strategy & Ethics

**Who they are.** Legal, compliance, AI governance leads. Ethics researchers.
People responsible for regulatory submissions and model governance policy.

**What they need from a tool.**
- Annex IV report (all 9 sections) ✅
- Audit trail: who ran what audit, when, on which model version ✅
- Model card generation (HuggingFace-compatible) ✅
- Bias analysis: demographic parity, counterfactual fairness tests
- Risk register: track risks identified across multiple audits

**Glassbox coverage today.** STRONG on documentation and governance foundations.
- AnnexIVReport: all 9 Annex IV sections, PDF + JSON ✅
- BlackBoxAuditor: any model, counterfactual + sensitivity ✅
- Explainability grades A–D with Article 13 mapping ✅
- Risk identification with article citations ✅
- `AnnexIVReport.to_model_card()` / `save_model_card()` — HF-compatible markdown ✅
- **[v2.9.0] `AuditLog`** — append-only JSONL with SHA-256 hash chain tamper detection ✅
  - `verify_chain()`, `export_csv()`, `export_json()`, `summary()` all implemented
  - Exported from top-level: `from glassbox import AuditLog`

**Remaining gaps.**
- Bias analysis module: demographic parity, counterfactual fairness (v3.0)
- Multi-audit comparison dashboard: compliance trajectory across model versions (v3.0)
- Risk register persistence (v3.0)

---

## Essential Roles — Current Coverage

| Role | Tools serving them | Status |
|------|-------------------|--------|
| Mechanistic Interpretability Researchers | GlassboxV2, CLI, pip, batch_analyze | ✅ Strong — Jupyter widget added v2.9 |
| SAE Specialists | SAEFeatureAttributor, neuronpedia links | ✅ Good — custom SAE upload v3.0 |
| AI/ML Engineers | FastAPI, Docker, Swagger, async jobs, TS SDK, GH Action | ✅ Strong — all closed v2.9 |
| UI/UX Designers (data viz) | Attribution heatmap, D3 circuit graph, Jupyter widget | ✅ Strong — SVG export next |
| Compliance Officers | AnnexIVReport PDF/JSON, grades A-D, AuditLog | ✅ Strong — bias analysis v3.0 |
| Legal/Strategy | Annex IV all 9 sections, risk flags, model card, audit log | ✅ Strong — multi-audit comparison v3.0 |

---

## 90-Day Build Priority (Updated)

| Version | Focus | Key deliverable | Status |
|---------|-------|----------------|--------|
| v2.7.0 | Compliance + API + Attribution heatmap | Live API, Dashboard, Heatmap viz | ✅ SHIPPED |
| v2.8.0 | Engineering + Research + UI redesign | Async jobs, D3 graph, model card, dashboard redesign | ✅ SHIPPED |
| v2.9.0 | Governance + SDK + DX | AuditLog, TS SDK, GH Action, Jupyter widget, attention API | ✅ SHIPPED |
| v3.0.0 | Enterprise | Multi-audit dashboard, SVG export, bias analysis, webhooks, SLA | 🔜 Next |

---

## The Acquisition Argument

Each department above maps to a team at Anthropic, DeepMind, or Meta FAIR.
When one of them evaluates Glassbox, they're asking: "does this replace 3 months of
internal tooling we'd otherwise have to build?"

For Anthropic's Interpretability team: yes — circuit discovery + FCAS + Jupyter widgets.
For DeepMind's Responsible AI team: yes — Annex IV compliance reports + tamper-evident audit log.
For Meta FAIR's infrastructure team: yes — REST API + Docker + async jobs + TypeScript SDK + GH Action.

As of v2.9.0: every department is STRONG. v3.0 closes the remaining enterprise gaps.
