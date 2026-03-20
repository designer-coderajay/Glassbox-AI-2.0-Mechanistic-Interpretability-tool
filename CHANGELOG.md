# Changelog

All notable changes to Glassbox are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
