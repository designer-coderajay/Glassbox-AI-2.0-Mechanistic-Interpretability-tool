#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ajay Pravin Mahale
"""
scripts/glassbox_ci_audit.py
============================
Glassbox EU AI Act compliance audit script for CI/CD pipelines.
Called by .github/workflows/glassbox-audit.yml on every model push.

Environment variables (set in the workflow):
    GLASSBOX_MODEL_NAME        TransformerLens model name (e.g. "gpt2")
    GLASSBOX_PROMPT            Audit prompt string
    GLASSBOX_CORRECT_TOKEN     Expected correct token
    GLASSBOX_INCORRECT_TOKEN   Contrastive incorrect token
    GLASSBOX_MIN_SUFFICIENCY   Minimum acceptable sufficiency (default 0.70)
    GLASSBOX_PROVIDER_NAME     Organisation name for Annex IV report
    GLASSBOX_USE_CASE          Deployment use case description
    GLASSBOX_DEPLOYMENT_CTX    Deployment context string
    OVERRIDE_MODEL             Optional workflow_dispatch override
    OVERRIDE_THRESHOLD         Optional workflow_dispatch override
    GITHUB_SHA                 Commit SHA (set by GitHub Actions)
    GITHUB_REPOSITORY          repo slug (set by GitHub Actions)
    GITHUB_RUN_ID              Actions run ID (set by GitHub Actions)
    GITHUB_OUTPUT              Path to GitHub Actions output file
    GITHUB_STEP_SUMMARY        Path to GitHub Actions step summary file

Exit codes:
    0   Audit passed (sufficiency >= threshold)
    1   Audit failed (sufficiency < threshold) or fatal error
"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Read environment
# ---------------------------------------------------------------------------
model_name   = os.environ.get("OVERRIDE_MODEL") or os.environ.get("GLASSBOX_MODEL_NAME", "gpt2")
prompt       = os.environ.get("GLASSBOX_PROMPT", "When Mary and John went to the store, John gave a drink to")
correct      = os.environ.get("GLASSBOX_CORRECT_TOKEN", " Mary")
incorrect    = os.environ.get("GLASSBOX_INCORRECT_TOKEN", " John")
min_suff_str = os.environ.get("OVERRIDE_THRESHOLD") or os.environ.get("GLASSBOX_MIN_SUFFICIENCY", "0.70")
min_suff     = float(min_suff_str)
provider     = os.environ.get("GLASSBOX_PROVIDER_NAME", "Unknown")
use_case     = os.environ.get("GLASSBOX_USE_CASE", "CI/CD compliance audit")
deploy_ctx   = os.environ.get("GLASSBOX_DEPLOYMENT_CTX", "other_high_risk")
sha          = os.environ.get("GITHUB_SHA", "unknown")[:8]
repo         = os.environ.get("GITHUB_REPOSITORY", "unknown")
run_id       = os.environ.get("GITHUB_RUN_ID", "")
run_url      = f"https://github.com/{repo}/actions/runs/{run_id}"
gh_output    = os.environ.get("GITHUB_OUTPUT", "")
gh_summary   = os.environ.get("GITHUB_STEP_SUMMARY", "")

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
print(f"[Glassbox] Loading model: {model_name}")

try:
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2
    from glassbox.explain import NaturalLanguageExplainer

    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    gb = GlassboxV2(model)

    print("[Glassbox] Running analysis...")
    t0 = time.time()
    result = gb.analyze(
        prompt=prompt,
        correct=correct,
        incorrect=incorrect,
    )
    elapsed = round(time.time() - t0, 2)

except Exception as e:
    print(f"[Glassbox] FATAL: {e}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Extract metrics
# ---------------------------------------------------------------------------
faith   = result.get("faithfulness", {})
suff    = faith.get("sufficiency", 0.0)
comp    = faith.get("comprehensiveness", 0.0)
f1      = faith.get("f1", 0.0)
n_heads = result.get("n_heads", 0)

if suff >= 0.90:
    grade = "Excellent"
elif suff >= 0.75:
    grade = "Good"
elif suff >= 0.50:
    grade = "Marginal"
else:
    grade = "Poor"

passed = suff >= min_suff

# ---------------------------------------------------------------------------
# Plain-English summary
# ---------------------------------------------------------------------------
ex = NaturalLanguageExplainer(verbosity="standard")
plain_summary = ex.explain(result, model_name=model_name, prompt=prompt)

# ---------------------------------------------------------------------------
# Write reports
# ---------------------------------------------------------------------------
os.makedirs("glassbox-reports", exist_ok=True)

report_data = {
    "glassbox_version":         "3.3.0",
    "commit_sha":               sha,
    "timestamp_utc":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_name":               model_name,
    "provider":                 provider,
    "use_case":                 use_case,
    "prompt":                   prompt,
    "correct_token":            correct,
    "incorrect_token":          incorrect,
    "faithfulness":             faith,
    "n_heads":                  n_heads,
    "grade":                    grade,
    "compliance_status":        "COMPLIANT" if passed else "NEEDS_REVIEW",
    "min_sufficiency_threshold": min_suff,
    "elapsed_seconds":          elapsed,
    "regulation":               "Regulation (EU) 2024/1689 - AI Act Annex IV",
    "plain_english_summary":    plain_summary,
    "ci_run_url":               run_url,
}

report_json_path = f"glassbox-reports/annex-iv-{sha}.json"
with open(report_json_path, "w") as f:
    json.dump(report_data, f, indent=2)

html_content = ex.to_html(result, model_name=model_name, prompt=prompt)
report_html_path = f"glassbox-reports/annex-iv-{sha}.html"
with open(report_html_path, "w") as f:
    f.write(html_content)

# ---------------------------------------------------------------------------
# Write GitHub Actions output variables
# ---------------------------------------------------------------------------
if gh_output:
    with open(gh_output, "a") as out:
        out.write(f"sufficiency={suff:.4f}\n")
        out.write(f"grade={grade}\n")
        out.write(f"n_heads={n_heads}\n")
        out.write(f"passed={'true' if passed else 'false'}\n")
        out.write(f"report_json={report_json_path}\n")
        out.write(f"report_html={report_html_path}\n")
        out.write(f"elapsed={elapsed}\n")

# ---------------------------------------------------------------------------
# Write GitHub Actions step summary (Markdown)
# Safe to use pipe characters here — this file is written at runtime,
# not embedded in the YAML.
# ---------------------------------------------------------------------------
if gh_summary:
    status_icon = "PASSED" if passed else "FAILED"
    with open(gh_summary, "a") as f:
        f.write(f"## Glassbox Compliance Audit - {grade}\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Model | `{model_name}` |\n")
        f.write(f"| Commit | `{sha}` |\n")
        f.write(f"| Sufficiency | `{suff:.1%}` |\n")
        f.write(f"| Comprehensiveness | `{comp:.1%}` |\n")
        f.write(f"| F1 | `{f1:.1%}` |\n")
        f.write(f"| Circuit Heads | {n_heads} |\n")
        f.write(f"| Grade | **{grade}** |\n")
        f.write(f"| Threshold | {min_suff:.0%} |\n")
        f.write(f"| Status | {status_icon} |\n")
        f.write(f"| Elapsed | {elapsed}s |\n\n")
        f.write(f"### Plain English Summary\n\n{plain_summary}\n\n")
        f.write("---\n")
        f.write("*Generated by [Glassbox AI](https://github.com/ajaymahale/glassbox-mech-interp)")
        f.write(" v3.3.0 - Regulation (EU) 2024/1689 - AI Act Annex IV*\n")

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
icon = "PASSED" if passed else "FAILED"
print(f"[Glassbox] Grade: {grade}  Sufficiency: {suff:.1%}  Heads: {n_heads}  {icon}")

if not passed:
    print(
        f"\n[Glassbox] CI FAILED: Sufficiency {suff:.1%} < threshold {min_suff:.0%}.\n"
        f"           Increase model quality or lower GLASSBOX_MIN_SUFFICIENCY.",
        file=sys.stderr,
    )
    sys.exit(1)
