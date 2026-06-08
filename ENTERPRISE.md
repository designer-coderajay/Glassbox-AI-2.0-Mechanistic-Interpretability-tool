# Enterprise Onboarding — Glassbox AI v4.3.1

This guide is for engineering and compliance teams at EU-regulated organisations
deploying AI in finance, healthcare, employment, or critical infrastructure.

---

## What you get

- **Annex IV automation** — all 9 sections generated in under 2 seconds (§8, the EU Declaration of Conformity, is a provider-signed legal reference)
- **Any-prompt support** — credit scoring, medical triage, HR screening, legal, dialogue
- **Billion-parameter support** — Llama-3-70B, Mistral, Gemma with gradient checkpointing
- **CI/CD gate** — CircuitDiff detects model behaviour changes on every deployment
- **Air-gapped deployment** — fully self-contained Docker image, no external calls
- **Signed evidence vaults** — tamper-evident audit trail for regulators

---

## Onboarding in 4 steps

### Step 1 — Install

```bash
# Production install with compliance extras
pip install "glassbox-mech-interp[compliance]>=4.3.0"

# For large models (7B+)
pip install "glassbox-mech-interp[compliance,large]>=4.3.0"
```

### Step 2 — Run your first compliance audit

Replace the example prompt with your actual production use case:

```python
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2
from glassbox.compliance import AnnexIVReport, DeploymentContext

# Load your model
model = HookedTransformer.from_pretrained("your-model-name")
gb    = GlassboxV2(model)

# Run analysis — auto-selects best corruption strategy for your domain
result = gb.analyze(
    prompt    = "Your production prompt here",
    correct   = " expected_output_token",
    incorrect = " wrong_output_token",
)

# Generate Annex IV evidence package
report = AnnexIVReport(
    model_name         = "YourModelV1",
    system_purpose     = "Description of what your AI system does",
    provider_name      = "Your Legal Entity Name",
    provider_address   = "Your Registered Address",
    deployment_context = DeploymentContext.FINANCIAL_SERVICES,  # or HEALTHCARE, EMPLOYMENT, etc.
)
report.add_analysis(result)
report.to_pdf("annex_iv_report.pdf")   # hand this to your notified body
report.to_json("annex_iv_report.json") # for regulator submission systems
```

### Step 3 — Add to CI/CD (Article 72 post-market monitoring)

```yaml
# .github/workflows/compliance.yml
name: AI Act Compliance Gate

on:
  push:
    branches: [main]

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: designer-coderajay/glassbox-mech@v4
        with:
          model:         your-model-name
          prompt:        "Your production prompt"
          correct:       " correct_token"
          incorrect:     " wrong_token"
          fail_below_f1: 0.40
```

The action uploads a full Annex IV report as a build artifact on every commit.
If faithfulness F1 drops below `fail_below_f1`, the CI job fails — alerting
your team before the model goes to production.

### Step 4 — Air-gapped / on-premises deployment

For regulated environments where no external network calls are permitted:

```bash
# Build the standalone scan container
git clone https://github.com/designer-coderajay/glassbox-mech
docker build -f scan.Dockerfile -t glassbox-scan:4.3.0 .

# Export for air-gapped transfer
docker save glassbox-scan:4.3.0 | gzip > glassbox-scan-4.3.0.tar.gz

# On the air-gapped machine
docker load < glassbox-scan-4.3.0.tar.gz

# Run compliance scan
docker run --rm -v $(pwd)/output:/output \
    -v /path/to/your/model:/model \
    glassbox-scan:4.3.0 \
    --model-path /model \
    --prompt "Your production prompt" \
    --correct " correct_token" \
    --incorrect " wrong_token" \
    --purpose "Your system purpose" \
    --provider "Your organisation" \
    --output /output/annex_iv.pdf
```

---

## What the Annex IV report covers

| Section | EU AI Act Reference | Generated from |
|---|---|---|
| 1. General description | Art. 13(3)(a) | Model name, version, purpose, deployment context |
| 2. Design & development | Art. 10, 11(1)(d) | Architecture metadata, training description |
| 3. Monitoring & oversight | Art. 9(6), 14 | CircuitDiff configuration, oversight measures |
| 4. Explainability assessment | Art. 13 | Circuit heads, faithfulness F1, grade A–F |
| 5. Data requirements | Art. 10 | Data quality statement, bias probe results |
| 6. Risk assessment | Art. 9 | Risk register entries, failure modes |
| 7. Accuracy & robustness | Art. 15 | Task accuracy, confidence calibration |
| 8. Post-market monitoring | Art. 72 | CircuitDiff baseline, alert configuration |

---

## Pricing

| Plan | Price | Audits | Features |
|---|---|---|---|
| **Open source** | Free (MIT) | Unlimited self-hosted | Full engine, all frameworks |
| **Pro** | €499/month | 500 / month | Signed vaults, API, dashboard |
| **Enterprise** | From €24,000/year | Unlimited | On-prem Docker, SLA, ISO 42001, NIST AI RMF, legal indemnification support |

Contact: **mahale.ajay01@gmail.com**

---

## Legal notice

Glassbox-generated documentation aids preparation of technical files under EU AI Act
Article 11. It is a documentation tool, not a legal certification. Whether your system
qualifies as high-risk under Article 6/Annex III, and whether this documentation
satisfies all applicable obligations, must be confirmed by qualified legal counsel
and/or a notified body (Article 43).

Non-compliance fine: up to €35M or 7% of global annual turnover (Article 99).
