# Security & Privacy

## API Key Handling

Glassbox never stores, logs, or retains your model provider API keys.

**Architecture (v3.6.0+):**
- API keys are passed via the `X-Provider-Api-Key` HTTP request header — not in the request body
- All log filters explicitly scrub any string matching known key patterns before writing to disk
- Keys are used in-memory only for the duration of the HTTP request
- Keys are never written to `_REPORT_STORE`, PDF reports, or any persistent storage
- Compliance reports (JSON/PDF) contain zero credential data — only model behaviour metrics

**Verification:** You can confirm this yourself by reading [`api/main.py`](api/main.py):
- `_StripKeyFilter` class: active on all log handlers
- `_REPORT_STORE` write: contains only `json`, `mode`, `created_at`
- `BlackBoxRequest` model: contains no `api_key` field

---

## Recommended: Self-Hosting

For production compliance audits, run Glassbox locally. Your keys never leave your infrastructure:

```bash
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
docker build -t glassbox .
docker run -p 8000:8000 glassbox
```

The hosted instance at `https://designer-coderajay-glassbox-ai-2-0-mechanistic-interpretability-tool.hf.space` is provided for evaluation and testing only. Do not use the hosted instance for production compliance workflows involving sensitive data.

---

## Reporting Vulnerabilities

Please report security issues **privately** to: [mahale.ajay01@gmail.com](mailto:mahale.ajay01@gmail.com)

Do **not** open a public GitHub issue for security vulnerabilities.

---

## Data Processing and GDPR

**No personal data is intentionally collected.** Glassbox processes:
- Model prompts and outputs (text provided by the user)
- API keys (in-memory only, never persisted — see above)
- Compliance metrics and scores (logged to the local report store with a random report ID)

**GDPR applicability (Regulation (EU) 2016/679):**

If you use Glassbox to audit an AI system that processes personal data (e.g., credit scoring prompts containing applicant details), you are the data controller for that processing. Glassbox acts as a data processor only for the duration of a live API call to the hosted instance.

For self-hosted deployments, all processing stays within your own infrastructure and Glassbox's author is not a data processor.

API keys are not personal data under GDPR Article 4(1) — they do not identify a natural person.

**Legitimate interest (Article 6(1)(f)):** Where the hosted API processes text snippets incidentally, the processing is for the legitimate interest of providing the requested compliance documentation service, and no data is retained beyond the individual request.

---

## Legal Jurisdiction and Governing Law

This project is developed under the laws of the Federal Republic of Germany. EU law (including Regulation (EU) 2024/1689 and Regulation (EU) 2016/679) applies directly.

---

## Disclaimer on Regulatory Adequacy

**Nothing in this security notice, or in the Glassbox software, constitutes a guarantee that use of Glassbox satisfies any obligation under Regulation (EU) 2024/1689 (EU AI Act), GDPR, or any other applicable law.** Whether your deployment of Glassbox in a compliance workflow satisfies your specific regulatory obligations is a matter for qualified legal counsel.

See also: [`README.md — Legal Notices & Regulatory Disclaimer`](README.md#legal-notices--regulatory-disclaimer)

---

## References

- [EU AI Act Regulation (EU) 2024/1689](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) — Article 12 (logging), Article 99(4) (penalties)
- [GDPR Regulation (EU) 2016/679](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679) — Articles 4, 6, 28, 35
