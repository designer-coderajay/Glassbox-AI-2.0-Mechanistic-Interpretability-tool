# Security & Privacy

## API Key Handling

Glassbox never stores, logs, or retains your model provider API keys.

**Architecture (v2.7.0+):**
- API keys are passed via the `X-Provider-Api-Key` HTTP request header — not in the request body
- All log filters explicitly scrub any string matching known key patterns before writing to disk
- Keys are used in-memory only for the duration of the HTTP request
- Keys are never written to `_REPORT_STORE`, PDF reports, or any persistent storage
- Compliance reports (JSON/PDF) contain zero credential data — only model behaviour metrics

**Verification:** You can confirm this yourself by reading [`api/main.py`](api/main.py):
- `_StripKeyFilter` class: active on all log handlers
- `_REPORT_STORE` write: contains only `json`, `mode`, `created_at`
- `BlackBoxRequest` model: contains no `api_key` field

## Recommended: Self-Hosting

For production compliance audits, run Glassbox locally. Your keys never leave your infrastructure:

```bash
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
docker build -t glassbox .
docker run -p 8000:8000 glassbox
```

The hosted instance at `https://glassbox-ai-2-0-mechanistic.onrender.com` is provided for evaluation and testing only.

## Reporting Vulnerabilities

Please report security issues privately to: mahale.ajay01@gmail.com

Do not open a public GitHub issue for security vulnerabilities.

## Legal Jurisdiction

This project is developed and operated under EU/German law. The GDPR applies to all personal data processed through this service. No personal data is intentionally collected. API keys are not personal data under GDPR Article 4.

## References

- [EU AI Act Regulation (EU) 2024/1689](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) — Article 99(4): penalties for non-compliance
- [GDPR Regulation (EU) 2016/679](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679)
