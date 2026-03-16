# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.3.x   | ✅ Active support  |
| 2.2.x   | ⚠️ Security fixes only |
| < 2.2   | ❌ End of life     |

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, report them privately via one of these channels:

1. **GitHub Private Security Advisory** (preferred):
   Go to the [Security tab](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/security/advisories/new)
   and open a private advisory.

2. **Email**: Send details to `mahale.ajay01@gmail.com`
   with subject line `[SECURITY] Glassbox — <brief description>`.

### What to include

- A clear description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact
- Any suggested fix (optional but appreciated)

### Response timeline

| Step                    | Time         |
|-------------------------|--------------|
| Acknowledgement         | Within 48 h  |
| Initial triage          | Within 5 days |
| Fix + coordinated disclosure | Within 30 days |

## Scope

Glassbox runs **entirely locally** — it performs no network requests and stores
no user data. The primary attack surface is:

- **Malicious model files** loaded via `HookedTransformer.from_pretrained()`.
  Glassbox inherits TransformerLens's trust model for checkpoint loading.
  Only load models from sources you trust.
- **Pickle deserialization** in PyTorch checkpoint loading (upstream risk,
  not specific to Glassbox).
- **SAE downloads** via sae-lens: only models from Neuronpedia's official
  registry are used by default.

## Hall of Fame

We publicly thank security researchers who responsibly disclose vulnerabilities.
