# Glassbox MCP Server

Mechanistic interpretability tools via the Model Context Protocol.
Run circuit discovery, faithfulness metrics, and EU AI Act compliance reports
from any Claude instance or MCP-compatible client.

## Tools

| Tool | Description |
|------|-------------|
| `glassbox_circuit_discovery` | Attribution patching — find which heads drive a prediction |
| `glassbox_faithfulness_metrics` | Compute sufficiency, comprehensiveness, F1, compliance grade |
| `glassbox_compliance_report` | Generate EU AI Act Annex IV report (all 9 sections) |
| `glassbox_attention_patterns` | Get attention weights for a specific layer/head |
| `glassbox_logit_lens` | Layer-by-layer prediction buildup analysis |

## Install

```bash
pip install mcp glassbox-mech-interp transformer-lens torch
python server.py
```

## Connect to Claude

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "glassbox": {
      "command": "python",
      "args": ["/path/to/glassbox-mcp-server/server.py"]
    }
  }
}
```

## Reference

Paper: arXiv 2603.09988 (Mahale, 2026)
PyPI: `glassbox-mech-interp`
