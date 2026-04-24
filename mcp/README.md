# glassbox-mcp

MCP server for [Glassbox](https://repo-ashen-psi.vercel.app) — mechanistic interpretability + EU AI Act Annex IV compliance tools, exposed via the [Model Context Protocol](https://modelcontextprotocol.io).

Connect Claude (or any MCP-compatible client) to Glassbox with a single pip install. No cloning, no manual server setup.

## Install

```bash
pip install glassbox-mcp
```

## Connect to Claude Desktop

Add to `~/.claude/claude_desktop_config.json` (macOS/Linux) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "glassbox": {
      "command": "glassbox-mcp"
    }
  }
}
```

Restart Claude Desktop. The Glassbox tools appear immediately.

## Tools

| Tool | What it does |
|------|-------------|
| `glassbox_circuit_discovery` | Attribution patching — find which attention heads causally drive a prediction |
| `glassbox_faithfulness_metrics` | Compute sufficiency, comprehensiveness, F1, and EU AI Act explainability grade (A–D) |
| `glassbox_compliance_report` | Generate a full EU AI Act Annex IV evidence package (all 9 sections) |
| `glassbox_attention_patterns` | Get attention weight heatmap for any layer/head |
| `glassbox_logit_lens` | Layer-by-layer residual stream projection — see how predictions build up |

## Supported Models

- `gpt2`, `gpt2-medium`, `gpt2-large`
- `EleutherAI/pythia-70m`, `pythia-160m`, `pythia-410m`

## Example usage in Claude

Once connected, ask Claude things like:

> *"Use Glassbox to analyse which attention heads drive GPT-2's prediction for 'The Eiffel Tower is in' → ' Paris'"*

> *"Generate an EU AI Act Annex IV compliance report for GPT-2 with provider name Acme Corp"*

> *"Show me the logit lens for GPT-2 on this sentence"*

## Requirements

- Python ≥ 3.10
- ~3 GB disk (PyTorch + TransformerLens model weights download on first use)

## Version

`4.2.6` — tracks `glassbox-mech-interp` v4.2.6

- Paper: [arXiv 2603.09988](https://arxiv.org/abs/2603.09988)
- PyPI (core library): [glassbox-mech-interp](https://pypi.org/project/glassbox-mech-interp)
- Website: [repo-ashen-psi.vercel.app](https://repo-ashen-psi.vercel.app)
- GitHub: [designer-coderajay/glassbox-mech](https://github.com/designer-coderajay/glassbox-mech)
