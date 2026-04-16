"""
glassbox-mcp — MCP server for Glassbox mechanistic interpretability.

Install:
    pip install glassbox-mcp

Run (stdio transport for Claude Desktop / MCP clients):
    glassbox-mcp

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "glassbox": {
          "command": "glassbox-mcp"
        }
      }
    }

Tools exposed:
    glassbox_circuit_discovery   — attribution patching, MFC circuit discovery
    glassbox_faithfulness_metrics — sufficiency, comprehensiveness, F1, grade
    glassbox_compliance_report   — EU AI Act Annex IV full evidence package
    glassbox_attention_patterns  — per-layer/head attention weight heatmap
    glassbox_logit_lens          — residual stream layer-by-layer projection
"""

__version__ = "4.2.5"
__author__ = "Ajay Pravin Mahale"
__email__ = "mahale.ajay01@gmail.com"

from glassbox_mcp.server import mcp, main  # noqa: F401

__all__ = ["mcp", "main", "__version__"]
