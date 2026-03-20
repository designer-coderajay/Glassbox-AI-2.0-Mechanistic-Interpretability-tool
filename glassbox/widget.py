"""
glassbox.widget
===============
Jupyter / IPython widget for interactive circuit analysis inside notebooks.

Install the extras:
    pip install glassbox-mech-interp[jupyter]

Usage
-----
    from glassbox import GlassboxV2
    from glassbox.widget import CircuitWidget

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2")
    gb = GlassboxV2(model)

    widget = CircuitWidget(gb)
    widget.show()           # renders inline in a Jupyter cell

    # or analyse a prompt directly:
    widget = CircuitWidget.from_prompt(
        gb,
        prompt="When Mary and John went to the store, John gave a drink to",
        correct=" Mary",
        incorrect=" John",
    )
    widget.show()

Requirements
------------
    ipywidgets >= 8.0
    IPython >= 7.0

These are installed automatically with:
    pip install glassbox-mech-interp[jupyter]
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["CircuitWidget", "HeatmapWidget"]

# ---------------------------------------------------------------------------
# Graceful import — widget is optional, core works without ipywidgets
# ---------------------------------------------------------------------------

try:
    import ipywidgets as widgets                    # type: ignore[import]
    from IPython.display import HTML, display       # type: ignore[import]
    _WIDGETS_AVAILABLE = True
except ImportError:
    _WIDGETS_AVAILABLE = False

if TYPE_CHECKING:
    from glassbox.core import GlassboxV2

_NOT_INSTALLED_MSG = (
    "ipywidgets is not installed. Run:\n"
    "    pip install 'glassbox-mech-interp[jupyter]'\n"
    "or:\n"
    "    pip install ipywidgets\n"
)


# ---------------------------------------------------------------------------
# Inline D3 + CSS template (self-contained, no external deps in notebooks)
# ---------------------------------------------------------------------------

_HEATMAP_TEMPLATE = """
<style>
  .gb-widget {{font-family:'Inter',system-ui,sans-serif;color:#e0e6f0;background:#0d1017;
    border-radius:12px;padding:18px;display:inline-block;}}
  .gb-title {{font-size:13px;font-weight:700;color:#8b92a5;text-transform:uppercase;
    letter-spacing:.7px;margin-bottom:12px;}}
  .gb-grid {{display:flex;flex-direction:column;gap:2px;}}
  .gb-row {{display:flex;align-items:center;gap:2px;}}
  .gb-lbl {{width:28px;font-size:9px;color:#555e72;font-family:monospace;text-align:right;
    padding-right:4px;flex-shrink:0;}}
  .gb-cell {{width:18px;height:18px;border-radius:2px;cursor:pointer;transition:transform .1s;}}
  .gb-cell:hover {{transform:scale(1.3);position:relative;z-index:2;}}
  .gb-circuit-member {{border:1.5px solid rgba(240,180,41,.85);box-shadow:0 0 4px rgba(240,180,41,.25);}}
  .gb-head-row {{display:flex;margin-left:28px;gap:2px;margin-bottom:3px;}}
  .gb-hlbl {{width:18px;font-size:9px;color:#3f4d5e;text-align:center;font-family:monospace;}}
  .gb-legend {{display:flex;align-items:center;gap:8px;margin-top:10px;font-size:10px;color:#555e72;}}
  .gb-grad {{width:60px;height:7px;border-radius:4px;
    background:linear-gradient(to right,#0d1017,#6366f1);flex-shrink:0;}}
  .gb-cdot {{width:10px;height:10px;border-radius:2px;border:1.5px solid rgba(240,180,41,.85);}}
  .gb-chip {{background:rgba(99,102,241,.14);border:1px solid rgba(99,102,241,.22);
    color:#818cf8;font-size:11px;font-weight:600;padding:3px 8px;border-radius:4px;
    font-family:monospace;display:inline-block;margin:2px;}}
  .gb-metric {{background:#131720;border:1px solid rgba(255,255,255,.07);border-radius:8px;
    padding:12px;text-align:center;display:inline-block;margin:4px;min-width:80px;}}
  .gb-mval {{font-size:20px;font-weight:800;font-family:monospace;}}
  .gb-mlbl {{font-size:10px;color:#555e72;text-transform:uppercase;letter-spacing:.5px;margin-top:3px;}}
  .gb-grade {{font-size:28px;font-weight:900;font-family:monospace;padding:8px 16px;
    border-radius:8px;display:inline-block;margin-right:12px;}}
  .gb-A {{background:rgba(34,197,94,.1);color:#4ade80;border:1.5px solid rgba(34,197,94,.3);}}
  .gb-B {{background:rgba(96,165,250,.1);color:#93c5fd;border:1.5px solid rgba(96,165,250,.3);}}
  .gb-C {{background:rgba(245,158,11,.1);color:#fbbf24;border:1.5px solid rgba(245,158,11,.3);}}
  .gb-D {{background:rgba(239,68,68,.1);color:#fca5a5;border:1.5px solid rgba(239,68,68,.3);}}
</style>
<div class="gb-widget">
  <div class="gb-title">Glassbox Attribution Heatmap — {model_name}</div>
  <div style="margin-bottom:14px;">
    <span class="gb-grade gb-{grade}">{grade}</span>
    <span style="font-size:15px;font-weight:700;">{grade_label}</span>
    <div style="font-size:11px;color:#555e72;margin-top:6px;">Report: GB-{report_id}</div>
  </div>
  <div style="margin-bottom:14px;">
    {metrics_html}
  </div>
  <div style="margin-bottom:10px;font-size:11px;font-weight:700;text-transform:uppercase;
    letter-spacing:.7px;color:#555e72;">Circuit Components</div>
  <div style="margin-bottom:14px;">{chips_html}</div>
  <div class="gb-title" style="margin-bottom:8px;">Attribution Heatmap</div>
  <div class="gb-head-row">{head_labels_html}</div>
  <div class="gb-grid">{grid_html}</div>
  <div class="gb-legend">
    <div class="gb-grad"></div>
    <span>Low → High attribution</span>
    &nbsp;
    <div class="gb-cdot"></div>
    <span>Circuit member</span>
  </div>
</div>
"""

_CELL_TEMPLATE = '<div class="gb-cell{circuit_cls}" style="background:rgb({r},{g},{b})" title="{key}: {score:.4f} ({pct:.1f}%)"></div>'


def _rgb_for_score(norm: float) -> Tuple[int, int, int]:
    """Interpolate from surface (#0d1017) to accent (#6366f1) by norm ∈ [0,1]."""
    r = int(13 + norm * (99 - 13))
    g = int(16 + norm * (102 - 16))
    b = int(23 + norm * (241 - 23))
    return r, g, b


def _build_heatmap_html(result: Dict[str, Any]) -> str:
    faith = result.get("faithfulness") or {}
    suff  = faith.get("sufficiency", 0)
    comp  = faith.get("comprehensiveness", 0)
    f1    = faith.get("f1", 0)

    grade_raw = result.get("explainability_grade") or "D — Unknown"
    grade = grade_raw[0] if grade_raw else "D"
    report_id = result.get("report_id") or "?"

    full  = result.get("full_report") or {}
    s2    = (full.get("sections") or {}).get("2_development_design") or {}
    model_name = result.get("model_name") or "model"

    circuit_heads: List[str] = s2.get("circuit_heads") or result.get("circuit") or []
    circuit_set = set(circuit_heads)
    attr_scores: Dict[str, float] = s2.get("attribution_scores") or {}
    n_layers = int(s2.get("n_layers") or 12)
    n_heads  = int(s2.get("n_heads")  or 12)

    def get_score(l: int, h: int) -> float:
        key = f"L{l}H{h}"
        if key in attr_scores:
            return attr_scores[key]
        if key in circuit_set:
            return 0.10
        return 0.0

    max_score = max((get_score(l, h) for l in range(n_layers) for h in range(n_heads)), default=1.0) or 1.0

    f1_color = "#4ade80" if f1 >= 0.70 else "#fbbf24" if f1 >= 0.50 else "#fca5a5"

    metrics_html = "".join([
        f'<div class="gb-metric"><div class="gb-mval" style="color:#4ade80">{suff:.3f}</div><div class="gb-mlbl">Sufficiency</div></div>',
        f'<div class="gb-metric"><div class="gb-mval" style="color:#93c5fd">{comp:.3f}</div><div class="gb-mlbl">Comprehensiveness</div></div>',
        f'<div class="gb-metric"><div class="gb-mval" style="color:{f1_color}">{f1:.3f}</div><div class="gb-mlbl">F1 Score</div></div>',
    ])

    chips_html = "".join(f'<span class="gb-chip">{h}</span>' for h in circuit_heads) or "<span style='color:#555e72;font-size:12px'>No circuit identified</span>"

    head_labels_html = "".join(f'<div class="gb-hlbl">{h}</div>' for h in range(n_heads))

    rows = []
    for l in range(n_layers - 1, -1, -1):
        cells = [f'<div class="gb-lbl">L{l}</div>']
        for h in range(n_heads):
            score = get_score(l, h)
            norm  = score / max_score
            r, g, b = _rgb_for_score(norm)
            key = f"L{l}H{h}"
            circuit_cls = " gb-circuit-member" if key in circuit_set else ""
            pct = norm * 100
            cells.append(_CELL_TEMPLATE.format(circuit_cls=circuit_cls, r=r, g=g, b=b, key=key, score=score, pct=pct))
        rows.append(f'<div class="gb-row">{"".join(cells)}</div>')

    grid_html = "\n".join(rows)

    return _HEATMAP_TEMPLATE.format(
        model_name=model_name,
        grade=grade,
        grade_label=grade_raw,
        report_id=report_id,
        metrics_html=metrics_html,
        chips_html=chips_html,
        head_labels_html=head_labels_html,
        grid_html=grid_html,
    )


# ---------------------------------------------------------------------------
# CircuitWidget
# ---------------------------------------------------------------------------

class CircuitWidget:
    """
    Interactive Jupyter widget showing the attribution heatmap and circuit
    analysis results for a Glassbox audit.

    Parameters
    ----------
    gb     : GlassboxV2 instance (already loaded with a model).
    result : Pre-computed result dict from ``gb.analyze()``.
             If None, call ``analyze_prompt()`` before ``show()``.

    Examples
    --------
    >>> from glassbox import GlassboxV2
    >>> from glassbox.widget import CircuitWidget
    >>> import transformer_lens
    >>> model = transformer_lens.HookedTransformer.from_pretrained("gpt2")
    >>> gb    = GlassboxV2(model)
    >>> w     = CircuitWidget.from_prompt(gb, "When Mary and John ...", " Mary", " John")
    >>> w.show()
    """

    def __init__(
        self,
        gb: "GlassboxV2",
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.gb     = gb
        self.result = result

    @classmethod
    def from_prompt(
        cls,
        gb: "GlassboxV2",
        prompt: str,
        correct: str,
        incorrect: str,
        method: str = "taylor",
    ) -> "CircuitWidget":
        """
        Convenience constructor: run ``gb.analyze()`` and wrap the result.

        Parameters
        ----------
        gb        : GlassboxV2 instance.
        prompt    : Decision prompt.
        correct   : Expected correct token.
        incorrect : Distractor token.
        method    : Attribution method ('taylor' or 'integrated_gradients').
        """
        logger.info("Running Glassbox analysis — this may take 5–30s...")
        result = gb.analyze(prompt, correct, incorrect, method=method)
        return cls(gb=gb, result=result)

    def analyze_prompt(
        self,
        prompt: str,
        correct: str,
        incorrect: str,
        method: str = "taylor",
    ) -> "CircuitWidget":
        """Run analysis and update the widget result in-place. Returns self."""
        self.result = self.gb.analyze(prompt, correct, incorrect, method=method)
        return self

    def _repr_html_(self) -> str:
        """Jupyter calls this to render the widget inline."""
        if self.result is None:
            return "<p style='color:#fca5a5'>No result. Call <code>analyze_prompt()</code> first.</p>"
        return _build_heatmap_html(self.result)

    def show(self) -> None:
        """
        Display the widget in the current Jupyter cell.
        Falls back to printing a summary if IPython is not available.
        """
        if not _WIDGETS_AVAILABLE:
            logger.warning(_NOT_INSTALLED_MSG)
            if self.result:
                faith = self.result.get("faithfulness") or {}
                print(
                    f"Glassbox result — grade: {self.result.get('explainability_grade','?')} "
                    f"| F1: {faith.get('f1',0):.3f} "
                    f"| circuit: {self.result.get('circuit',[])}"
                )
            return

        if self.result is None:
            display(HTML("<p style='color:#fca5a5'>No result. Call <code>analyze_prompt()</code> first.</p>"))
            return

        display(HTML(_build_heatmap_html(self.result)))

    def to_html(self) -> str:
        """Return the full HTML string (useful for exporting to a file)."""
        return self._repr_html_()

    def summary(self) -> Dict[str, Any]:
        """Return a dict summary of the analysis result."""
        if not self.result:
            return {}
        faith = self.result.get("faithfulness") or {}
        return {
            "grade":           (self.result.get("explainability_grade") or "D")[0],
            "f1":              faith.get("f1", 0),
            "sufficiency":     faith.get("sufficiency", 0),
            "comprehensiveness": faith.get("comprehensiveness", 0),
            "circuit":         self.result.get("circuit", []),
            "report_id":       self.result.get("report_id", ""),
        }

    def __repr__(self) -> str:
        if self.result:
            faith = self.result.get("faithfulness") or {}
            return (
                f"CircuitWidget(model={self.result.get('model_name','?')!r}, "
                f"grade={(self.result.get('explainability_grade','D')[0])!r}, "
                f"f1={faith.get('f1',0):.3f})"
            )
        return "CircuitWidget(no result)"


# ---------------------------------------------------------------------------
# HeatmapWidget — standalone, takes a result dict directly
# ---------------------------------------------------------------------------

class HeatmapWidget:
    """
    Minimal widget that takes a pre-computed result dict and renders the
    attribution heatmap. Use this if you already have a result from the
    REST API or from ``gb.analyze()`` and just want to visualise it.

    Parameters
    ----------
    result : Dict from ``GlassboxV2.analyze()`` or the REST API JSON response.

    Example
    -------
    >>> import requests
    >>> resp = requests.post("https://glassbox-ai-2-0-mechanistic.onrender.com/v1/audit/analyze", json={...})
    >>> from glassbox.widget import HeatmapWidget
    >>> HeatmapWidget(resp.json()).show()
    """

    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result

    def _repr_html_(self) -> str:
        return _build_heatmap_html(self.result)

    def show(self) -> None:
        if not _WIDGETS_AVAILABLE:
            print("Install ipywidgets: pip install 'glassbox-mech-interp[jupyter]'")
            return
        display(HTML(_build_heatmap_html(self.result)))

    def to_html(self) -> str:
        return self._repr_html_()
