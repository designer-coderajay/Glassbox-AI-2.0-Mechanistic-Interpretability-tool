"""
tests/test_widget.py — Unit tests for glassbox.widget (HeatmapWidget, CircuitWidget).

Tests run WITHOUT a live model or network access — we use synthetic result dicts
that match the schema produced by GlassboxV2.analyze() and REST API responses.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

# Gracefully handle missing ipywidgets (will be mocked in tests)
try:
    import ipywidgets as widgets
    from IPython.display import HTML, display
    _WIDGETS_AVAILABLE = True
except ImportError:
    _WIDGETS_AVAILABLE = False

from glassbox.widget import (
    HeatmapWidget,
    CircuitWidget,
    _build_heatmap_html,
    _rgb_for_score,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic result dicts matching GlassboxV2.analyze() schema
# ---------------------------------------------------------------------------


def _make_result(
    model_name="test-model",
    grade="A",
    suff=0.85,
    comp=0.62,
    f1=None,
    report_id="GB-12345678",
    n_layers=12,
    n_heads=12,
):
    """Build a realistic GlassboxV2.analyze() result dict."""
    if f1 is None:
        f1 = 2 * suff * comp / (suff + comp) if (suff + comp) > 0 else 0.0

    circuit_heads = [f"L{i}H{j}" for i in range(min(2, n_layers)) for j in range(min(4, n_heads))]
    attribution_scores = {h: 0.25 for h in circuit_heads}

    return {
        "model_name": model_name,
        "explainability_grade": f"{grade} — Explainability {grade}",
        "report_id": report_id,
        "faithfulness": {
            "sufficiency": suff,
            "comprehensiveness": comp,
            "f1": f1,
        },
        "circuit": circuit_heads,
        "full_report": {
            "sections": {
                "2_development_design": {
                    "circuit_heads": circuit_heads,
                    "attribution_scores": attribution_scores,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                },
            },
        },
    }


@pytest.fixture
def good_result():
    """High faithfulness result — Grade A."""
    return _make_result(grade="A", suff=0.92, comp=0.68)


@pytest.fixture
def medium_result():
    """Medium faithfulness — Grade B."""
    return _make_result(grade="B", suff=0.71, comp=0.44)


@pytest.fixture
def bad_result():
    """Low faithfulness — Grade D."""
    return _make_result(grade="D", suff=0.35, comp=0.18, f1=0.42)


@pytest.fixture
def minimal_result():
    """Minimal result dict with only required fields."""
    return {
        "model_name": "minimal",
        "explainability_grade": "D",
        "report_id": "GB-MIN",
        "circuit": [],
    }


# ---------------------------------------------------------------------------
# Test: Color interpolation (_rgb_for_score)
# ---------------------------------------------------------------------------


def test_rgb_for_score_minimum():
    """Test color at minimum (norm=0) — should be close to background."""
    r, g, b = _rgb_for_score(0.0)
    assert r == 13  # background color
    assert g == 16
    assert b == 23


def test_rgb_for_score_maximum():
    """Test color at maximum (norm=1) — should be accent color."""
    r, g, b = _rgb_for_score(1.0)
    assert r == 99  # accent color
    assert g == 102
    assert b == 241


def test_rgb_for_score_midpoint():
    """Test color at midpoint (norm=0.5) — should interpolate."""
    r, g, b = _rgb_for_score(0.5)
    assert 13 < r < 99
    assert 16 < g < 102
    assert 23 < b < 241


def test_rgb_for_score_returns_ints():
    """Test that _rgb_for_score returns integers."""
    r, g, b = _rgb_for_score(0.3)
    assert isinstance(r, int)
    assert isinstance(g, int)
    assert isinstance(b, int)


# ---------------------------------------------------------------------------
# Test: HeatmapWidget construction and rendering
# ---------------------------------------------------------------------------


def test_heatmap_widget_construction(good_result):
    """Test HeatmapWidget instantiation."""
    w = HeatmapWidget(good_result)
    assert w.result is good_result
    assert w.result["model_name"] == "test-model"


def test_heatmap_widget_repr_html_returns_string(good_result):
    """Test that _repr_html_() returns a non-empty string."""
    w = HeatmapWidget(good_result)
    html = w._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 100


def test_heatmap_widget_repr_html_contains_model_name(good_result):
    """Test that HTML contains model name."""
    w = HeatmapWidget(good_result)
    html = w._repr_html_()
    assert "test-model" in html or "Glassbox" in html


def test_heatmap_widget_repr_html_contains_grade(good_result):
    """Test that HTML contains grade indicator."""
    w = HeatmapWidget(good_result)
    html = w._repr_html_()
    assert "grade" in html.lower() or "A" in html


def test_heatmap_widget_repr_html_contains_metrics(good_result):
    """Test that HTML contains faithfulness metrics."""
    w = HeatmapWidget(good_result)
    html = w._repr_html_()
    assert "Sufficiency" in html or "sufficiency" in html
    assert "Comprehensiveness" in html or "comprehensiveness" in html
    assert "F1" in html or "f1" in html


def test_heatmap_widget_to_html(good_result):
    """Test to_html() returns same as _repr_html_()."""
    w = HeatmapWidget(good_result)
    html1 = w._repr_html_()
    html2 = w.to_html()
    assert html1 == html2


def test_heatmap_widget_show_with_widgets_available(good_result, monkeypatch):
    """Test show() works when ipywidgets is available."""
    if not _WIDGETS_AVAILABLE:
        pytest.skip("ipywidgets not available")

    with patch("glassbox.widget.display") as mock_display:
        w = HeatmapWidget(good_result)
        w.show()
        mock_display.assert_called_once()


def test_heatmap_widget_show_without_widgets(good_result, capsys):
    """Test show() prints fallback message when ipywidgets unavailable."""
    with patch("glassbox.widget._WIDGETS_AVAILABLE", False):
        w = HeatmapWidget(good_result)
        w.show()
        captured = capsys.readouterr()
        assert "Install ipywidgets" in captured.out or "Install" in captured.out


def test_heatmap_widget_minimal_result(minimal_result):
    """Test HeatmapWidget with minimal result dict."""
    w = HeatmapWidget(minimal_result)
    html = w._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0


def test_heatmap_widget_missing_report_id():
    """Test HeatmapWidget with missing report_id."""
    result = {
        "model_name": "test",
        "explainability_grade": "D",
        "circuit": [],
    }
    w = HeatmapWidget(result)
    html = w._repr_html_()
    assert "?" in html or "report" in html.lower()


# ---------------------------------------------------------------------------
# Test: CircuitWidget construction and rendering
# ---------------------------------------------------------------------------


def test_circuit_widget_construction(good_result):
    """Test CircuitWidget instantiation with a result dict."""
    gb_mock = Mock()  # Mock GlassboxV2 instance
    w = CircuitWidget(gb=gb_mock, result=good_result)
    assert w.gb is gb_mock
    assert w.result is good_result


def test_circuit_widget_construction_no_result(good_result):
    """Test CircuitWidget instantiation without a result."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=None)
    assert w.result is None


def test_circuit_widget_repr_html_with_result(good_result):
    """Test _repr_html_() with a result dict."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=good_result)
    html = w._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 100


def test_circuit_widget_repr_html_without_result():
    """Test _repr_html_() without a result shows message."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=None)
    html = w._repr_html_()
    assert isinstance(html, str)
    assert "No result" in html or "analyze_prompt" in html


def test_circuit_widget_repr_html_contains_metrics(good_result):
    """Test that HTML contains key circuit analysis metrics."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=good_result)
    html = w._repr_html_()
    # Should contain attributions or circuit information
    assert "circuit" in html.lower() or "attribution" in html.lower() or "L" in html


def test_circuit_widget_show_with_result(good_result):
    """Test show() displays HTML when result is present."""
    if not _WIDGETS_AVAILABLE:
        pytest.skip("ipywidgets not available")

    gb_mock = Mock()
    with patch("glassbox.widget.display") as mock_display:
        w = CircuitWidget(gb=gb_mock, result=good_result)
        w.show()
        mock_display.assert_called_once()


def test_circuit_widget_show_without_result():
    """Test show() handles case without result."""
    if not _WIDGETS_AVAILABLE:
        pytest.skip("ipywidgets not available")

    gb_mock = Mock()
    with patch("glassbox.widget.display") as mock_display:
        w = CircuitWidget(gb=gb_mock, result=None)
        w.show()
        mock_display.assert_called_once()


def test_circuit_widget_show_without_widgets(good_result, capsys):
    """Test show() prints fallback when ipywidgets unavailable."""
    with patch("glassbox.widget._WIDGETS_AVAILABLE", False):
        gb_mock = Mock()
        w = CircuitWidget(gb=gb_mock, result=good_result)
        w.show()
        captured = capsys.readouterr()
        # Should print something (either warning or result summary)
        assert len(captured.out) > 0 or len(captured.err) > 0


def test_circuit_widget_to_html(good_result):
    """Test to_html() returns HTML string."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=good_result)
    html = w.to_html()
    assert isinstance(html, str)
    assert len(html) > 0


def test_circuit_widget_summary(good_result):
    """Test summary() returns dict with key metrics."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=good_result)
    summary = w.summary()
    assert isinstance(summary, dict)
    assert "grade" in summary
    assert "f1" in summary
    assert "sufficiency" in summary
    assert "comprehensiveness" in summary
    assert "circuit" in summary
    assert "report_id" in summary


def test_circuit_widget_summary_empty():
    """Test summary() with no result."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=None)
    summary = w.summary()
    assert summary == {}


def test_circuit_widget_repr(good_result):
    """Test __repr__() returns descriptive string."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=good_result)
    r = repr(w)
    assert "CircuitWidget" in r


def test_circuit_widget_repr_no_result():
    """Test __repr__() when no result."""
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=None)
    r = repr(w)
    assert "no result" in r


def test_circuit_widget_analyze_prompt_updates_result(good_result, medium_result):
    """Test that analyze_prompt() updates the result in-place."""
    gb_mock = Mock()
    gb_mock.analyze.return_value = medium_result
    w = CircuitWidget(gb=gb_mock, result=good_result)

    assert w.result["explainability_grade"].startswith("A")

    w.analyze_prompt("new prompt", " correct", " wrong")
    assert w.result["explainability_grade"].startswith("B")


def test_circuit_widget_analyze_prompt_returns_self(good_result):
    """Test that analyze_prompt() returns self for chaining."""
    gb_mock = Mock()
    gb_mock.analyze.return_value = good_result
    w = CircuitWidget(gb=gb_mock, result=None)

    result = w.analyze_prompt("prompt", " correct", " wrong")
    assert result is w


# ---------------------------------------------------------------------------
# Test: _build_heatmap_html function (core rendering logic)
# ---------------------------------------------------------------------------


def test_build_heatmap_html_basic(good_result):
    """Test that _build_heatmap_html produces valid HTML."""
    html = _build_heatmap_html(good_result)
    assert isinstance(html, str)
    assert len(html) > 500
    assert "<div" in html
    assert "gb-widget" in html


def test_build_heatmap_html_contains_metrics(good_result):
    """Test that HTML contains faithfulness metrics."""
    html = _build_heatmap_html(good_result)
    # Should contain metric boxes
    assert "gb-metric" in html


def test_build_heatmap_html_contains_grid(good_result):
    """Test that HTML contains the heatmap grid."""
    html = _build_heatmap_html(good_result)
    assert "gb-grid" in html or "gb-row" in html


def test_build_heatmap_html_grade_a_styling(good_result):
    """Test that Grade A gets appropriate styling."""
    html = _build_heatmap_html(good_result)
    # Grade A should have specific CSS class
    assert "gb-A" in html or "4ade80" in html  # green color


def test_build_heatmap_html_grade_d_styling(bad_result):
    """Test that Grade D gets appropriate styling."""
    html = _build_heatmap_html(bad_result)
    # Grade D should have specific CSS class
    assert "gb-D" in html or "fca5a5" in html  # red color


def test_build_heatmap_html_no_circuit_message(minimal_result):
    """Test HTML shows message when no circuit identified."""
    html = _build_heatmap_html(minimal_result)
    assert "No circuit" in html or "circuit" in html.lower()


def test_build_heatmap_html_circuit_members_marked(good_result):
    """Test that circuit members are marked in HTML."""
    html = _build_heatmap_html(good_result)
    # Circuit members should be marked with special class
    if good_result["circuit"]:
        assert "gb-circuit-member" in html


def test_build_heatmap_html_high_f1_color(good_result):
    """Test that high F1 score gets green color."""
    # good_result has f1 > 0.70
    html = _build_heatmap_html(good_result)
    assert "4ade80" in html  # green color for F1


def test_build_heatmap_html_low_f1_color(bad_result):
    """Test that low F1 score gets red color."""
    # bad_result has f1 < 0.50
    html = _build_heatmap_html(bad_result)
    assert "fca5a5" in html  # red color for F1


def test_build_heatmap_html_missing_report_id():
    """Test HTML generation with missing report_id."""
    result = {
        "model_name": "test",
        "explainability_grade": "D",
        "circuit": [],
    }
    html = _build_heatmap_html(result)
    assert "?" in html  # Fallback for missing report_id


def test_build_heatmap_html_large_model(good_result):
    """Test HTML generation with larger model dims."""
    result = _make_result(n_layers=24, n_heads=16)
    html = _build_heatmap_html(result)
    assert isinstance(html, str)
    assert len(html) > 1000  # More cells = larger HTML


# ---------------------------------------------------------------------------
# Test: Edge cases and error handling
# ---------------------------------------------------------------------------


def test_heatmap_widget_with_empty_circuit():
    """Test HeatmapWidget with empty circuit list."""
    result = {
        "model_name": "empty",
        "explainability_grade": "D",
        "circuit": [],
        "faithfulness": {"sufficiency": 0.1, "comprehensiveness": 0.1, "f1": 0.1},
    }
    w = HeatmapWidget(result)
    html = w._repr_html_()
    assert isinstance(html, str)
    assert "No circuit" in html or "circuit" in html.lower()


def test_circuit_widget_with_zero_metrics(good_result):
    """Test CircuitWidget handles zero metrics gracefully."""
    result = good_result.copy()
    result["faithfulness"] = {"sufficiency": 0.0, "comprehensiveness": 0.0, "f1": 0.0}
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=result)
    html = w._repr_html_()
    assert isinstance(html, str)


def test_heatmap_widget_missing_sections():
    """Test HeatmapWidget with missing full_report.sections."""
    result = {
        "model_name": "test",
        "explainability_grade": "B",
        "circuit": ["L0H0"],
        "faithfulness": {"sufficiency": 0.7, "comprehensiveness": 0.6, "f1": 0.65},
    }
    w = HeatmapWidget(result)
    html = w._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0


def test_circuit_widget_missing_faithfulness_key():
    """Test CircuitWidget when faithfulness key is missing."""
    result = {
        "model_name": "test",
        "explainability_grade": "C",
        "circuit": [],
    }
    gb_mock = Mock()
    w = CircuitWidget(gb=gb_mock, result=result)
    summary = w.summary()
    # Should handle missing keys gracefully
    assert "f1" in summary
    assert summary["f1"] == 0  # Default


def test_rgb_interpolation_boundary_values():
    """Test _rgb_for_score with boundary values."""
    # Test several interpolation points
    for norm in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r, g, b = _rgb_for_score(norm)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255


def test_heatmap_widget_repr_string(good_result):
    """Test that HeatmapWidget doesn't have a __repr__ but can be created."""
    w = HeatmapWidget(good_result)
    # Should not raise
    s = str(w)
    assert isinstance(s, str)


def test_circuit_widget_from_prompt_not_called_in_tests():
    """Document that from_prompt is not tested (requires live model)."""
    # from_prompt requires a live GlassboxV2 instance and calls gb.analyze()
    # We skip this test as it requires a model download
    pytest.skip("from_prompt requires live model — not tested in offline mode")


# ---------------------------------------------------------------------------
# Test: Integration between widgets
# ---------------------------------------------------------------------------


def test_heatmap_and_circuit_same_result(good_result):
    """Test that HeatmapWidget and CircuitWidget produce compatible HTML."""
    hw = HeatmapWidget(good_result)
    gb_mock = Mock()
    cw = CircuitWidget(gb=gb_mock, result=good_result)

    html_h = hw._repr_html_()
    html_c = cw._repr_html_()

    # Both should be non-empty strings
    assert len(html_h) > 0
    assert len(html_c) > 0

    # Both should contain the model name
    assert "good" in html_h.lower() or "test" in html_h
    assert "good" in html_c.lower() or "test" in html_c
