"""Tests for glassbox/scaling.py — batch planning + hierarchical screening."""
import pytest

from glassbox.scaling import hierarchical_screen, plan_batches


# ── plan_batches ────────────────────────────────────────────────────────────
def test_empty_and_invalid():
    assert plan_batches([], 4) == []
    with pytest.raises(ValueError):
        plan_batches(["a"], 0)


def test_respects_max_batch_size_and_conserves_items():
    items = ["aa", "b", "cccc", "dd", "e"]
    batches = plan_batches(items, max_batch_size=2)
    assert all(len(b) <= 2 for b in batches)
    assert sum(len(b) for b in batches) == len(items)
    flat = [x for b in batches for x in b]
    assert sorted(flat) == sorted(items)


def test_length_sorted_grouping():
    items = ["aa", "b", "cccc", "dd", "e"]
    batches = plan_batches(items, max_batch_size=2)
    # first batch should hold the two shortest (len 1)
    assert all(len(x) == 1 for x in batches[0])


def test_padded_token_budget():
    items = ["aa", "b", "cccc", "dd", "e"]  # lengths 2,1,4,2,1
    batches = plan_batches(items, max_batch_size=10, max_padded_tokens=4)
    for b in batches:
        padded = max(len(x) for x in b) * len(b)
        assert padded <= 4
    assert sum(len(b) for b in batches) == 5


def test_single_oversized_item_kept_alone():
    batches = plan_batches(["xxxxxx"], max_batch_size=10, max_padded_tokens=4)
    assert batches == [["xxxxxx"]]


# ── hierarchical_screen ─────────────────────────────────────────────────────
def test_empty_layer_scores_prunes_all():
    screened, pruned, report = hierarchical_screen({}, {"h": 0})
    assert screened == []
    assert pruned == ["h"]
    assert report["false_negative_risk"] is False


def test_keeps_top_layers_low_risk():
    layer_scores = {0: 0.1, 1: 0.9, 2: 0.05, 3: 0.02}
    head_to_layer = {"h0": 0, "h1a": 1, "h1b": 1, "h2": 2, "h3": 3}
    screened, pruned, report = hierarchical_screen(
        layer_scores, head_to_layer, layer_keep_frac=0.5
    )
    assert set(screened) == {"h0", "h1a", "h1b"}   # layers 0 and 1 kept
    assert set(pruned) == {"h2", "h3"}
    assert report["n_layers_kept"] == 2
    assert report["false_negative_risk"] is False  # pruned mass ~6.5%


def test_flags_false_negative_risk():
    # half the mass sits in the pruned layers -> risky screen
    layer_scores = {0: 0.5, 1: 0.5, 2: 0.4, 3: 0.4}
    head_to_layer = {f"h{i}": i for i in range(4)}
    _, _, report = hierarchical_screen(layer_scores, head_to_layer, layer_keep_frac=0.5)
    assert report["pruned_layer_mass_fraction"] > 0.10
    assert report["false_negative_risk"] is True


def test_min_layers_respected():
    layer_scores = {0: 0.9, 1: 0.1, 2: 0.05}
    head_to_layer = {"a": 0, "b": 1, "c": 2}
    screened, _, report = hierarchical_screen(
        layer_scores, head_to_layer, layer_keep_frac=0.0, min_layers=1
    )
    assert report["n_layers_kept"] == 1
    assert screened == ["a"]  # only the top layer's head
