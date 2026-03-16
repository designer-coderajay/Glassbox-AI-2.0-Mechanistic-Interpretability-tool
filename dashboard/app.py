"""
Glassbox 2.0 — Interactive Mechanistic Interpretability Dashboard
Run: streamlit run dashboard/app.py
"""
import ast
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Glassbox 2.0", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name):
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2
    model = HookedTransformer.from_pretrained(model_name)
    return model, GlassboxV2(model)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Glassbox 2.0")
st.sidebar.markdown("**Mechanistic Interpretability**")
st.sidebar.divider()

model_choice = st.sidebar.selectbox("Model", ["gpt2", "gpt2-medium", "gpt2-large"])
prompt_input = st.sidebar.text_area(
    "Prompt",
    value="When Mary and John went to the store, John gave a bottle to",
    height=100,
)
target_input     = st.sidebar.text_input("Target token",     value="Mary")
distractor_input = st.sidebar.text_input("Distractor token", value="John")

method_choice = st.sidebar.selectbox(
    "Attribution method",
    ["taylor", "integrated_gradients"],
    help=(
        "taylor: fast O(3 passes), first-order approximation.\n"
        "integrated_gradients: slower but more accurate path-integral attribution "
        "(Sundararajan et al. 2017)."
    ),
)
n_steps = 10
if method_choice == "integrated_gradients":
    n_steps = st.sidebar.slider("IG steps (accuracy vs speed)", 5, 20, 10)

run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Glassbox 2.0 — Circuit Analysis")
st.caption(
    "Attribution patching · MLP attribution · Integrated gradients · "
    "Bootstrap 95% CI · FCAS alignment"
)

if not run_btn:
    st.info("Fill in the prompt and tokens on the left, then click **Run Analysis**.")
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────
model, gb = load_model(model_choice)

with st.spinner("Running attribution patching..."):
    try:
        result = gb.analyze(
            prompt_input, target_input, distractor_input,
            method=method_choice, n_steps=n_steps,
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

faith   = result["faithfulness"]
circuit = result["circuit"]

# Parse attributions — analyze() returns string keys e.g. "(9, 9)"
raw_attrs = result["attributions"]
attrs: dict = {}
for key_str, score in raw_attrs.items():
    try:
        layer, head = ast.literal_eval(key_str)
        attrs[(layer, head)] = float(score)
    except Exception:
        pass

# MLP attributions — keyed by layer string
mlp_raw = result.get("mlp_attributions", {})
mlp_attrs = {int(k): float(v) for k, v in mlp_raw.items()}

n_layers = model.cfg.n_layers
n_heads  = model.cfg.n_heads

# ── Faithfulness scores ───────────────────────────────────────────────────────
st.subheader("Faithfulness Scores")

approx_note = (
    "¹ Taylor approximation (fast, first-order)"
    if method_choice == "taylor"
    else "¹ Integrated gradients (path-integral, more accurate)"
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sufficiency¹",       f"{faith['sufficiency']:.1%}")
col2.metric("Comprehensiveness",  f"{faith['comprehensiveness']:.1%}")
col3.metric("F1",                 f"{faith['f1']:.1%}")
col4.metric("Category",           faith["category"].upper())
col5.metric("Clean LD",           f"{result['clean_ld']:.4f}")

st.caption(approx_note + " · Comprehensiveness is always exact (Wang et al. 2022)")
st.divider()

# ── Attention attribution heatmap ─────────────────────────────────────────────
st.subheader("Attention Head Attribution Heatmap")
st.caption(
    "Each cell = attr(layer, head) = ∇z · Δz at last sequence position. "
    "Red = promotes target, Blue = suppresses target."
)

grid = np.zeros((n_layers, n_heads))
for (layer, head), score in attrs.items():
    grid[layer, head] = score

fig = go.Figure(go.Heatmap(
    z=grid,
    x=[f"H{h}" for h in range(n_heads)],
    y=[f"L{l}" for l in range(n_layers)],
    colorscale="RdBu",
    zmid=0,
    hovertemplate="Layer %{y} Head %{x}<br>Score: %{z:.4f}<extra></extra>",
))
fig.update_layout(
    xaxis_title="Head",
    yaxis_title="Layer",
    height=420,
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig, use_container_width=True)
st.divider()

# ── MLP attribution bar chart ─────────────────────────────────────────────────
st.subheader("MLP Layer Attribution")
st.caption(
    "Per-layer MLP contribution via hook_mlp_out. "
    "Extends the circuit picture beyond attention heads. "
    "Positive = MLP layer promotes target at last position."
)

if mlp_attrs:
    layers_sorted = sorted(mlp_attrs.keys())
    mlp_scores    = [mlp_attrs[l] for l in layers_sorted]
    colors        = ["#e74c3c" if v >= 0 else "#3498db" for v in mlp_scores]

    fig_mlp = go.Figure(go.Bar(
        x=[f"L{l}" for l in layers_sorted],
        y=mlp_scores,
        marker_color=colors,
        hovertemplate="Layer %{x}<br>MLP attr: %{y:.4f}<extra></extra>",
    ))
    fig_mlp.update_layout(
        xaxis_title="Layer",
        yaxis_title="Attribution score",
        height=300,
        margin=dict(l=40, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_mlp, use_container_width=True)
st.divider()

# ── Minimum Faithful Circuit ──────────────────────────────────────────────────
st.subheader(f"Minimum Faithful Circuit ({len(circuit)} heads)")
st.caption(
    "Red = heads in the MFC. Circuit found via greedy forward selection "
    "+ exact backward pruning (Wang et al. 2022 / Nanda et al. 2023)."
)

if circuit:
    cgrid = np.zeros((n_layers, n_heads))
    for (layer, head) in circuit:
        cgrid[layer, head] = 1.0

    circuit_scores = [
        f"L{l}H{h}  attr={attrs.get((l, h), 0.0):.4f}"
        for l, h in sorted(circuit, key=lambda x: attrs.get(x, 0.0), reverse=True)
    ]

    fig2 = go.Figure(go.Heatmap(
        z=cgrid,
        x=[f"H{h}" for h in range(n_heads)],
        y=[f"L{l}" for l in range(n_layers)],
        colorscale=[[0, "#f0f2f6"], [1, "#e74c3c"]],
        showscale=False,
        hovertemplate="Layer %{y} Head %{x}<br>In circuit: %{z:.0f}<extra></extra>",
    ))
    fig2.update_layout(
        xaxis_title="Head",
        yaxis_title="Layer",
        height=380,
        margin=dict(l=40, r=20, t=10, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Circuit heads (ranked by attribution):**")
    st.code("\n".join(circuit_scores))
else:
    st.warning("No circuit found — clean logit difference may be zero or all attributions negative.")

st.divider()

# ── Corrupted prompt ──────────────────────────────────────────────────────────
st.subheader("Name-Swap Corruption")
st.caption("Bidirectional name-swap (Wang et al. 2022). Used as the corrupted baseline.")
col_a, col_b = st.columns(2)
col_a.text_input("Clean prompt",     value=prompt_input,          disabled=True)
col_b.text_input("Corrupted prompt", value=result["corr_prompt"], disabled=True)

st.divider()

# ── Top heads table ───────────────────────────────────────────────────────────
st.subheader("Top Attention Heads")
st.caption("Sorted by attribution score. rel_depth = layer / (n_layers − 1).")

top_heads = result.get("top_heads", [])
if top_heads:
    import pandas as pd
    df = pd.DataFrame(top_heads).rename(columns={
        "layer": "Layer", "head": "Head",
        "attr": "Attribution", "rel_depth": "Rel. Depth",
    })
    df["Attribution"] = df["Attribution"].map("{:.4f}".format)
    df["Rel. Depth"]  = df["Rel. Depth"].map("{:.3f}".format)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    "Glassbox 2.0 · "
    "[arXiv:2603.09988](https://arxiv.org/abs/2603.09988) · "
    "[GitHub](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) · "
    "[PyPI](https://pypi.org/project/glassbox-mech-interp/) · "
    "Built by Ajay Pravin Mahale"
)
