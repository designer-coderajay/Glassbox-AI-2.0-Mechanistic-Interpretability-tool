"""
Glassbox 2.0 — Interactive Mechanistic Interpretability Dashboard
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Glassbox 2.0",
    page_icon="🔬",
    layout="wide",
)

# ── Load model (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name):
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2
    model = HookedTransformer.from_pretrained(model_name)
    gb = GlassboxV2(model)
    return model, gb

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🔬 Glassbox 2.0")
st.sidebar.markdown("**Mechanistic Interpretability**")
st.sidebar.divider()

model_choice = st.sidebar.selectbox(
    "Model", ["gpt2", "gpt2-medium", "gpt2-large"],
    help="Model to analyse. gpt2 is fastest."
)

st.sidebar.divider()
st.sidebar.markdown("**IOI Defaults**")
prompt_input = st.sidebar.text_area(
    "Prompt",
    value="When Mary and John went to the store, John gave a bottle to",
    height=100,
)
target_input    = st.sidebar.text_input("Target token",    value="Mary")
distractor_input = st.sidebar.text_input("Distractor token", value="John")
run_btn = st.sidebar.button("▶ Run Analysis", type="primary", use_container_width=True)

# ── Main layout ───────────────────────────────────────────────────────
st.title("🔬 Glassbox 2.0 — Circuit Analysis")
st.caption("Attribution patching · O(3) complexity · Bootstrap 95% CI · FCAS alignment")

if not run_btn:
    st.info("Fill in the prompt and tokens on the left, then click **Run Analysis**.")
    st.markdown("""
    **What Glassbox does:**
    - Runs attribution patching in O(3) forward passes (clean, corrupted, gradient)
    - Identifies the minimum faithful circuit — the smallest set of attention heads that preserves model behaviour
    - Scores faithfulness: sufficiency, comprehensiveness, and F1
    - Computes Bootstrap 95% confidence intervals on attribution scores

    **Example tasks:**
    - IOI: *"When Mary and John went to the store, John gave a bottle to"* → Mary
    - SVA: *"The keys to the cabinet"* → are
    - GEO: *"The capital of France is"* → Paris
    """)
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────
model, gb = load_model(model_choice)

with st.spinner("Running attribution patching..."):
    try:
        clean_tok = model.to_tokens(prompt_input)
        t_tok = model.to_single_token(target_input)
        d_tok = model.to_single_token(distractor_input)
        scores, clean_ld = gb.attribution_patching(clean_tok, clean_tok, t_tok, d_tok)
        result = gb.analyze(prompt_input, target_input, distractor_input)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# ── Faithfulness metrics ──────────────────────────────────────────────
faith = result["faithfulness"]
circuit = result["circuit"]

st.subheader("Faithfulness Scores")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sufficiency",        f"{faith['sufficiency']:.1%}")
col2.metric("Comprehensiveness",  f"{faith['comprehensiveness']:.1%}")
col3.metric("F1",                 f"{faith['f1']:.1%}")
col4.metric("Category",           faith["category"].upper())

st.divider()

# ── Attribution heatmap ───────────────────────────────────────────────
st.subheader("Attribution Scores — All Heads")

n_layers = model.cfg.n_layers
n_heads  = model.cfg.n_heads
grid     = np.zeros((n_layers, n_heads))

for (layer, head), score in scores.items():
    grid[layer, head] = float(score)

fig_heat = go.Figure(go.Heatmap(
    z=grid,
    x=[f"H{h}" for h in range(n_heads)],
    y=[f"L{l}" for l in range(n_layers)],
    colorscale="RdBu",
    zmid=0,
    colorbar=dict(title="Attribution"),
))
fig_heat.update_layout(
    xaxis_title="Head",
    yaxis_title="Layer",
    height=420,
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Circuit heads ─────────────────────────────────────────────────────
st.subheader(f"Minimum Faithful Circuit — {len(circuit)} heads")

if circuit:
    circuit_grid = np.zeros((n_layers, n_heads))
    for (layer, head) in circuit:
        circuit_grid[layer, head] = 1.0

    fig_circ = go.Figure(go.Heatmap(
        z=circuit_grid,
        x=[f"H{h}" for h in range(n_heads)],
        y=[f"L{l}" for l in range(n_layers)],
        colorscale=[[0, "#f0f2f6"], [1, "#e74c3c"]],
        showscale=False,
    ))
    fig_circ.update_layout(
        xaxis_title="Head",
        yaxis_title="Layer",
        height=380,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_circ, use_container_width=True)

    with st.expander("Circuit head list"):
        for layer, head in sorted(circuit):
            attr = scores.get((layer, head), 0.0)
            st.write(f"L{layer}H{head} — attribution score: {attr:.4f}")
else:
    st.warning("No circuit heads found.")

st.divider()

# ── Top-K bar chart ───────────────────────────────────────────────────
st.subheader("Top 15 Heads by Attribution Magnitude")

sorted_heads = sorted(scores.items(), key=lambda x: -abs(x[1]))[:15]
labels = [f"L{l}H{h}" for (l, h), _ in sorted_heads]
values = [v for _, v in sorted_heads]
colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

fig_bar = go.Figure(go.Bar(
    x=labels, y=values,
    marker_color=colors,
))
fig_bar.update_layout(
    xaxis_title="Head",
    yaxis_title="Attribution Score",
    height=320,
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.caption("Red = positive attribution (promotes target). Blue = negative (suppresses target).")
