"""
Glassbox 2.0 — Interactive Mechanistic Interpretability Dashboard
Run locally:  streamlit run dashboard/app.py
Hosted:       https://huggingface.co/spaces/designer-coderajay/Glassbox-ai
"""
import ast
import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Glassbox 2.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Glassbox 2.0")
st.sidebar.markdown(
    "**Mechanistic Interpretability**  \n"
    "Circuits · SAE Features · Composition · Token Attribution"
)
st.sidebar.divider()

model_choice = st.sidebar.selectbox(
    "Model",
    ["gpt2", "gpt2-medium", "gpt2-large"],
    help="GPT-2 small is fastest. All models run on CPU.",
)
prompt_input = st.sidebar.text_area(
    "Prompt",
    value="When Mary and John went to the store, John gave a bottle to",
    height=100,
)
target_input     = st.sidebar.text_input("Target token",     value=" Mary")
distractor_input = st.sidebar.text_input("Distractor token", value=" John")

method_choice = st.sidebar.selectbox(
    "Attribution method",
    ["taylor", "integrated_gradients"],
    help=(
        "**taylor** — fast O(3 passes), first-order approximation (Nanda et al. 2023).  \n"
        "**integrated_gradients** — path-integral attribution, more accurate but slower "
        "(Sundararajan et al. 2017). Costs 2+n_steps passes."
    ),
)
n_steps = 10
if method_choice == "integrated_gradients":
    n_steps = st.sidebar.slider("IG steps", 5, 20, 10)

st.sidebar.divider()
include_logit_lens = st.sidebar.checkbox("Include Logit Lens", value=True)

run_btn = st.sidebar.button("▶  Run Analysis", type="primary", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Glassbox 2.0 — Circuit Analysis")
st.caption(
    "Attribution patching · Edge AP · Logit lens · Token attribution · "
    "Composition scores · Bootstrap CI · FCAS  ·  "
    "[arXiv:2603.09988](https://arxiv.org/abs/2603.09988)"
)

if not run_btn:
    st.info(
        "Fill in the prompt and tokens on the left, then click **▶ Run Analysis**.  \n\n"
        "**Example** (IOI — Indirect Object Identification):  \n"
        "- Prompt: `When Mary and John went to the store, John gave a bottle to`  \n"
        "- Target: ` Mary`  \n"
        "- Distractor: ` John`"
    )
    st.stop()

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(model_name: str):
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2
    model = HookedTransformer.from_pretrained(model_name)
    return model, GlassboxV2(model)

model, gb = load_model(model_choice)
n_layers  = model.cfg.n_layers
n_heads   = model.cfg.n_heads

# ── Run analysis ───────────────────────────────────────────────────────────────
with st.spinner("Running circuit analysis…"):
    try:
        result = gb.analyze(
            prompt_input.strip(), target_input.strip(), distractor_input.strip(),
            method=method_choice, n_steps=n_steps,
            include_logit_lens=include_logit_lens,
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# ── Parse outputs ─────────────────────────────────────────────────────────────
faith   = result["faithfulness"]
circuit = result["circuit"]

attrs: dict = {}
for key_str, score in result["attributions"].items():
    try:
        layer, head = ast.literal_eval(key_str)
        attrs[(layer, head)] = float(score)
    except Exception:
        pass

mlp_attrs = {int(k): float(v) for k, v in result.get("mlp_attributions", {}).items()}

# ── Token attribution (runs quickly) ──────────────────────────────────────────
tokens_c = model.to_tokens(prompt_input.strip())
try:
    t_tok = model.to_single_token(target_input.strip())
    d_tok = model.to_single_token(distractor_input.strip())
except Exception:
    t_tok = model.to_tokens(target_input.strip())[0, -1].item()
    d_tok = model.to_tokens(distractor_input.strip())[0, -1].item()

with st.spinner("Computing token attribution…"):
    try:
        tok_attr = gb.token_attribution(tokens_c, t_tok, d_tok)
    except Exception:
        tok_attr = None

# ── TAB LAYOUT ────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Circuit",
    "🎯 Logit Lens",
    "🔗 Edge AP",
    "🔤 Token Attribution",
    "🧩 Composition",
    "📈 Faithfulness",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Circuit (Attribution Heatmap + MFC + MLP)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    # Faithfulness KPIs
    approx_note = (
        "¹ Taylor approximation (first-order, fast)"
        if method_choice == "taylor"
        else "¹ Integrated gradients (path-integral, accurate)"
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sufficiency¹",       f"{faith['sufficiency']:.1%}")
    c2.metric("Comprehensiveness",  f"{faith['comprehensiveness']:.1%}")
    c3.metric("F1",                 f"{faith['f1']:.1%}")
    c4.metric("Category",           faith["category"].upper())
    c5.metric("Clean LD",           f"{result['clean_ld']:.4f}")
    st.caption(approx_note + " · Comprehensiveness is exact (Wang et al. 2022)")
    st.divider()

    # Attribution heatmap
    st.subheader("Attention Head Attribution Heatmap")
    st.caption(
        "attr(l,h) = ∇_z LD · Δz at last position. "
        "Red = promotes target. Blue = suppresses."
    )
    grid = np.zeros((n_layers, n_heads))
    for (l, h), score in attrs.items():
        grid[l, h] = score

    fig_heat = go.Figure(go.Heatmap(
        z=grid,
        x=[f"H{h}" for h in range(n_heads)],
        y=[f"L{l}" for l in range(n_layers)],
        colorscale="RdBu", zmid=0,
        hovertemplate="L%{y} H%{x}<br>Score: %{z:.4f}<extra></extra>",
    ))
    fig_heat.update_layout(
        xaxis_title="Head", yaxis_title="Layer",
        height=400, margin=dict(l=40, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.divider()

    # MFC grid
    st.subheader(f"Minimum Faithful Circuit  ({len(circuit)} heads)")
    if circuit:
        cgrid = np.zeros((n_layers, n_heads))
        for l, h in circuit:
            cgrid[l, h] = 1.0
        fig_mfc = go.Figure(go.Heatmap(
            z=cgrid,
            x=[f"H{h}" for h in range(n_heads)],
            y=[f"L{l}" for l in range(n_layers)],
            colorscale=[[0, "#f0f2f6"], [1, "#e74c3c"]],
            showscale=False,
            hovertemplate="L%{y} H%{x}  in circuit: %{z:.0f}<extra></extra>",
        ))
        fig_mfc.update_layout(
            xaxis_title="Head", yaxis_title="Layer",
            height=380, margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_mfc, use_container_width=True)
        circuit_rows = [
            {"Head": f"L{l:02d}H{h:02d}", "Attribution": f"{attrs.get((l,h), 0.0):.4f}",
             "Rel. Depth": f"{l / max(n_layers - 1, 1):.3f}"}
            for l, h in sorted(circuit, key=lambda x: attrs.get(x, 0.0), reverse=True)
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(circuit_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No circuit found — check that LD > 0.")
    st.divider()

    # MLP attribution
    st.subheader("MLP Layer Attribution")
    st.caption(
        "Per-layer MLP contribution via hook_mlp_out. "
        "Extends circuit picture beyond attention heads."
    )
    if mlp_attrs:
        ls = sorted(mlp_attrs.keys())
        fig_mlp = go.Figure(go.Bar(
            x=[f"L{l}" for l in ls],
            y=[mlp_attrs[l] for l in ls],
            marker_color=["#e74c3c" if mlp_attrs[l] >= 0 else "#3498db" for l in ls],
            hovertemplate="L%{x}<br>MLP attr: %{y:.4f}<extra></extra>",
        ))
        fig_mlp.update_layout(
            xaxis_title="Layer", yaxis_title="Attribution",
            height=280, margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_mlp, use_container_width=True)

    # Corruption
    st.divider()
    st.subheader("Name-Swap Corruption")
    ca, cb = st.columns(2)
    ca.text_input("Clean prompt",     value=prompt_input,          disabled=True)
    cb.text_input("Corrupted prompt", value=result["corr_prompt"], disabled=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Logit Lens
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Logit Lens")
    st.caption(
        "Layer-by-layer logit difference + per-head direct effects  "
        "(nostalgebraist 2020 · Elhage et al. 2021 §2.3).  "
        "Shows when the model's preference for the target crystallises."
    )
    if "logit_lens" not in result:
        st.info("Enable **Include Logit Lens** in the sidebar and re-run.")
    else:
        ll = result["logit_lens"]

        # LD trajectory
        ld_vals  = ll["logit_diffs"]
        x_labels = ["Embed"] + [f"L{i}" for i in range(len(ld_vals) - 1)]

        fig_ll = go.Figure()
        fig_ll.add_trace(go.Scatter(
            x=x_labels, y=ld_vals, mode="lines+markers",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=7),
            hovertemplate="%{x}: LD=%{y:.4f}<extra></extra>",
            name="Logit diff",
        ))
        fig_ll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_ll.update_layout(
            xaxis_title="Position (embedding → last layer)",
            yaxis_title=f"LD  [logit({target_input.strip()}) − logit({distractor_input.strip()})]",
            height=320, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_ll, use_container_width=True)

        # Logit shifts per layer
        st.markdown("**Logit shifts per layer (ΔLD)**")
        shifts    = ll["logit_shifts"]
        shift_x   = [f"L{i}" for i in range(len(shifts))]
        shift_col = ["#e74c3c" if s >= 0 else "#3498db" for s in shifts]

        fig_shift = go.Figure(go.Bar(
            x=shift_x, y=shifts, marker_color=shift_col,
            hovertemplate="%{x}: Δ=%{y:.4f}<extra></extra>",
        ))
        fig_shift.update_layout(
            xaxis_title="Layer", yaxis_title="ΔLD",
            height=260, margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_shift, use_container_width=True)

        # Per-head direct effects heatmap
        st.markdown("**Per-head direct effects** (W_O @ z · unembed_dir — linear approx.)")
        hde = ll["head_direct_effects"]
        hde_grid = np.array([hde[l] for l in sorted(hde.keys())])  # [n_layers, n_heads]

        fig_hde = go.Figure(go.Heatmap(
            z=hde_grid,
            x=[f"H{h}" for h in range(n_heads)],
            y=[f"L{l}" for l in sorted(hde.keys())],
            colorscale="RdBu", zmid=0,
            hovertemplate="L%{y} H%{x}<br>Direct effect: %{z:.4f}<extra></extra>",
        ))
        fig_hde.update_layout(
            xaxis_title="Head", yaxis_title="Layer",
            height=400, margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_hde, use_container_width=True)
        st.caption(
            "⚠️  Direct effects use the linear approximation: LN scale is not applied "
            "per-head. Relative rankings are preserved; absolute values are directional."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Edge Attribution Patching
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Edge Attribution Patching (EAP)")
    st.caption(
        "Scores every directed edge (sender → receiver) in the computation graph.  "
        "EAP(u→v) = (∂LD/∂resid_pre_v) · Δh_u  (Syed et al. 2024).  "
        "More informative than node-level attribution patching."
    )

    tokens_corr = model.to_tokens(result["corr_prompt"])

    with st.spinner("Running Edge Attribution Patching…"):
        try:
            eap = gb.edge_attribution_patching(
                tokens_c, tokens_corr, t_tok, d_tok, top_k=30,
            )
        except Exception as e:
            st.error(f"EAP failed: {e}")
            eap = None

    if eap:
        top_edges = eap["top_edges"]
        if top_edges:
            labels   = [f"{e['sender']} → {e['receiver']}" for e in top_edges]
            scores   = [e["score"] for e in top_edges]
            colors_e = ["#e74c3c" if s >= 0 else "#3498db" for s in scores]

            fig_eap = go.Figure(go.Bar(
                x=scores[::-1], y=labels[::-1],
                orientation="h",
                marker_color=colors_e[::-1],
                hovertemplate="%{y}<br>Score: %{x:.4f}<extra></extra>",
            ))
            fig_eap.update_layout(
                xaxis_title="EAP Score",
                height=max(300, min(len(top_edges) * 22, 700)),
                margin=dict(l=180, r=20, t=10, b=40),
            )
            st.plotly_chart(fig_eap, use_container_width=True)

            import pandas as pd
            st.dataframe(
                pd.DataFrame(top_edges)[["sender", "receiver", "score"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.warning("No edges found — check logit difference sign.")

        st.metric("Total edges scored", eap["n_edges"])
        st.metric("Clean LD", f"{eap['clean_ld']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Token Attribution
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Token Attribution (Saliency Map)")
    st.caption(
        "Per-input-token signed attribution via gradient × embedding  "
        "(Simonyan et al. 2014).  "
        "Positive = token pushes model toward target. Negative = toward distractor."
    )
    if tok_attr is None:
        st.warning("Token attribution failed. Check that the target/distractor tokens are valid.")
    else:
        tok_strs = tok_attr["token_strs"]
        tok_vals = tok_attr["attributions"]
        colors_t = ["#e74c3c" if v >= 0 else "#3498db" for v in tok_vals]

        fig_tok = go.Figure(go.Bar(
            x=tok_strs, y=tok_vals,
            marker_color=colors_t,
            hovertemplate="Token: %{x}<br>Attribution: %{y:.4f}<extra></extra>",
        ))
        fig_tok.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_tok.update_layout(
            xaxis_title="Input token",
            yaxis_title="Attribution score (gradient × embedding)",
            height=320, margin=dict(l=40, r=20, t=10, b=60),
        )
        st.plotly_chart(fig_tok, use_container_width=True)

        st.markdown("**Top 5 tokens by |attribution|:**")
        import pandas as pd
        top_df = pd.DataFrame(tok_attr["top_tokens"])[
            ["rank", "token_str", "attribution", "position"]
        ]
        top_df["attribution"] = top_df["attribution"].map("{:+.4f}".format)
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        st.caption(
            "⚠️  Gradient × input is a first-order approximation. For large prompts or "
            "adversarial inputs, integrated gradients (coming in a future release) would "
            "be more accurate at the token level."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Head Composition
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Head Composition Scores  (Elhage et al. 2021)")
    st.caption(
        "Q/K/V composition scores for circuit head pairs.  "
        "C_Q = ‖W_Q^recv · W_OV^sender‖_F / (‖W_Q^recv‖_F · ‖W_OV^sender‖_F)  "
        "(Elhage et al. 2021, §3.2).  "
        "Large score → sender's residual-stream output is read by receiver's query/key/value."
    )
    if len(circuit) < 2:
        st.info("Need at least 2 circuit heads to compute composition scores. Run analysis first.")
    else:
        with st.spinner("Computing composition scores…"):
            try:
                from glassbox.composition import HeadCompositionAnalyzer
                comp_analyzer = HeadCompositionAnalyzer(model)
                comp_result   = comp_analyzer.all_composition_scores(circuit[:8], min_score=0.0)
            except Exception as e:
                st.error(f"Composition failed: {e}")
                comp_result = None

        if comp_result:
            kind_tab = st.radio("Composition type", ["Q (Query)", "K (Key)", "V (Value)"],
                                horizontal=True)
            kind_key = {"Q (Query)": "q", "K (Key)": "k", "V (Value)": "v"}[kind_tab]

            mat    = comp_result[kind_key]["matrix"]
            labels = comp_result["head_labels"]

            fig_comp = go.Figure(go.Heatmap(
                z=mat,
                x=labels,
                y=labels,
                colorscale="Viridis",
                hovertemplate="%{y} → %{x}<br>Score: %{z:.4f}<extra></extra>",
            ))
            fig_comp.update_layout(
                xaxis_title="Receiver head (reads)",
                yaxis_title="Sender head (writes)",
                height=420, margin=dict(l=100, r=20, t=20, b=80),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("**Significant composition edges:**")
            sig_edges = comp_result[kind_key]["significant_edges"]
            if sig_edges:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(sig_edges)[["sender", "receiver", "score"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No edges above the significance threshold.")

            st.caption(
                f"Mean {kind_key.upper()}-comp score: "
                f"{comp_result[kind_key]['mean_score']:.4f}  |  "
                f"Max: {comp_result[kind_key]['max_score']:.4f}  |  "
                "Only causally valid pairs (sender_layer < receiver_layer) shown."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Faithfulness Details
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Faithfulness Analysis")
    st.caption(
        "ERASER framework metrics (DeYoung et al. 2020).  "
        "Bootstrap CIs require multiple prompts — enter in the expander below."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Sufficiency",       f"{faith['sufficiency']:.4f}")
    c2.metric("Comprehensiveness", f"{faith['comprehensiveness']:.4f}")
    c3.metric("F1",                f"{faith['f1']:.4f}")

    st.markdown(f"**Category:** `{faith['category']}`")
    st.markdown("""
| Category | Condition |
|----------|-----------|
| `faithful` | suff > 0.7 and comp > 0.5 |
| `backup_mechanisms` | suff > 0.9 and comp < 0.4 |
| `moderate` | everything else |
| `incomplete` | suff < 0.5 |
| `weak` | suff < 0.6 and comp < 0.5 |
""")

    st.info(
        "**Sufficiency** is a first-order Taylor approximation in this run. "
        "It is accurate for tasks where head contributions are approximately additive. "
        "For exact causal sufficiency, run bootstrap_metrics() with multiple prompts."
    )

    with st.expander("Run bootstrap confidence intervals"):
        st.markdown(
            "Add more prompts below (one per line: `prompt | target | distractor`). "
            "Minimum 5 recommended for stable percentile CIs."
        )
        boot_input = st.text_area(
            "Prompts",
            value=(
                "When Mary and John went to the store, John gave a bottle to | Mary | John\n"
                "After Alice and Bob entered the room, Bob handed the key to | Alice | Bob\n"
                "When Sarah and Tom left the park, Tom passed the ball to | Sarah | Tom"
            ),
            height=120,
        )
        n_boot     = st.slider("Bootstrap resamples", 100, 1000, 300)
        boot_btn   = st.button("Run bootstrap", type="secondary")

        if boot_btn:
            prompts = []
            for line in boot_input.strip().split("\n"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    prompts.append(tuple(parts))
            if len(prompts) < 2:
                st.warning("Need at least 2 prompts.")
            else:
                with st.spinner(f"Running bootstrap (n={n_boot}, {len(prompts)} prompts)…"):
                    boot = gb.bootstrap_metrics(prompts, n_boot=n_boot)
                bc1, bc2, bc3 = st.columns(3)
                for col, metric in zip((bc1, bc2, bc3),
                                       ("sufficiency", "comprehensiveness", "f1")):
                    m = boot[metric]
                    col.metric(
                        metric.capitalize(),
                        f"{m['mean']:.3f}",
                        f"95% CI [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]",
                    )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Glassbox 2.0 v2.3.0 · "
    "[arXiv:2603.09988](https://arxiv.org/abs/2603.09988) · "
    "[GitHub](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) · "
    "[PyPI](https://pypi.org/project/glassbox-mech-interp/) · "
    "[HuggingFace](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) · "
    "Built by Ajay Pravin Mahale"
)
