"""
Glassbox 2.0 - Interactive Mechanistic Interpretability Dashboard
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title='Glassbox 2.0', page_icon='🔬', layout='wide')

@st.cache_resource(show_spinner='Loading model...')
def load_model(model_name):
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2
    model = HookedTransformer.from_pretrained(model_name)
    return model, GlassboxV2(model)

st.sidebar.title('🔬 Glassbox 2.0')
st.sidebar.markdown('**Mechanistic Interpretability**')
st.sidebar.divider()
model_choice = st.sidebar.selectbox('Model', ['gpt2', 'gpt2-medium', 'gpt2-large'])
prompt_input = st.sidebar.text_area('Prompt',
    value='When Mary and John went to the store, John gave a bottle to', height=100)
target_input = st.sidebar.text_input('Target token', value='Mary')
distractor_input = st.sidebar.text_input('Distractor token', value='John')
run_btn = st.sidebar.button('Run Analysis', type='primary', use_container_width=True)

st.title('🔬 Glassbox 2.0 — Circuit Analysis')
st.caption('Attribution patching · O(3) complexity · Bootstrap 95% CI · FCAS alignment')

if not run_btn:
    st.info('Fill in the prompt and tokens on the left, then click Run Analysis.')
    st.stop()

model, gb = load_model(model_choice)
with st.spinner('Running attribution patching...'):
    try:
        clean_tok = model.to_tokens(prompt_input)
        t_tok = model.to_single_token(target_input)
        d_tok = model.to_single_token(distractor_input)
        scores, clean_ld = gb.attribution_patching(clean_tok, clean_tok, t_tok, d_tok)
        result = gb.analyze(prompt_input, target_input, distractor_input)
    except Exception as e:
        st.error(f'Analysis failed: {e}')
        st.stop()

faith = result['faithfulness']
circuit = result['circuit']
st.subheader('Faithfulness Scores')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Sufficiency', f"{faith['sufficiency']:.1%}")
col2.metric('Comprehensiveness', f"{faith['comprehensiveness']:.1%}")
col3.metric('F1', f"{faith['f1']:.1%}")
col4.metric('Category', faith['category'].upper())
st.divider()

st.subheader('Attribution Heatmap')
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
grid = np.zeros((n_layers, n_heads))
for (layer, head), score in scores.items():
    grid[layer, head] = float(score)
fig = go.Figure(go.Heatmap(z=grid,
    x=[f'H{h}' for h in range(n_heads)],
    y=[f'L{l}' for l in range(n_layers)],
    colorscale='RdBu', zmid=0))
fig.update_layout(xaxis_title='Head', yaxis_title='Layer', height=420)
st.plotly_chart(fig, use_container_width=True)
st.divider()

st.subheader(f'Minimum Faithful Circuit ({len(circuit)} heads)')
if circuit:
    cgrid = np.zeros((n_layers, n_heads))
    for (layer, head) in circuit:
        cgrid[layer, head] = 1.0
    fig2 = go.Figure(go.Heatmap(z=cgrid,
        x=[f'H{h}' for h in range(n_heads)],
        y=[f'L{l}' for l in range(n_layers)],
        colorscale=[[0,'#f0f2f6'],[1,'#e74c3c']], showscale=False))
    fig2.update_layout(xaxis_title='Head', yaxis_title='Layer', height=380)
    st.plotly_chart(fig2, use_container_width=True)
