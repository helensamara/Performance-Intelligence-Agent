"""
app.py — Streamlit chat interface
Run with: streamlit run app.py
"""
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loader import load_sugarwod
from analysis import attendance, sentiment, performance, ml_models
from agent import chat

st.set_page_config(page_title='Performance Agent', page_icon='🏋️', layout='wide')

# ── Load data (cached) ────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_sugarwod()
    df = sentiment.enrich(df)
    return df

try:
    df = get_data()
except Exception as e:
    st.error(f'Could not load data/workouts.csv: {e}')
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title('🏋️ Performance Agent')
    st.markdown('---')
    st.markdown('**Dataset**')
    st.success(f'{len(df)} workouts')
    st.caption(f"{df['date'].min().date()} → {df['date'].max().date()}")
    st.markdown(f"PRs: **{int(df['is_pr'].sum())}**")
    st.markdown(f"RX: **{(df['rx_or_scaled']=='RX').mean()*100:.0f}%**  ·  "
                f"Scaled: **{(df['rx_or_scaled']=='SCALED').mean()*100:.0f}%**")
    st.markdown('---')
    st.markdown('**Quick charts**')
    for label, key in [
        ('📅 Attendance',  'attendance'),
        ('😊 Sentiment',   'sentiment'),
        ('📈 RX Rate',     'rx_rate'),
        ('💪 Strength',    'strength'),
        ('🔗 Lift Correlation', 'correlation'),
        ('🤖 Clusters',   'clusters'),
        ('🔮 PR Forecast', 'forecast'),
        ('⚠️ Anomalies',  'anomalies'),
    ]:
        if st.button(label):
            st.session_state['chart'] = key
    st.markdown('---')
    if st.button('🗑️ Clear chat'):
        st.session_state['messages'] = []
        st.rerun()

# ── Chart panel ───────────────────────────────────────────────────
chart = st.session_state.get('chart')
if chart:
    with st.expander(f'Chart: {chart}', expanded=True):
        try:
            if chart == 'attendance':
                fig, _ = attendance.monthly_chart(df); st.pyplot(fig); plt.close(fig)
                fig = attendance.weekly_trend(df);     st.pyplot(fig); plt.close(fig)
                fig = attendance.day_of_week(df);      st.pyplot(fig); plt.close(fig)
            elif chart == 'sentiment':
                fig = sentiment.sentiment_over_time(df); st.pyplot(fig); plt.close(fig)
                fig = sentiment.word_frequency(df);      st.pyplot(fig); plt.close(fig)
                fig, r = sentiment.sentiment_vs_performance(df)
                st.pyplot(fig); plt.close(fig)
                st.caption(f'Correlation r = {r:.3f}')
            elif chart == 'rx_rate':
                fig = performance.rx_rate_over_time(df); st.pyplot(fig); plt.close(fig)
            elif chart == 'strength':
                fig = performance.strength_progression(df); st.pyplot(fig); plt.close(fig)
            elif chart == 'correlation':
                fig, _ = performance.lift_correlation(df); st.pyplot(fig); plt.close(fig)
            elif chart == 'clusters':
                _, _, fig = ml_models.cluster_workouts(df); st.pyplot(fig); plt.close(fig)
            elif chart == 'forecast':
                _, fig = ml_models.forecast_prs(df); st.pyplot(fig); plt.close(fig)
            elif chart == 'anomalies':
                _, fig = ml_models.detect_anomalies(df); st.pyplot(fig); plt.close(fig)
        except Exception as e:
            st.error(f'Chart error: {e}')

# ── Chat ──────────────────────────────────────────────────────────
st.markdown('## Chat with your data')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Render history (string messages only)
for msg in st.session_state['messages']:
    if isinstance(msg, dict) and isinstance(msg.get('content'), str):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

# Suggested questions when chat is empty
if not st.session_state['messages']:
    st.markdown('**Try asking:**')
    cols = st.columns(2)
    suggestions = [
        'How consistent have I been training?',
        'When is my next PR likely?',
        'What do my notes say about how I feel?',
        'Am I overtraining?',
        'What are my workout archetypes?',
        'Which months produced the most PRs?',
    ]
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, key=f's{i}'):
            st.session_state['pending'] = q
            st.rerun()

user_input = st.session_state.pop('pending', None) or st.chat_input('Ask your agent anything...')

if user_input:
    with st.chat_message('user'):
        st.markdown(user_input)

    # Pass full history to agent (includes tool_use/tool_result blocks from prior turns)
    history = list(st.session_state['messages'])
    history.append({'role': 'user', 'content': user_input})

    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            try:
                reply, updated = chat(history)
                st.markdown(reply)
            except Exception as e:
                reply = f'Error: {e}'
                updated = history + [{'role': 'assistant', 'content': reply}]
                st.error(reply)

    st.session_state['messages'] = updated
