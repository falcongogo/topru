import streamlit as st
from calculate_conditions import calculate_conditions

st.set_page_config(page_title="TOPる – 麻雀オーラス逆転条件計算ツール", page_icon="🀄", layout="wide")
st.title("TOPる – 麻雀オーラス逆転条件計算ツール")

# default demo state
if 'scores' not in st.session_state:
    st.session_state.scores = {"自分": 25000, "下家": 32000, "対面": 30000, "上家": 13000}
    st.session_state.oya = "下家"
    st.session_state.tsumibo = 1
    st.session_state.kyotaku = 0

st.subheader("点数入力（百点単位）")
players = ["自分", "下家", "対面", "上家"]
cols = st.columns(4)
scores = {}
for i, p in enumerate(players):
    with cols[i]:
        default = st.session_state.scores[p] // 100
        v = st.number_input(p + " の点数", min_value=0, max_value=999, step=1, value=default, key=p)
        scores[p] = v * 100

oya = st.selectbox("親の位置", players, index=players.index(st.session_state.oya))
tsumibo = st.number_input("積み棒本数", min_value=0, step=1, value=st.session_state.tsumibo)
kyotaku = st.number_input("供託棒本数", min_value=0, step=1, value=st.session_state.kyotaku)

if st.button("計算"):
    st.session_state.scores = scores
    st.session_state.oya = oya
    st.session_state.tsumibo = tsumibo
    st.session_state.kyotaku = kyotaku

    with st.spinner("計算中…"):
        results = calculate_conditions(scores, oya, tsumibo, kyotaku)

    st.subheader("逆転条件（直撃ロン / 他家放銃ロン / ツモ）")
    cols = st.columns(len(results))
    for i, r in enumerate(results):
        with cols[i]:
            # color coding
            if r['rank'] in ('不可能',):
                bgcolor = '#ffcccc'
            elif r['rank'] in ('満貫','不要'):
                bgcolor = '#ccffdd'
            else:
                bgcolor = '#fff4cc'
            st.markdown(f"""<div style='background:{bgcolor};padding:12px;border-radius:8px'>
                <h4 style='margin:0'>{r['条件']}</h4>
                <div style='font-size:18px;font-weight:600'>{r['rank']}</div>
                <div style='font-size:16px'>点数: {r['points']}</div>
                <div style='font-size:12px;color:#555;margin-top:6px'>{r['detail']}</div>
                </div>""", unsafe_allow_html=True)
