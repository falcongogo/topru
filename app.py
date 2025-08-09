import streamlit as st
from calculate_conditions import calculate_conditions

st.set_page_config(page_title="TOPる", page_icon="🀄", layout="centered")
st.title("TOPる – 麻雀オーラス逆転条件計算ツール")

# --- デバッグ用ダミーデータ ---
if "scores" not in st.session_state:
    st.session_state.scores = {"自分": 32000, "下家": 25000, "対面": 30000, "上家": 13000}
    st.session_state.oya = "下家"
    st.session_state.tsumibo = 1
    st.session_state.kyotaku = 0

st.subheader("点数入力（百点単位）")
scores = {}
players = ["自分", "下家", "対面", "上家"]
for p in players:
    default_val = st.session_state.scores[p] // 100
    val = st.number_input(f"{p} の点数", min_value=0, max_value=999, step=1, value=default_val, key=p)
    scores[p] = val * 100

oya = st.selectbox("親の位置", players, index=players.index(st.session_state.oya))
tsumibo = st.number_input("積み棒", min_value=0, step=1, value=st.session_state.tsumibo)
kyotaku = st.number_input("供託棒", min_value=0, step=1, value=st.session_state.kyotaku)

if st.button("計算"):
    st.session_state.scores = scores
    st.session_state.oya = oya
    st.session_state.tsumibo = tsumibo
    st.session_state.kyotaku = kyotaku

    results = calculate_conditions(scores, oya, tsumibo, kyotaku)

    st.markdown("## 結果")
    for r in results:
        if r["rank"] == "不可能":
            color = "#ff4d4d"  # 赤
        elif r["points"] == 0:
            color = "#00cc66"  # 緑
        else:
            color = "#ffaa00"  # 橙
        st.markdown(
            f"<div style='background-color:{color};padding:8px;border-radius:6px;'>"
            f"<b>{r['条件']}</b> : {r['rank']}（{r['points']} 点）<br>"
            f"<small>{r['detail']}</small>"
            "</div>",
            unsafe_allow_html=True
        )
