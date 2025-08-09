import streamlit as st
from calculate_conditions import calculate_conditions

st.title("TOPる - 麻雀オーラス逆転条件計算ツール")

# 初期ダミーデータ（自分がトップではない）
scores = [
    28000,  # 自分（南家）
    35000,  # 下家（西家、トップ）
    30000,  # 対面（北家）
    27000   # 上家（東家）
]

dealer = st.selectbox("親の位置", ["自分", "下家", "対面", "上家"], index=3)
tsumi = st.number_input("積み棒", 0, 10, 0)
kyotaku = st.number_input("供託棒", 0, 10, 0)

if st.button("計算"):
    result = calculate_conditions(scores, ["自分", "下家", "対面", "上家"].index(dealer), tsumi, kyotaku)
    st.write("### 結果")
    for k, v in result.items():
        st.write(f"{k}: {v}")
