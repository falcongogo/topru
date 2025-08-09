import streamlit as st
from calculate_conditions import calculate_conditions

st.set_page_config(page_title='TOPる', page_icon='🀄', layout='wide')
st.title('TOPる – 麻雀オーラス逆転条件計算ツール')

# default demo (self not top)
if 'scores' not in st.session_state:
    st.session_state.scores = {'自分': 28000, '下家': 35000, '対面': 30000, '上家': 27000}
    st.session_state.oya = '下家'
    st.session_state.tsumibo = 0
    st.session_state.kyotaku = 0

st.subheader('点数入力（百点単位）')
players = ['自分', '下家', '対面', '上家']
cols = st.columns(4)
scores = {}
for i, p in enumerate(players):
    with cols[i]:
        default = st.session_state.scores[p] // 100
        v = st.number_input(p + ' の点数', min_value=0, max_value=999, step=1, value=default, key=p)
        scores[p] = v * 100

oya = st.selectbox('親の位置', players, index=players.index(st.session_state.oya))
tsumibo = st.number_input('積み棒本数', min_value=0, step=1, value=st.session_state.tsumibo)
kyotaku = st.number_input('供託棒本数', min_value=0, step=1, value=st.session_state.kyotaku)

if st.button('計算'):
    st.session_state.scores = scores
    st.session_state.oya = oya
    st.session_state.tsumibo = tsumibo
    st.session_state.kyotaku = kyotaku

    data = calculate_conditions(scores, oya, tsumibo, kyotaku)
    top_diff = data['top_diff']
    leader = data['leader']

    # Top diff display
    if top_diff <= 0:
        st.markdown("""<h3 style='color:green;'>あなたは現在トップです！</h3>""", unsafe_allow_html=True)
    else:
        color = 'red' if top_diff >= 10000 else 'orange' if top_diff >= 5000 else 'green'
        st.markdown(f"""<h2 style='color:{color};'>TOPとの差：<span style='font-weight:700'>{top_diff} 点</span>（トップ: {leader}）</h2>""", unsafe_allow_html=True)

    st.subheader('逆転条件（直撃ロン / 他家放銃ロン / ツモ）')
    cols = st.columns(3)
    for i, r in enumerate(data['results']):
        with cols[i]:
            if r['rank'] == '不可能':
                bgcolor = '#ffd6d6'
            elif r['rank'] in ('満貫','不要'):
                bgcolor = '#e6ffef'
            else:
                bgcolor = '#fff6e6'
            st.markdown(f"""<div style='background:{bgcolor};padding:12px;border-radius:8px'>
                <h4 style='margin:0'>{r['条件']}</h4>
                <div style='font-size:18px;font-weight:700'>{r['rank']}</div>
                <div style='font-size:16px'>表示点数: {r['display']}</div>
                <div style='font-size:12px;color:#555;margin-top:6px'>必要（内部）: {r['need_points']} 点</div>
                </div>""", unsafe_allow_html=True)
    st.write('---')
    st.caption('注: 満貫未満の表示は20/30/40/50符のみ。60符以上はUI上表示しません。')
