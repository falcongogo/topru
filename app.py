import streamlit as st
from calculate_conditions import calculate_conditions
from typing import Dict, Any

# 定数定義
PLAYERS = ['自分', '下家', '対面', '上家']
DEFAULT_SCORES = {'自分': 28000, '下家': 35000, '対面': 30000, '上家': 27000}
COLOR_THRESHOLDS = {
    'red': 10000,
    'orange': 5000,
    'green': 0
}

def initialize_session_state():
    """セッション状態の初期化"""
    if 'scores' not in st.session_state:
        st.session_state.scores = DEFAULT_SCORES
        st.session_state.oya = '下家'
        st.session_state.tsumibo = 0
        st.session_state.kyotaku = 0

def validate_inputs(scores: Dict[str, int], tsumibo: int, kyotaku: int) -> bool:
    """入力値の検証"""
    if any(score < 0 for score in scores.values()):
        st.error("点数は0以上で入力してください")
        return False
    if tsumibo < 0 or kyotaku < 0:
        st.error("積み棒・供託棒は0以上で入力してください")
        return False
    return True

def get_condition_style(result: Dict[str, Any]) -> Dict[str, str]:
    """条件に応じたスタイル設定を取得"""
    rank = result['rank']
    is_direct = result['is_direct']
    
    if rank == '不可能':
        return {
            'bgcolor': '#ffd6d6',
            'badge': "❌",
            'style': ''
        }
    elif rank.startswith('満貫'):
        return {
            'bgcolor': '#ffe566',
            'badge': "🌟",
            'style': 'font-weight:700;'
        }
    elif any(x in rank for x in ['跳満', '倍満', '三倍満', '役満']):
        return {
            'bgcolor': '#ffd700',
            'badge': "💎",
            'style': 'font-weight:700;'
        }
    elif is_direct:
        return {
            'bgcolor': '#e0f7fa',
            'badge': "直撃",
            'style': 'font-weight:700;'
        }
    else:
        return {
            'bgcolor': '#fff6e6',
            'badge': "",
            'style': ''
        }

def render_score_inputs() -> Dict[str, int]:
    """点数入力UIの描画"""
    st.subheader('点数入力（百点単位）')
    cols = st.columns(4)
    scores = {}
    
    for i, player in enumerate(PLAYERS):
        with cols[i]:
            default = st.session_state.scores[player] // 100
            value = st.number_input(
                f'{player} の点数', 
                min_value=0, 
                max_value=999, 
                step=1, 
                value=default, 
                key=f'score_{player}'
            )
            scores[player] = value * 100
    
    return scores

def render_condition_card(result: Dict[str, Any]) -> None:
    """条件カードの描画"""
    style_config = get_condition_style(result)
    
    st.markdown(f"""
    <div style='background:{style_config["bgcolor"]};padding:12px;border-radius:8px'>
        <span style='font-size:1.3em;{style_config["style"]}'>{style_config["badge"]} {result['条件']}</span><br>
        <span style='font-size:1.1em;{style_config["style"]}'>{result['rank']}（{result['display']}）</span>
    </div>
    """, unsafe_allow_html=True)

def display_top_difference(top_diff: int, leader: str) -> None:
    """トップとの差を表示"""
    if top_diff <= 0:
        st.success("あなたは現在トップです！")
    else:
        # 色の決定
        if top_diff >= COLOR_THRESHOLDS['red']:
            color = 'red'
        elif top_diff >= COLOR_THRESHOLDS['orange']:
            color = 'orange'
        else:
            color = 'green'
        
        st.markdown(f"""
        <h2 style='color:{color};'>
            TOPとの差：<span style='font-weight:700'>{top_diff} 点</span>（トップ: {leader}）
        </h2>
        """, unsafe_allow_html=True)

def main():
    """メインアプリケーション"""
    st.set_page_config(page_title='TOPる', page_icon='🀄', layout='wide')
    st.title('TOPる – 麻雀オーラス逆転条件計算ツール')
    
    # セッション状態の初期化
    initialize_session_state()
    
    # 点数入力
    scores = render_score_inputs()
    
    # 親・積み棒・供託棒入力
    oya = st.selectbox('親の位置', PLAYERS, index=PLAYERS.index(st.session_state.oya))
    tsumibo = st.number_input('積み棒本数', min_value=0, step=1, value=st.session_state.tsumibo)
    kyotaku = st.number_input('供託棒本数', min_value=0, step=1, value=st.session_state.kyotaku)
    
    # 計算結果表示エリアのコンテナを作成
    results_container = st.container()
    
    # 計算ボタン
    if st.button('計算', type='primary'):
        # 入力値の検証
        if not validate_inputs(scores, tsumibo, kyotaku):
            return
        
        # セッション状態の更新
        st.session_state.scores = scores
        st.session_state.oya = oya
        st.session_state.tsumibo = tsumibo
        st.session_state.kyotaku = kyotaku
        
        try:
            # 条件計算
            data = calculate_conditions(scores, oya, tsumibo, kyotaku)
            top_diff = data['top_diff']
            leader = data['leader']
            
            # 計算結果エリアに移動するためのアンカー
            st.markdown('<div id="results"></div>', unsafe_allow_html=True)
            
            with results_container:
                # トップとの差を表示
                display_top_difference(top_diff, leader)
                
                # 逆転条件を表示
                st.subheader('逆転条件（直撃ロン / 他家放銃ロン / ツモ）')
                cols = st.columns(3)
                
                for i, result in enumerate(data['results']):
                    with cols[i]:
                        render_condition_card(result)
            
            # 計算結果に自動スクロール
            st.markdown("""
            <script>
                document.getElementById('results').scrollIntoView({behavior: 'smooth'});
            </script>
            """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"計算エラーが発生しました: {str(e)}")
            st.info("入力値を確認して再度お試しください。")

if __name__ == "__main__":
    main()