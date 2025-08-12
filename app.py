"""
麻雀の逆転条件を計算するStreamlitウェブアプリケーション。

このモジュールは「TOPる」ツールのユーザーインターフェースを提供します。
ユーザーは4人のプレイヤーの点数、現在の親、積み棒、供託棒を入力できます。
アプリケーションは、オーラスでトップになるための必要条件を計算し、表示します。
"""
import streamlit as st
from calculate_conditions import calculate_conditions
from typing import Dict, Any

# --- 定数 ---
PLAYERS = ['自分', '下家', '対面', '上家']
DEFAULT_SCORES = {'自分': 28000, '下家': 35000, '対面': 30000, '上家': 27000}
COLOR_THRESHOLDS = {
    'red': 10000,
    'orange': 5000,
    'green': 0
}

# --- 関数 ---

def initialize_session_state():
    """
    Streamlitのセッション状態をデフォルト値で初期化します。

    'scores', 'oya', 'tsumibo', 'kyotaku' がセッション状態にない場合、
    この関数はそれらをデフォルトの開始値に設定します。
    これにより、アプリの初回実行時やリセット時に状態が保証されます。
    """
    if 'scores' not in st.session_state:
        st.session_state.scores = DEFAULT_SCORES
        st.session_state.oya = '下家'
        st.session_state.tsumibo = 0
        st.session_state.kyotaku = 0

def validate_inputs(scores: Dict[str, int], tsumibo: int, kyotaku: int) -> bool:
    """
    ユーザーが入力した点数と棒の値が正しいか検証します。

    Args:
        scores: 4人全員の点数を含む辞書。
        tsumibo: 積み棒の本数。
        kyotaku: 供託棒の本数。

    Returns:
        全ての入力が有効（非負）であればTrue、そうでなければFalse。
        検証に失敗した場合はUIにエラーメッセージを表示します。
    """
    if any(score < 0 for score in scores.values()):
        st.error("点数は0以上で入力してください")
        return False
    if tsumibo < 0 or kyotaku < 0:
        st.error("積み棒・供託棒は0以上で入力してください")
        return False
    return True

def get_condition_style(result: Dict[str, Any]) -> Dict[str, str]:
    """
    計算結果の内容に基づいて、結果表示カードのUIスタイルを決定します。

    Args:
        result: 逆転条件の詳細を含む辞書。

    Returns:
        スタイリング情報（'bgcolor', 'badge', 'style'）を含む辞書。
    """
    rank = result['rank']
    is_direct = result['is_direct']
    
    if rank == '不可能':
        return {'bgcolor': '#ffd6d6', 'badge': "❌", 'style': ''}
    elif rank.startswith('満貫'):
        return {'bgcolor': '#ffe566', 'badge': "🌟", 'style': 'font-weight:700;'}
    elif any(x in rank for x in ['跳満', '倍満', '三倍満', '役満']):
        return {'bgcolor': '#ffd700', 'badge': "💎", 'style': 'font-weight:700;'}
    elif is_direct:
        return {'bgcolor': '#e0f7fa', 'badge': "直撃", 'style': 'font-weight:700;'}
    else:
        return {'bgcolor': '#fff6e6', 'badge': "", 'style': ''}

def render_score_inputs() -> Dict[str, int]:
    """
    4人全員の点数入力フィールドを描画します。

    Streamlitのカラム機能を使用して、入力ボックスをきれいに配置します。

    Returns:
        ユーザーが入力した最新の点数を含む辞書。
    """
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
    """
    単一の逆転条件の結果カードを描画します。

    Args:
        result: ランク、表示点、スタイリング情報など、
                逆転条件の詳細を含む辞書。
    """
    style_config = get_condition_style(result)
    
    total_info = ""
    if 'total_points' in result and 'opponent_loss' in result and 'difference_points' in result:
        if isinstance(result['opponent_loss'], str):
            total_info = f"<br><small>合計: {result['total_points']}点<br>相手支払い: {result['opponent_loss']}<br>差分: {result['difference_points']}点</small>"
        else:
            total_info = f"<br><small>合計: {result['total_points']}点<br>相手支払い: {result['opponent_loss']}点<br>差分: {result['difference_points']}点</small>"
    
    st.markdown(f"""
    <div style='background:{style_config["bgcolor"]};padding:12px;border-radius:8px'>
        <span style='font-size:1.3em;{style_config["style"]}'>{style_config["badge"]} {result['条件']}</span><br>
        <span style='font-size:1.1em;{style_config["style"]}'>{result['rank']}（{result['display']}）</span>
        {total_info}
    </div>
    """, unsafe_allow_html=True)

def display_top_difference(top_diff: int, leader: str) -> None:
    """
    トップのプレイヤーとの点数差を表示します。

    点差の大きさによってテキストの色が変わります。

    Args:
        top_diff: トップとの点差。
        leader: 現在トップのプレイヤー名。
    """
    if top_diff <= 0:
        st.success("あなたは現在トップです！")
    else:
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
    """
    Streamlitアプリケーションを実行するメイン関数。

    ページ設定、タイトル設定、UI描画、入力ハンドリング、
    計算ロジックの呼び出しを統括します。
    """
    st.set_page_config(page_title='TOPる', page_icon='🀄', layout='wide')
    st.title('TOPる – 麻雀オーラス逆転条件計算ツール')
    
    initialize_session_state()
    
    scores = render_score_inputs()
    
    oya = st.selectbox('親の位置', PLAYERS, index=PLAYERS.index(st.session_state.oya))
    tsumibo = st.number_input('積み棒本数', min_value=0, step=1, value=st.session_state.tsumibo)
    kyotaku = st.number_input('供託棒本数', min_value=0, step=1, value=st.session_state.kyotaku)
    
    results_container = st.container()
    
    if st.button('計算', type='primary'):
        if not validate_inputs(scores, tsumibo, kyotaku):
            return
        
        st.session_state.scores = scores
        st.session_state.oya = oya
        st.session_state.tsumibo = tsumibo
        st.session_state.kyotaku = kyotaku
        
        try:
            data = calculate_conditions(scores, oya, tsumibo, kyotaku)
            top_diff = data['top_diff']
            leader = data['leader']
            
            st.markdown('<div id="results"></div>', unsafe_allow_html=True)
            
            with results_container:
                display_top_difference(top_diff, leader)
                
                st.subheader('逆転条件（直撃ロン / 他家放銃ロン / ツモ）')
                cols = st.columns(3)
                
                for i, result in enumerate(data['results']):
                    with cols[i]:
                        render_condition_card(result)
            
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