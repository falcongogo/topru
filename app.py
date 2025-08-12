"""
Streamlit-based web application for calculating Mahjong win conditions.

This module provides the user interface for the "TOPる" tool. Users can input
the scores of four players, the current dealer (oya), bonus sticks (tsumibo),
and riichi sticks (kyotaku). The application then calculates the necessary
conditions to win in the final round (All-Last) and displays them.
"""
import streamlit as st
from calculate_conditions import calculate_conditions
from typing import Dict, Any

# --- Constants ---
PLAYERS = ['自分', '下家', '対面', '上家']
DEFAULT_SCORES = {'自分': 28000, '下家': 35000, '対面': 30000, '上家': 27000}
COLOR_THRESHOLDS = {
    'red': 10000,
    'orange': 5000,
    'green': 0
}

# --- Functions ---

def initialize_session_state():
    """
    Initializes the Streamlit session state with default values.

    If 'scores', 'oya', 'tsumibo', or 'kyotaku' are not already in the
    session state, this function sets them to default starting values.
    This ensures the app has a consistent state on first run or reset.
    """
    if 'scores' not in st.session_state:
        st.session_state.scores = DEFAULT_SCORES
        st.session_state.oya = '下家'
        st.session_state.tsumibo = 0
        st.session_state.kyotaku = 0

def validate_inputs(scores: Dict[str, int], tsumibo: int, kyotaku: int) -> bool:
    """
    Validates user inputs for scores and sticks.

    Args:
        scores: A dictionary containing the scores of all four players.
        tsumibo: The number of bonus sticks.
        kyotaku: The number of riichi sticks.

    Returns:
        True if all inputs are valid (non-negative), False otherwise.
        Displays an error message in the UI if validation fails.
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
    Determines the UI styling for a result card based on its content.

    Args:
        result: A dictionary containing the details of a win condition.

    Returns:
        A dictionary with styling information ('bgcolor', 'badge', 'style').
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
    Renders the score input fields for all four players.

    Uses Streamlit's columns to create a neat layout for the input boxes.

    Returns:
        A dictionary containing the latest scores entered by the user.
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
    Renders a single result card for a win condition.

    Args:
        result: A dictionary containing the details of a win condition,
                including rank, display points, and styling info.
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
    Displays the point difference to the top player.

    The color of the text changes based on how large the difference is.

    Args:
        top_diff: The point difference to the leader.
        leader: The name of the player currently in the lead.
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
    The main function to run the Streamlit application.

    Sets up the page configuration, title, and orchestrates the UI rendering,
    input handling, and calculation logic.
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