import streamlit as st
import streamlit.components.v1 as components
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

def display_calculation_details(calculation_details: Dict[str, Any]) -> None:
    """計算詳細を表示"""
    st.subheader('📊 計算詳細')
    
    # 基本情報
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**現在の状況**")
        st.write(f"トップ: {calculation_details['leader']} ({calculation_details['leader_score']}点)")
        st.write(f"自分の点数: {calculation_details['my_score']}点")
        st.write(f"トップとの差: {calculation_details['top_diff']}点")
        st.write(f"役割: {calculation_details['role_str']}")
    
    with col2:
        st.markdown("**調整要素**")
        st.write(f"供託棒: {calculation_details['kyotaku_points']}点")
        st.write(f"積み棒（ロン）: {calculation_details['tsumibo_points']}点")
        st.write(f"積み棒（ツモ）: {calculation_details['tsumo_tsumibo_points']}点")
    
    # 統計情報
    st.markdown("**📈 統計情報**")
    scores = calculation_details['current_scores']
    score_values = list(scores.values())
    score_values.sort(reverse=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("最高点", f"{score_values[0]}点")
    with col2:
        st.metric("最低点", f"{score_values[-1]}点")
    with col3:
        st.metric("平均点", f"{sum(score_values)//len(score_values)}点")
    with col4:
        st.metric("点差範囲", f"{score_values[0] - score_values[-1]}点")

def display_analysis_summary(data: Dict[str, Any]) -> None:
    """分析サマリーを表示"""
    st.subheader('🎯 分析サマリー')
    
    results = data['results']
    calculation_details = data['calculation_details']
    
    # 最も簡単な条件を特定
    easiest_conditions = []
    for result in results:
        if result['rank'] != '不可能':
            easiest_conditions.append({
                'condition': result['条件'],
                'rank': result['rank'],
                'points': result['need_points'],
                'is_direct': result['is_direct']
            })
    
    if easiest_conditions:
        # 点数でソート
        easiest_conditions.sort(key=lambda x: x['points'])
        
        st.markdown("**💡 推奨戦略**")
        st.write(f"最も簡単な条件: **{easiest_conditions[0]['condition']}** ({easiest_conditions[0]['rank']})")
        
        if len(easiest_conditions) > 1:
            st.write("その他の選択肢:")
            for i, condition in enumerate(easiest_conditions[1:3]):  # 上位3つまで表示
                st.write(f"- {condition['condition']} ({condition['rank']})")
    
    # 直撃の価値を分析
    direct_results = [r for r in results if r.get('is_direct', False)]
    if direct_results:
        direct_result = direct_results[0]
        other_results = [r for r in results if not r.get('is_direct', False)]
        
        if other_results:
            other_result = other_results[0]
            if 'need_points_original' in direct_result:
                savings = direct_result['need_points_original'] - direct_result['need_points']
                st.markdown("**🎯 直撃の価値**")
                st.write(f"直撃することで必要点数を **{savings}点** 節約できます")
                st.write(f"（{direct_result['need_points_original']}点 → {direct_result['need_points']}点）")

def display_calculation_steps(result: Dict[str, Any]) -> None:
    """計算過程を表示"""
    if 'calculation_steps' not in result:
        return
    
    steps = result['calculation_steps']
    
    st.markdown(f"**{result['条件']}の計算過程**")
    
    # 計算過程をステップバイステップで表示
    step_items = []
    
    if 'top_diff' in steps:
        step_items.append(f"1. トップとの差: {steps['top_diff']}点")
    
    if 'minus_kyotaku' in steps:
        step_items.append(f"2. 供託棒を引く: {steps['top_diff']} - {steps['top_diff'] - steps['minus_kyotaku']} = {steps['minus_kyotaku']}点")
    
    if 'minus_tsumibo' in steps:
        step_items.append(f"3. 積み棒を引く: {steps['minus_kyotaku']} - {steps['minus_kyotaku'] - steps['minus_tsumibo']} = {steps['minus_tsumibo']}点")
    
    if 'minus_tsumo_tsumibo' in steps:
        step_items.append(f"3. ツモ積み棒を引く: {steps['minus_kyotaku']} - {steps['minus_kyotaku'] - steps['minus_tsumo_tsumibo']} = {steps['minus_tsumo_tsumibo']}点")
    
    if 'divided_by_2' in steps:
        step_items.append(f"4. 直撃ボーナス（半分）: {steps['minus_tsumibo']} ÷ 2 = {steps['divided_by_2']}点")
    
    if 'divided_by_3' in steps:
        step_items.append(f"4. 3人で割る: {steps['minus_tsumo_tsumibo']} ÷ 3 = {steps['divided_by_3']}点")
    
    if 'ceiled' in steps:
        step_items.append(f"5. 切り上げ: {steps['ceiled']}点")
    
    if 'final_points' in steps:
        step_items.append(f"6. **最終必要点数: {steps['final_points']}点**")
    
    for item in step_items:
        st.write(item)
    
    # 役の詳細情報を表示
    if 'calculation_details' in result:
        details = result['calculation_details']
        st.markdown("**役の詳細**")
        
        if details.get('type') == '通常役':
            st.write(f"役種: {details.get('fu', 'N/A')}符{details.get('han', 'N/A')}翻")
            st.write(f"理由: {details.get('reason', 'N/A')}")
            if 'payment' in details:
                st.write(f"支払い: {details['payment']}")
        elif details.get('type') in ['満貫', '跳満', '倍満', '三倍満', '役満']:
            st.write(f"役種: {details['type']}")
            st.write(f"理由: {details.get('reason', 'N/A')}")
            if 'payment' in details:
                st.write(f"支払い: {details['payment']}")
        else:
            st.write(f"理由: {details.get('reason', 'N/A')}")
    
    st.markdown("---")

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
    
    # 計算結果表示エリアのプレースホルダー
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
            
            # 計算結果を表示
            with results_container:
                # 結果セクションの開始マーカー
                st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
                
                # トップとの差を表示
                display_top_difference(top_diff, leader)
                
                # 計算詳細を表示
                display_calculation_details(data['calculation_details'])
                
                # 分析サマリーを表示
                display_analysis_summary(data)
                
                # 逆転条件を表示
                st.subheader('逆転条件（直撃ロン / 他家放銃ロン / ツモ）')
                cols = st.columns(3)
                
                for i, result in enumerate(data['results']):
                    with cols[i]:
                        render_condition_card(result)
                
                # 計算過程の詳細表示
                st.subheader('🔍 計算過程の詳細')
                for result in data['results']:
                    display_calculation_steps(result)
            
            # スクロール用のJavaScript
            scroll_script = """
            <script>
            setTimeout(function() {
                const resultsSection = document.querySelector('#results-section');
                if (resultsSection) {
                    resultsSection.scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'start' 
                    });
                }
            }, 100);
            </script>
            """
            components.html(scroll_script, height=0)
                    
        except Exception as e:
            st.error(f"計算エラーが発生しました: {str(e)}")
            st.info("入力値を確認して再度お試しください。")

if __name__ == "__main__":
    main()