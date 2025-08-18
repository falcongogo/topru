import streamlit as st
from calculate_conditions import calculate_conditions
import config
import ui

def initialize_session_state():
    """セッション状態の初期化"""
    if 'scores' not in st.session_state:
        st.session_state.scores = config.DEFAULT_SCORES
        st.session_state.oya = '下家'
        st.session_state.tsumibo = 0
        st.session_state.kyotaku = 0
        st.session_state.image_processor = None

def load_css(file_name):
    """CSSファイルを読み込んで適用する"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    """メインアプリケーション"""
    st.set_page_config(page_title='TOPる', page_icon='🀄', layout='wide')
    load_css('styles.css')

    st.title('TOPる – 麻雀オーラス逆転条件計算ツール')
    
    initialize_session_state()
    
    scores = ui.render_score_inputs()
    
    oya = st.selectbox('親の位置', config.PLAYERS, index=config.PLAYERS.index(st.session_state.oya))
    tsumibo = st.number_input('積み棒本数', min_value=0, step=1, value=st.session_state.tsumibo)
    kyotaku = st.number_input('供託棒本数', min_value=0, step=1, value=st.session_state.kyotaku)
    
    results_container = st.container()
    
    if st.button('計算', type='primary'):
        st.session_state.ocr_status = None # 計算実行時にOCRメッセージをクリア
        if not ui.validate_inputs(scores, tsumibo, kyotaku):
            return
        
        st.session_state.scores = scores
        st.session_state.oya = oya
        st.session_state.tsumibo = tsumibo
        st.session_state.kyotaku = kyotaku
        
        try:
            data = calculate_conditions(scores, oya, tsumibo, kyotaku)
            
            with results_container:
                ui.display_top_difference(data['top_diff'], data['leader'])
                
                st.subheader('逆転条件（直撃ロン / 他家放銃ロン / ツモ）')
                cols = st.columns(3)
                
                for i, result in enumerate(data['results']):
                    with cols[i]:
                        ui.render_condition_card(result)
                    
        except Exception as e:
            st.error(f"計算エラーが発生しました: {str(e)}")
            st.info("入力値を確認して再度お試しください。")

if __name__ == "__main__":
    main()