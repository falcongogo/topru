import streamlit as st
import numpy as np
from calculate_conditions import calculate_conditions
from image_processor import ScoreImageProcessor
from typing import Dict, Any
import tempfile
import os
import cv2
from PIL import Image

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
        st.session_state.image_processor = None

def initialize_image_processor():
    """画像処理クラスの初期化"""
    if st.session_state.image_processor is None:
        try:
            st.session_state.image_processor = ScoreImageProcessor()
        except Exception as e:
            st.error(f"画像処理モジュールの初期化に失敗しました: {str(e)}")
            return False
    return True

def process_uploaded_image(uploaded_file) -> Dict[str, int]:
    """アップロードされた画像を処理して点数を取得"""
    if not initialize_image_processor():
        return {}
    
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # 画像処理
        scores = st.session_state.image_processor.process_score_image(tmp_path)
        
        # 一時ファイルを削除
        os.unlink(tmp_path)
        
        return scores
        
    except Exception as e:
        st.error(f"画像処理中にエラーが発生しました: {str(e)}")
        return {}

def process_uploaded_image_full_debug(uploaded_file):
    """アップロードされた画像を処理して詳細なデバッグ情報を取得"""
    if not initialize_image_processor():
        return None
    
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("アップロードされた画像を読み込めませんでした")
        
        return st.session_state.image_processor.get_full_debug_bundle(image)
        
    except Exception as e:
        st.error(f"デバッグ処理中にエラーが発生しました: {str(e)}")
        return None

def render_image_upload_section() -> Dict[str, int]:
    """画像アップロードセクションの描画"""
    st.subheader('📷 スリムスコア28S画像から自動入力')
    
    # セッションステートの初期化
    if 'last_uploaded_file_id' not in st.session_state:
        st.session_state.last_uploaded_file_id = None

    uploaded_file = st.file_uploader(
        "スリムスコア28Sの点数表示画像をアップロードしてください",
        type=['png', 'jpg', 'jpeg'],
        help="スマホで撮影したスリムスコア28Sの点数表示画像をアップロードすると、自動的に点数を読み取ります"
    )
    
    if uploaded_file is not None:
        # 新しいファイルがアップロードされた場合、自動でOCRを実行
        if uploaded_file.file_id != st.session_state.last_uploaded_file_id:
            with st.spinner('画像を処理中...'):
                scores = process_uploaded_image(uploaded_file)

            if scores:
                st.success(f"点数を読み取りました: {scores}")
                st.session_state.scores = scores
            else:
                st.warning("点数を読み取れませんでした。画像の角度や明るさを確認してください。")

            # 処理済みのファイルIDを保存し、画面を再実行してUIに反映
            st.session_state.last_uploaded_file_id = uploaded_file.file_id
            st.rerun()

        # --- ここから下の部分は、ファイルがアップロードされている場合に常に表示 ---

        # 画像プレビュー
        st.image(uploaded_file, caption="アップロードされた画像", width=300)
        
        # デバッグモードの切り替え
        debug_mode = st.checkbox('🔧 デバッグモード（検出領域を表示）', value=True)
        
        # デバッグモードの場合、詳細な途中経過を自動表示
        if debug_mode:
            st.markdown("---")
            st.subheader("🛠️ デバッグ情報")

            # セッションステートにデバッグ情報がなければ生成
            if 'debug_bundle' not in st.session_state or st.session_state.last_uploaded_file_id != uploaded_file.file_id:
                uploaded_file.seek(0)
                with st.spinner('詳細デバッグ情報を生成中...'):
                    st.session_state.debug_bundle = process_uploaded_image_full_debug(uploaded_file)

            debug_bundle = st.session_state.get('debug_bundle')

            if debug_bundle:
                # 処理の経過を順に表示
                if 'main_frame' in debug_bundle:
                    st.image(debug_bundle['main_frame'], caption="1. メインフレーム検出", use_container_width=True, channels="BGR")

                if 'warped_screen' in debug_bundle:
                    st.image(debug_bundle['warped_screen'], caption="2. 傾き補正後のスクリーン", use_container_width=True, channels="BGR")

                if 'shear_corrected_screen' in debug_bundle:
                    st.markdown("##### 3. せん断補正後のスクリーン全体")
                    st.image(debug_bundle['shear_corrected_screen'], caption="スクリーン全体にせん断補正を適用 (9°で固定)", use_container_width=True)

                if 'deskewed_digits' in debug_bundle and debug_bundle['deskewed_digits']:
                    st.markdown("##### 4. 最終的な切り出し数字")
                    for player, digits in debug_bundle['deskewed_digits'].items():
                        st.write(f"**{player}**")
                        if not digits:
                            st.write("（数字の切り出しに失敗）")
                            continue

                        # PILに変換して結合
                        pil_images = [Image.fromarray(d) for d in digits if d is not None and d.size > 0]
                        if pil_images:
                            widths, heights = zip(*(i.size for i in pil_images))
                            total_width = sum(widths)
                            max_height = max(heights)

                            concatenated_image = Image.new('L', (total_width, max_height))
                            x_offset = 0
                            for im in pil_images:
                                concatenated_image.paste(im, (x_offset,0))
                                x_offset += im.size[0]
                            st.image(concatenated_image, caption=f"{player} 切り出し後", use_container_width=True)
                else:
                    st.warning("傾き補正後のデバッグ画像を生成できませんでした。")
            else:
                st.warning("デバッグ情報を生成できませんでした。")
    
    # セッションステートからスコアを返す
    return st.session_state.get('scores', {})

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
    
    # 画像アップロードセクション
    image_scores = render_image_upload_section()
    
    # 手動入力セクション
    st.markdown("---")
    st.markdown("**手動入力**")
    cols = st.columns(4)
    scores = {}
    
    for i, player in enumerate(PLAYERS):
        with cols[i]:
            # 画像から読み取った点数があれば使用、なければ現在のセッション状態から
            if image_scores and player in image_scores:
                default = image_scores[player] // 100
            else:
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
    
    # 合計点数、相手のマイナス点数、差分点数を表示
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
    # lang属性をjaに設定
    st.markdown(
        """
        <script>
            document.documentElement.lang = 'ja';
        </script>
        """,
        unsafe_allow_html=True,
    )
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