import streamlit as st
import numpy as np
from typing import Dict, Any
import tempfile
import os
import cv2
from PIL import Image
import config
from image_processor import ScoreImageProcessor

def process_uploaded_image(uploaded_file, shear_method: str, manual_angle: float) -> Dict[str, Any]:
    """アップロードされた画像を処理して点数とステータスを返す"""
    if 'image_processor' not in st.session_state or st.session_state.image_processor is None:
        try:
            st.session_state.image_processor = ScoreImageProcessor()
        except Exception as e:
            return {'status': 'error', 'message': f"画像処理モジュールの初期化に失敗しました: {str(e)}"}

    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("アップロードされた画像を読み込めませんでした")

        scores = st.session_state.image_processor.process_score_image(
            image, shear_correction_method=shear_method, manual_shear_angle=manual_angle
        )

        if not scores:
            return {'status': 'warning', 'message': "点数を読み取れませんでした。画像の角度や明るさを確認してください。"}

        return {'status': 'success', 'scores': scores, 'message': f"点数を読み取りました: {scores}"}

    except Exception as e:
        return {'status': 'error', 'message': f"画像処理中にエラーが発生しました: {str(e)}"}

def process_uploaded_image_full_debug(uploaded_file, shear_method: str, manual_angle: float):
    """アップロードされた画像を処理して詳細なデバッグ情報を取得"""
    if 'image_processor' not in st.session_state or st.session_state.image_processor is None:
        try:
            st.session_state.image_processor = ScoreImageProcessor()
        except Exception as e:
            st.error(f"画像処理モジュールの初期化に失敗しました: {str(e)}")
            return None

    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("アップロードされた画像を読み込めませんでした")

        return st.session_state.image_processor.get_full_debug_bundle(
            image, shear_correction_method=shear_method, manual_shear_angle=manual_angle
        )

    except Exception as e:
        st.error(f"デバッグ処理中にエラーが発生しました: {str(e)}")
        return None

def render_image_upload_section() -> Dict[str, int]:
    """画像アップロードセクションの描画"""
    st.subheader('📷 スリムスコア28S画像から自動入力')

    # Initialize session state keys
    if 'last_uploaded_file_id' not in st.session_state:
        st.session_state.last_uploaded_file_id = None
    if 'ocr_status' not in st.session_state:
        st.session_state.ocr_status = None

    # Display OCR status from previous run
    if st.session_state.ocr_status:
        status = st.session_state.ocr_status
        if status['status'] == 'success':
            st.success(status['message'])
        elif status['status'] == 'warning':
            st.warning(status['message'])
        elif status['status'] == 'error':
            st.error(status['message'])

    uploaded_file = st.file_uploader(
        "スリムスコア28Sの点数表示画像をアップロードしてください",
        type=['png', 'jpg', 'jpeg'],
        help="スマホで撮影したスリムスコア28Sの点数表示画像をアップロードすると、自動的に点数を読み取ります"
    )

    st.markdown("##### 画像補正設定")
    method_options = {
        'Hough変換': 'hough',
        '下2桁の"00"を利用': 'zeros',
        '角度を手動入力': 'manual',
        '補正なし': 'none'
    }
    method_display = st.selectbox(
        "せん断補正（傾き補正）の方法",
        options=list(method_options.keys()),
        index=1
    )
    shear_method = method_options[method_display]

    manual_angle = 0.0
    if shear_method == 'manual':
        manual_angle = st.number_input(
            "補正角度（°）",
            min_value=-45.0,
            max_value=45.0,
            value=9.0,
            step=0.1,
            help="画像の傾き角度を度数で入力します。右に傾いている場合は正の値を入力します。"
        )

    if uploaded_file is not None:
        current_config_id = f"{uploaded_file.file_id}-{shear_method}-{manual_angle}"
        last_config_id = st.session_state.get('last_uploaded_file_id')

        # Reprocess if the file or any setting has changed
        if current_config_id != last_config_id:
            uploaded_file.seek(0)
            with st.spinner('画像を処理中...'):
                result = process_uploaded_image(uploaded_file, shear_method, manual_angle)
            st.session_state.ocr_status = result
            if result['status'] == 'success':
                st.session_state.scores = result['scores']

            # Invalidate the old debug bundle as config has changed
            if 'debug_bundle' in st.session_state:
                del st.session_state['debug_bundle']

            st.session_state.last_uploaded_file_id = current_config_id

        st.image(uploaded_file, caption="アップロードされた画像", width=300)

        debug_mode = st.checkbox('🔧 デバッグモード（検出領域を表示）', value=True)

        if debug_mode:
            st.markdown("---")
            st.subheader("🛠️ デバッグ情報")

            # Regenerate debug bundle if it doesn't exist (it was deleted on config change)
            if 'debug_bundle' not in st.session_state:
                uploaded_file.seek(0)
                with st.spinner('詳細デバッグ情報を生成中...'):
                    st.session_state.debug_bundle = process_uploaded_image_full_debug(uploaded_file, shear_method, manual_angle)

            debug_bundle = st.session_state.get('debug_bundle')
            if debug_bundle:
                if 'warped_screen' in debug_bundle:
                    st.image(debug_bundle['warped_screen'], caption="1. スクリーン領域の切り出し", use_container_width=True, channels="BGR")
                if 'shear_corrected_screen' in debug_bundle:
                    angle = debug_bundle.get('shear_angles', {}).get('screen', 0)
                    st.markdown(f"##### 2. せん断補正後のスクリーン (補正角度: {angle:.2f}°)")
                    st.image(debug_bundle['shear_corrected_screen'], caption="スクリーン全体にせん断補正を適用", use_container_width=True)
                if 'deskewed_digits' in debug_bundle and debug_bundle['deskewed_digits']:
                    st.markdown("##### 3. 最終的な切り出し数字")
                    for player, digits in debug_bundle['deskewed_digits'].items():
                        st.write(f"**{player}**")
                        if not digits:
                            st.write("（数字の切り出しに失敗）")
                            continue
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

    return st.session_state.get('scores', {})

def render_score_inputs() -> Dict[str, int]:
    """点数入力UIの描画"""
    st.subheader('点数入力（百点単位）')
    image_scores = render_image_upload_section()

    st.markdown("---")
    st.markdown("**手動入力**")
    cols = st.columns(4)
    scores = {}

    for i, player in enumerate(config.PLAYERS):
        with cols[i]:
            default = (image_scores.get(player) or st.session_state.scores.get(player, 0)) // 100
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

def get_condition_style(result: Dict[str, Any]) -> tuple[str, str, str]:
    """条件に応じたスタイル設定を返す"""
    rank = result['rank']
    is_direct = result.get('is_direct', False)

    if rank == '不可能':
        return '#ffd6d6', "❌", ""
    elif rank.startswith('満貫'):
        return '#ffe566', "🌟", "bold"
    elif any(x in rank for x in ['跳満', '倍満', '三倍満', '役満']):
        return '#ffd700', "💎", "bold"
    elif is_direct:
        return '#e0f7fa', "直撃", "bold"
    else:
        return '#fff6e6', "", ""

def render_condition_card(result: Dict[str, Any]) -> None:
    """条件カードを描画"""
    bgcolor, badge, weight_class = get_condition_style(result)

    total_info = ""
    if 'total_points' in result and 'opponent_loss' in result and 'difference_points' in result:
        total_info = f"<div class='details'>合計: {result['total_points']} / 相手: {result['opponent_loss']} / 差分: {result['difference_points']}</div>"

    st.markdown(f"""
    <div class='condition-card' style='background:{bgcolor};'>
        <span class='title {weight_class}'>{badge} {result['条件']}</span><br>
        <span class='rank {weight_class}'>{result['rank']}（{result['display']}）</span>
        {total_info}
    </div>
    """, unsafe_allow_html=True)

def display_top_difference(top_diff: int, leader: str) -> None:
    """トップとの差を表示"""
    if top_diff <= 0:
        st.success("あなたは現在トップです！")
    else:
        if top_diff >= config.TOP_DIFF_COLOR_THRESHOLDS['red']:
            color = 'red'
        elif top_diff >= config.TOP_DIFF_COLOR_THRESHOLDS['orange']:
            color = 'orange'
        else:
            color = 'green'

        st.markdown(f"""
        <h2 style='color:{color};'>
            TOPとの差：<span class='top-diff-leader'>{top_diff} 点</span>（トップ: {leader}）
        </h2>
        """, unsafe_allow_html=True)

def validate_inputs(scores: Dict[str, int], tsumibo: int, kyotaku: int) -> bool:
    """入力値の検証"""
    if any(score < 0 for score in scores.values()):
        st.error("点数は0以上で入力してください")
        return False
    if tsumibo < 0 or kyotaku < 0:
        st.error("積み棒・供託棒は0以上で入力してください")
        return False
    return True
