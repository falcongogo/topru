"""
麻雀の手役の点数逆引き機能。

このモジュールは、特定のスコアを達成するために最低限必要な手役（符と翻、
または満貫以上のランク）を見つける機能を提供します。
これはアプリケーションの「逆引き」機能の中核です。
"""
from points_table import POINTS_TABLE
import math

# --- 満貫の閾値に関する定数 ---
CHILD_RON_MANGAN = 8000
PARENT_RON_MANGAN = 12000
PARENT_TSUMO_MANGAN = 4000
CHILD_TSUMO_MANGAN = 2000

def ceil100(x: float) -> int:
    """
    数値を最も近い100の倍数に切り上げます。

    Args:
        x: 切り上げる数値。

    Returns:
        入力された数値を次の100の倍数に切り上げた整数。
    """
    return int(math.ceil(x / 100.0) * 100)

def reverse_lookup(points: int, method: str, is_parent: bool) -> dict:
    """
    指定された点数を達成するために最低限必要な手役を見つけます。

    この関数は「逆引き」を行い、目標の`points`以上の価値がある最小の手を
    見つけ出します。まず満貫以上の手をチェックし、必要な点数がそれより低い
    場合は、`POINTS_TABLE`から最適な符と翻の組み合わせを検索します。

    Args:
        points: 手役が持つべき最低点数。ツモの場合、これはプレイヤー一人あたり
                （または子のツモの場合は子一人あたり）の支払額です。
        method: 和了の方法。「ron」または「tsumo」。
        is_parent: プレイヤーが親である場合はTrue、そうでない場合はFalse。

    Returns:
        'rank'（例：「30符4翻」、「満貫」）と、その手の実際の'points'の値または
        表示文字列を含む辞書。points <= 0の場合は「不要」、適切な手が見つから
        ない場合は「不可能」を返します。
    """
    if points <= 0:
        return {'rank': '不要', 'points': 0}

    # --- 満貫以上 ---
    # 高得点の手には、符/翻の代わりにランク名を返す。
    mangan_value = PARENT_RON_MANGAN if is_parent else CHILD_RON_MANGAN
    if method == 'ron' and points >= mangan_value:
        yakuman_val = 48000 if is_parent else 32000
        if points >= yakuman_val: return {'rank': '役満', 'points': yakuman_val}
        if points >= yakuman_val * 0.75: return {'rank': '三倍満', 'points': int(yakuman_val * 0.75)}
        if points >= yakuman_val * 0.5: return {'rank': '倍満', 'points': int(yakuman_val * 0.5)}
        if points >= yakuman_val * 0.375: return {'rank': '跳満', 'points': int(yakuman_val * 0.375)}
        return {'rank': '満貫', 'points': mangan_value}

    mangan_tsumo_value = PARENT_TSUMO_MANGAN if is_parent else CHILD_TSUMO_MANGAN
    if method == 'tsumo' and points >= mangan_tsumo_value:
        yakuman_val = 16000 if is_parent else 8000
        if points >= yakuman_val: return {'rank': '役満', 'points': f"{yakuman_val}オール" if is_parent else f"{yakuman_val}-{yakuman_val*2}"}
        if points >= yakuman_val * 0.75: return {'rank': '三倍満', 'points': f"{int(yakuman_val*0.75)}オール" if is_parent else f"{int(yakuman_val*0.75)}-{int(yakuman_val*0.75)*2}"}
        if points >= yakuman_val * 0.5: return {'rank': '倍満', 'points': f"{int(yakuman_val*0.5)}オール" if is_parent else f"{int(yakuman_val*0.5)}-{int(yakuman_val*0.5)*2}"}
        if points >= yakuman_val * 0.375: return {'rank': '跳満', 'points': f"{int(yakuman_val*0.375)}オール" if is_parent else f"{int(yakuman_val*0.375)}-{int(yakuman_val*0.375)*2}"}
        return {'rank': '満貫', 'points': f"{mangan_tsumo_value}オール" if is_parent else f"{mangan_tsumo_value}-{mangan_tsumo_value*2}"}

    # --- 満貫未満 ---
    # 点数表から最適な候補を検索する。
    role = 'parent' if is_parent else 'child'
    table = POINTS_TABLE[role][method]

    candidates = []
    for (fu, han), val in table.items():
        # ルール：ロンで20符はなし、この表では50符/4翻まで
        if fu > 50 or han > 4: continue
        if method == 'ron' and fu == 20: continue

        required_val = val if method == 'ron' or is_parent else val[0]
        if required_val >= points:
            candidates.append((han, fu, val))

    if not candidates:
        return {'rank': '不可能', 'points': points}

    # 最適な候補を見つける（翻が最も低く、次に符が低く、次に点数が低い）
    candidates.sort(key=lambda x: (x[0], x[1], x[2] if isinstance(x[2], int) else x[2][0]))
    han, fu, val = candidates[0]

    if method == 'ron':
        return {'rank': f"{fu}符{han}翻", 'points': val}
    else: # tsumo
        if is_parent:
            return {'rank': f"{fu}符{han}翻", 'points': f"{val}オール"}
        else:
            child_pay, parent_pay = val
            return {'rank': f"{fu}符{han}翻", 'points': f"{child_pay}-{parent_pay}"}