from points_table import POINTS_TABLE
import math
import config

def ceil100(x):
    return int(math.ceil(x / 100.0) * 100)

def _lookup_high_tier_hand(points, method, is_parent):
    """満貫以上の手をデータ駆動で検索する"""
    role = 'parent' if is_parent else 'child'
    thresholds = config.HIGH_TIER_HANDS_THRESHOLDS[method][role]

    for threshold, rank in thresholds:
        if points >= threshold:
            if method == 'tsumo':
                if is_parent:
                    display_val = f"{points}オール"
                    return {'rank': rank, 'points': display_val, 'display': display_val, 'raw_points': points}
                else: # child tsumo
                    parent_pay = points * 2
                    display_val = f"{points}-{parent_pay}"
                    raw_points = (points, parent_pay)
                    return {'rank': rank, 'points': display_val, 'display': display_val, 'raw_points': raw_points}
            else: # ron
                return {'rank': rank, 'points': points}
    return None

def reverse_lookup(points, method, is_parent):
    if points <= 0:
        return {'rank': '不要', 'points': 0, 'display': 0}

    # 満貫以上をチェック
    high_tier_result = _lookup_high_tier_hand(points, method, is_parent)
    if high_tier_result:
        return high_tier_result

    role = 'parent' if is_parent else 'child'

    # 満貫未満を探索
    low_tier_result = _lookup_low_tier_hand(points, method, role)
    if low_tier_result:
        return low_tier_result

    return {'rank': '不可能', 'points': points, 'display': points}

def _format_low_tier_result(han, fu, val, method, role):
    """満貫未満の結果辞書をフォーマットする"""
    rank = f"{fu}符{han}翻"
    if method == 'ron':
        return {'rank': rank, 'points': val, 'display': val}
    else: # tsumo
        if role == 'parent':
            display_val = f"{val}オール"
            return {'rank': rank, 'points': display_val, 'display': display_val, 'raw_points': val}
        else: # child
            child_pay, parent_pay = val
            display_val = f"{child_pay}-{parent_pay}"
            return {'rank': rank, 'points': display_val, 'display': display_val, 'raw_points': val}

def _lookup_low_tier_hand(points, method, role):
    """満貫未満の手を点数表から検索する"""
    table = POINTS_TABLE[role][method]

    candidates = []
    for (fu, han), val in table.items():
        if fu > 50 or han > 4:
            continue
        if method == 'ron' and fu == 20:
            continue  # ロンで20符は不可

        # 条件を満たすかチェック
        meets_condition = False
        if method == 'ron':
            if val >= points:
                meets_condition = True
        else: # tsumo
            if role == 'parent':
                if val >= points:
                    meets_condition = True
            else: # child tsumo
                child_pay, _ = val
                if child_pay >= points:
                    meets_condition = True

        if meets_condition:
            candidates.append((han, fu, val))

    if candidates:
        # 最小の翻数、符数、点数の手を選択
        candidates.sort(key=lambda x: (x[0], x[1], x[2] if isinstance(x[2], int) else x[2][0]))
        han, fu, val = candidates[0]
        return _format_low_tier_result(han, fu, val, method, role)

    return None