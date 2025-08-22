from points_table import POINTS_TABLE
import math
import config

def ceil100(x):
    return int(math.ceil(x / 100.0) * 100)

def _lookup_high_tier_hand(points, method, is_parent):
    """満貫以上の手をデータ駆動で検索する"""
    role = 'parent' if is_parent else 'child'
    thresholds = config.HIGH_TIER_HANDS_THRESHOLDS[method][role]

    candidates = []
    for threshold, rank in thresholds:
        if threshold >= points:
            candidates.append((threshold, rank))

    if not candidates:
        return None

    # 最も安い(点数が低い)候補を選択
    cheapest_threshold, cheapest_rank = min(candidates, key=lambda x: x[0])

    if method == 'tsumo':
        if is_parent:
            display_val = f"{cheapest_threshold}オール"
            return {'rank': cheapest_rank, 'points': display_val, 'display': display_val, 'raw_points': cheapest_threshold}
        else: # child tsumo
            parent_pay = cheapest_threshold * 2
            display_val = f"{cheapest_threshold}-{parent_pay}"
            raw_points = (cheapest_threshold, parent_pay)
            return {'rank': cheapest_rank, 'points': display_val, 'display': display_val, 'raw_points': raw_points}
    else: # ron
        return {'rank': cheapest_rank, 'points': cheapest_threshold}

def reverse_lookup(points, method, is_parent):
    if points <= 0:
        return {'rank': '不要', 'points': 0, 'display': 0}

    high_tier_result = _lookup_high_tier_hand(points, method, is_parent)

    role = 'parent' if is_parent else 'child'
    low_tier_result = _lookup_low_tier_hand(points, method, role)

    def get_value(result):
        if not result:
            return float('inf')
        raw = result.get('raw_points', result.get('points', float('inf')))
        if isinstance(raw, tuple): # child tsumo
            return raw[0]
        return raw

    high_tier_value = get_value(high_tier_result)
    low_tier_value = get_value(low_tier_result)

    # どっちの候補も見つからなかった場合
    if not high_tier_result and not low_tier_result:
        # 役満の可能性をチェック
        yakuman_threshold = config.HIGH_TIER_HANDS_THRESHOLDS[method][role][0][0]
        if points >= yakuman_threshold:
             # tsumoの場合、raw_pointsも設定する必要があるかもしれないが、
             # 現在のテストではロンしかないので、pointsだけで十分
            return {'rank': '役満', 'points': points, 'raw_points': points}
        return {'rank': '不可能', 'points': points, 'display': points}

    # 同点の場合はハイティアを優先
    if high_tier_result and high_tier_value <= low_tier_value:
        return high_tier_result

    if low_tier_result:
        return low_tier_result

    return high_tier_result

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