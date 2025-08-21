from points_lookup import reverse_lookup, ceil100
import math
import config

def _create_result_dict(condition_title, need_points, lookup_result, is_direct):
    """結果を格納する辞書を作成するヘルパー関数（ロン専用に簡略化）"""
    points = lookup_result.get('points', 0)
    total_points = points
    opponent_loss = points
    difference_points = total_points * 2 if is_direct else total_points

    return {
        '条件': condition_title,
        'need_points': need_points,
        'rank': lookup_result['rank'],
        'display': lookup_result.get('display', points),
        'total_points': total_points,
        'opponent_loss': opponent_loss,
        'difference_points': difference_points,
        'is_direct': is_direct
    }

def _calculate_direct_ron(top_diff, is_parent, tsumibo, kyotaku, leader_name, role_str):
    """直撃ロンの条件を計算"""
    kyotaku_points = kyotaku * config.POINTS_PER_KYOTAKU
    tsumibo_points = tsumibo * config.POINTS_PER_TSUMIBO_RON
    
    needed_score = ceil100((top_diff - kyotaku_points - tsumibo_points) / 2)
    needed_score = max(0, needed_score)
    
    lookup_result = reverse_lookup(needed_score, 'ron', is_parent)
    
    title = f'直撃ロン（{leader_name}）（{role_str}）'
    return _create_result_dict(title, needed_score, lookup_result, is_direct=True)

def _calculate_other_ron(top_diff, is_parent, tsumibo, kyotaku, role_str):
    """他家ロンの条件を計算"""
    kyotaku_points = kyotaku * config.POINTS_PER_KYOTAKU
    tsumibo_points = tsumibo * config.POINTS_PER_TSUMIBO_RON
    
    needed_score = top_diff - kyotaku_points - tsumibo_points
    needed_score = ceil100(max(0, needed_score))
    
    lookup_result = reverse_lookup(needed_score, 'ron', is_parent)

    title = f'他家放銃ロン（{role_str}）'
    result = _create_result_dict(title, needed_score, lookup_result, is_direct=False)

    # 他家ロンの場合、トップの失点は0
    result['opponent_loss'] = 0

    return result

def _calculate_tsumo(top_diff, is_parent, tsumibo, kyotaku, role_str, top_is_parent):
    """ツモ和了の条件を計算"""
    kyotaku_points = kyotaku * config.POINTS_PER_KYOTAKU
    tsumibo_swing = tsumibo * config.POINTS_PER_TSUMIBO_TSUMO * 4

    # P > (top_diff - 1 - kyotaku_points - tsumibo_swing) / divisor
    # smallest integer P is math.floor(numerator / divisor) + 1
    numerator = top_diff - 1 - kyotaku_points - tsumibo_swing

    if is_parent:
        # 親ツモ：トップは必ず子。点差は4P(自分の収入3P + トップの失点1P)で縮まる
        divisor = 4
        needed_score = math.floor(numerator / divisor) + 1
        needed_score = max(0, needed_score)
        lookup_result = reverse_lookup(needed_score, 'tsumo', True)
    else:
        # 子ツモ
        if top_is_parent:
            # トップが親。点差は6P(自分の収入4P + トップの失点2P)で縮まる
            divisor = 6
        else:
            # トップも子。点差は5P(自分の収入4P + トップの失点1P)で縮まる
            divisor = 5

        needed_score = math.floor(numerator / divisor) + 1
        needed_score = max(0, needed_score)
        lookup_result = reverse_lookup(needed_score, 'tsumo', False)

    title = f'ツモ（{role_str}）'

    result = {
        '条件': title,
        'need_points': needed_score,
        'rank': lookup_result['rank'],
        'display': lookup_result.get('display', lookup_result['points']),
        'is_direct': False
    }

    # ツモ和了の得点情報を計算して格納
    if is_parent:
        per_person_actual = lookup_result.get('raw_points', 0)
        result['total_points'] = per_person_actual * 3
        result['opponent_loss'] = per_person_actual # トップは子なので、失点は子の支払い分
    else:
        child_pay, parent_pay = lookup_result.get('raw_points', (0,0))
        result['total_points'] = child_pay * 2 + parent_pay
        if top_is_parent:
            result['opponent_loss'] = parent_pay # トップは親なので、失点は親の支払い分
        else:
            result['opponent_loss'] = child_pay # トップは子なので、失点は子の支払い分

    result['difference_points'] = result['total_points'] + result['opponent_loss']

    return result

def calculate_conditions(scores, oya, tsumibo, kyotaku):
    me = '自分'
    if me not in scores:
        raise ValueError('scores must include "自分"')

    leader = max(scores, key=lambda k: (scores[k], k))
    my_score = scores[me]
    top_diff = scores[leader] - my_score + 1

    is_parent = (oya == me)
    top_is_parent = (oya == leader)
    role_str = "親" if is_parent else "子"

    results = [
        _calculate_direct_ron(top_diff, is_parent, tsumibo, kyotaku, leader, role_str),
        _calculate_other_ron(top_diff, is_parent, tsumibo, kyotaku, role_str),
        _calculate_tsumo(top_diff, is_parent, tsumibo, kyotaku, role_str, top_is_parent)
    ]

    return {'top_diff': top_diff, 'leader': leader, 'results': results}
