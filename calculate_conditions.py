from points_lookup import reverse_lookup, ceil100
import math
import config

def _create_result_dict(condition_title, need_points, lookup_result, is_direct):
    """結果を格納する辞書を作成するヘルパー関数"""
    total_points = 0
    opponent_loss = 0
    difference_points = 0

    raw_points = lookup_result.get('raw_points', lookup_result.get('points', 0))

    if isinstance(raw_points, (int, float)):
        total_points = int(raw_points)
        opponent_loss = int(raw_points)
        difference_points = total_points * 2 if is_direct else total_points
    elif isinstance(raw_points, tuple): # 子ツモ
        child_pay, parent_pay = raw_points
        total_points = child_pay * 2 + parent_pay
        opponent_loss = child_pay
        difference_points = total_points + opponent_loss
    else: # 満貫以上のツモ
        total_points = lookup_result.get('total_points', 0)
        opponent_loss = lookup_result.get('opponent_loss', 0)
        difference_points = total_points + opponent_loss

    # 'display'が数値の場合の合計点数などを再計算
    if 'display' in lookup_result and isinstance(lookup_result['display'], (int, float)):
         total_points = int(lookup_result['display'])
         opponent_loss = int(lookup_result['display'])
         difference_points = total_points * 2 if is_direct else total_points
    
    # 親ツモの特殊処理
    if 'オール' in str(lookup_result.get('display', '')):
        per_person_actual = lookup_result.get('raw_points', 0)
        total_points = per_person_actual * 3
        opponent_loss = per_person_actual
        difference_points = total_points + opponent_loss

    return {
        '条件': condition_title,
        'need_points': need_points,
        'rank': lookup_result['rank'],
        'display': lookup_result.get('display', lookup_result['points']),
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
    return _create_result_dict(title, needed_score, lookup_result, is_direct=False)

def _calculate_tsumo(top_diff, is_parent, tsumibo, kyotaku, role_str):
    """ツモ和了の条件を計算"""
    kyotaku_points = kyotaku * config.POINTS_PER_KYOTAKU
    tsumo_tsumibo_points = tsumibo * config.POINTS_PER_TSUMIBO_TSUMO # 1本場につき100点ずつ増える

    if is_parent:
        # 親ツモ：全員から支払い
        needed_per_person = ceil100((top_diff - kyotaku_points) / 3)
        needed_per_person -= tsumo_tsumibo_points
        needed_per_person = max(0, needed_per_person)
        lookup_result = reverse_lookup(needed_per_person, 'tsumo', True)
    else:
        # 子ツモ：親は子の倍払う
        needed_child_pay = ceil100((top_diff - kyotaku_points) / 4)
        needed_child_pay -= tsumo_tsumibo_points
        needed_child_pay = max(0, needed_child_pay)
        lookup_result = reverse_lookup(needed_child_pay, 'tsumo', False)

    needed_score = needed_per_person if is_parent else needed_child_pay
    title = f'ツモ（{role_str}）'
    result = _create_result_dict(title, needed_score, lookup_result, is_direct=False)

    # ツモの場合の合計点を再計算
    if is_parent:
        per_person_actual = lookup_result.get('raw_points', 0)
        result['total_points'] = per_person_actual * 3
        result['opponent_loss'] = per_person_actual
    else:
        child_pay, parent_pay = lookup_result.get('raw_points', (0,0))
        result['total_points'] = child_pay * 2 + parent_pay
        result['opponent_loss'] = f"{child_pay} / {parent_pay}"

    return result

def calculate_conditions(scores, oya, tsumibo, kyotaku):
    me = '自分'
    if me not in scores:
        raise ValueError('scores must include "自分"')

    leader = max(scores, key=lambda k: (scores[k], k))
    my_score = scores[me]
    top_diff = scores[leader] - my_score + 1

    is_parent = (oya == me)
    role_str = "親" if is_parent else "子"

    results = [
        _calculate_direct_ron(top_diff, is_parent, tsumibo, kyotaku, leader, role_str),
        _calculate_other_ron(top_diff, is_parent, tsumibo, kyotaku, role_str),
        _calculate_tsumo(top_diff, is_parent, tsumibo, kyotaku, role_str)
    ]

    return {'top_diff': top_diff, 'leader': leader, 'results': results}
