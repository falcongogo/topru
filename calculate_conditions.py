from points_lookup import reverse_lookup, ceil100
import math

def calculate_conditions(scores, oya, tsumibo, kyotaku):
    me = '自分'
    if me not in scores:
        raise ValueError('scores must include "自分"')

    leader = max(scores, key=lambda k: (scores[k], k))
    leader_score = scores[leader]
    my_score = scores[me]
    top_diff = leader_score - my_score + 1

    kyotaku_points = kyotaku * 1000
    tsumibo_points = tsumibo * 300 * 2
    tsumo_tsumibo_points = tsumibo * 400

    is_parent = (oya == me)
    role_str = "親" if is_parent else "子"

    # 計算詳細を格納する辞書
    calculation_details = {
        'current_scores': scores.copy(),
        'leader': leader,
        'leader_score': leader_score,
        'my_score': my_score,
        'top_diff': top_diff,
        'kyotaku_points': kyotaku_points,
        'tsumibo_points': tsumibo_points,
        'tsumo_tsumibo_points': tsumo_tsumibo_points,
        'is_parent': is_parent,
        'role_str': role_str
    }

    results = []

    # Direct Ron (from leader) - 直撃時は点差を半分に
    need_direct = top_diff - kyotaku_points - tsumibo_points
    need_direct = max(0, need_direct)
    need_direct_original = need_direct
    need_direct = need_direct / 2
    need_direct = ceil100(need_direct)
    rev_direct = reverse_lookup(need_direct, 'ron', is_parent)
    results.append({
        '条件': f'直撃ロン（{leader}）（{role_str}）',
        'need_points': need_direct,
        'need_points_original': need_direct_original,
        'rank': rev_direct['rank'],
        'display': rev_direct['points'],
        'is_direct': True,
        'calculation_steps': {
            'top_diff': top_diff,
            'minus_kyotaku': top_diff - kyotaku_points,
            'minus_tsumibo': top_diff - kyotaku_points - tsumibo_points,
            'divided_by_2': need_direct_original / 2,
            'final_points': need_direct
        }
    })

    # Other Ron (no name)
    need_other = top_diff - kyotaku_points - tsumibo_points
    need_other = max(0, need_other)
    need_other = ceil100(need_other)
    rev_other = reverse_lookup(need_other, 'ron', is_parent)
    results.append({
        '条件': f'他家放銃ロン（{role_str}）',
        'need_points': need_other,
        'rank': rev_other['rank'],
        'display': rev_other['points'],
        'is_direct': False,
        'calculation_steps': {
            'top_diff': top_diff,
            'minus_kyotaku': top_diff - kyotaku_points,
            'minus_tsumibo': top_diff - kyotaku_points - tsumibo_points,
            'final_points': need_other
        }
    })

    # Tsumo
    total_needed = top_diff - kyotaku_points - tsumo_tsumibo_points
    total_needed = max(0, total_needed)
    if is_parent:
        # 親ツモ：3人から同じ点数をもらう
        per_person = math.ceil(total_needed / 3.0)
        per_person = ceil100(per_person)
        rev_t = reverse_lookup(per_person, 'tsumo', True)
        results.append({
            '条件': f'ツモ（{role_str}）',
            'need_points': per_person,
            'total_needed': total_needed,
            'rank': rev_t['rank'],
            'display': rev_t['points'],
            'is_direct': False,
            'calculation_steps': {
                'top_diff': top_diff,
                'minus_kyotaku': top_diff - kyotaku_points,
                'minus_tsumo_tsumibo': top_diff - kyotaku_points - tsumo_tsumibo_points,
                'divided_by_3': total_needed / 3.0,
                'ceiled': math.ceil(total_needed / 3.0),
                'final_points': per_person
            }
        })
    else:
        # 子ツモ：親から2倍、子から1倍の点数をもらう
        # 親2人 + 子1人 = 合計3人から点数をもらう
        per_child = math.ceil(total_needed / 3.0)
        per_child = ceil100(per_child)
        rev_t = reverse_lookup(per_child, 'tsumo', False)
        results.append({
            '条件': f'ツモ（{role_str}）',
            'need_points': per_child,
            'total_needed': total_needed,
            'rank': rev_t['rank'],
            'display': rev_t['points'],
            'is_direct': False,
            'calculation_steps': {
                'top_diff': top_diff,
                'minus_kyotaku': top_diff - kyotaku_points,
                'minus_tsumo_tsumibo': top_diff - kyotaku_points - tsumo_tsumibo_points,
                'divided_by_3': total_needed / 3.0,
                'ceiled': math.ceil(total_needed / 3.0),
                'final_points': per_child
            }
        })

    return {
        'top_diff': top_diff, 
        'leader': leader, 
        'results': results,
        'calculation_details': calculation_details
    }
