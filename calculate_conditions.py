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
    tsumibo_points = tsumibo * 300

    is_parent = (oya == me)
    role_str = "親" if is_parent else "子"

    results = []

    # Direct Ron (from leader) - 直撃時は点差を半分に
    need_direct = math.ceil(top_diff / 2)
    need_direct = max(0, need_direct - kyotaku_points - tsumibo_points)
    need_direct = ceil100(need_direct)
    rev_direct = reverse_lookup(need_direct, 'ron', is_parent)
    results.append({
        '条件': f'直撃ロン（{leader}）（{role_str}）',
        'need_points': need_direct,
        'rank': rev_direct['rank'],
        'display': rev_direct['points'],
        'is_direct': True
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
        'is_direct': False
    })

    # Tsumo
    total_needed = top_diff - kyotaku_points - tsumibo_points
    total_needed = max(0, total_needed)
    if is_parent:
        per = math.ceil(total_needed / 3.0)
        per = ceil100(per)
        rev_t = reverse_lookup(per, 'tsumo', True)
        results.append({
            '条件': f'ツモ（{role_str}）',
            'need_points': per,
            'rank': rev_t['rank'],
            'display': rev_t['points'],
            'is_direct': False
        })
    else:
        x = math.ceil(total_needed / 4.0)
        x = ceil100(x)
        rev_t = reverse_lookup(x, 'tsumo', False)
        results.append({
            '条件': f'ツモ（{role_str}）',
            'need_points': x,
            'rank': rev_t['rank'],
            'display': rev_t['points'],
            'is_direct': False
        })

    return {'top_diff': top_diff, 'leader': leader, 'results': results}
