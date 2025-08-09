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

    results = []

    # Direct Ron (from leader)
    need = top_diff - kyotaku_points - tsumibo_points
    need = max(0, need)
    need = ceil100(need)
    rev = reverse_lookup(need, 'ron', is_parent)
    results.append({'条件': f'直撃ロン（{leader}）', 'need_points': need, 'rank': rev['rank'], 'display': rev['points']})

    # Other Ron (no name)
    need_o = top_diff - kyotaku_points - tsumibo_points
    need_o = max(0, need_o)
    need_o = ceil100(need_o)
    rev_o = reverse_lookup(need_o, 'ron', is_parent)
    results.append({'条件': '他家放銃ロン', 'need_points': need_o, 'rank': rev_o['rank'], 'display': rev_o['points']})

    # Tsumo
    total_needed = top_diff - kyotaku_points - tsumibo_points
    total_needed = max(0, total_needed)
    if is_parent:
        per = math.ceil(total_needed / 3.0)
        per = ceil100(per)
        rev_t = reverse_lookup(per, 'tsumo', True)
        results.append({'条件': 'ツモ（親）', 'need_points': per, 'rank': rev_t['rank'], 'display': rev_t['points']})
    else:
        x = math.ceil(total_needed / 4.0)
        x = ceil100(x)
        rev_t = reverse_lookup(x, 'tsumo', False)
        results.append({'条件': 'ツモ（子）', 'need_points': x, 'rank': rev_t['rank'], 'display': rev_t['points']})

    return {'top_diff': top_diff, 'leader': leader, 'results': results}
