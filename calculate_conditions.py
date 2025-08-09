from points_lookup import reverse_lookup
import math

def ceil100(x):
    return int(math.ceil(x / 100.0) * 100)

def calculate_conditions(scores, oya, tsumibo, kyotaku):
    '''
    scores: dict of full points (e.g., 25000)
    oya: string name of dealer ('自分' etc.)
    tsumibo: int
    kyotaku: int
    returns list of dicts for conditions
    '''
    me = '自分'
    if me not in scores:
        raise ValueError('scores must include "自分" key')
    # find leader (if tie, highest by order will be leader but same value tie treated as leader)
    leader = max(scores, key=lambda k: (scores[k], k))
    if leader == me:
        return [{'条件': 'すでにトップ', 'rank': '不要', 'points': 0, 'detail': '既にトップです'}]

    leader_score = scores[leader]
    my_score = scores[me]

    # required difference to be strictly greater (same score => lose due to 上家優位)
    diff = leader_score - my_score + 1

    # kyotaku (1000 per) and tsumibo (300 per) are awarded to winner; subtract these from required gain
    kyotaku_points = kyotaku * 1000
    tsumibo_points = tsumibo * 300

    results = []

    is_parent = (oya == me)

    # --- 直撃ロン (ron from leader) ---
    need = diff - kyotaku_points - tsumibo_points
    need = max(0, need)
    need = ceil100(need)
    rev = reverse_lookup(need, 'ron', is_parent)
    results.append({'条件': f'直撃ロン（{leader}）', 'rank': rev['rank'], 'points': rev['points'], 'detail': f'差:{diff} 積:{tsumibo_points} 供:{kyotaku_points}'})

    # --- 他家放銃ロン (ron from anyone else) ---
    # We consider the minimal other opponent (choose one that yields minimal required ron)
    others = [p for p in scores if p != me and p != leader]
    best_other = None
    best = None
    for other in others:
        # If other pays, your score increases by ron points; leader remains same
        need_o = diff - kyotaku_points - tsumibo_points
        need_o = max(0, need_o)
        need_o = ceil100(need_o)
        rev_o = reverse_lookup(need_o, 'ron', is_parent)
        # pick by numeric points if possible (rev_o['points'] may be string for tsumo-like), coerce to int when possible
        try:
            val = int(str(rev_o['points']).split('-')[0])
        except:
            val = need_o
        if best is None or val < best:
            best = val
            best_other = (other, rev_o, need_o)
    if best_other:
        other_name, rev_o, need_o = best_other
        results.append({'条件': f'他家放銃ロン（{other_name}）', 'rank': rev_o['rank'], 'points': rev_o['points'], 'detail': f'差:{diff} 積:{tsumibo_points} 供:{kyotaku_points}'})

    # --- ツモ ---
    # total needed gain = diff - kyotaku - tsumibo
    total_needed = diff - kyotaku_points - tsumibo_points
    total_needed = max(0, total_needed)
    if is_parent:
        # parent: per-person pay = total_needed / 3
        per = math.ceil(total_needed / 3.0)
        per = ceil100(per)
        rev_t = reverse_lookup(per, 'tsumo', True)
        results.append({'条件': 'ツモ（親）', 'rank': rev_t['rank'], 'points': rev_t['points'], 'detail': f'総必要:{total_needed} per:{per} (オール)'})
    else:
        # child: unit x such that total = 4*x (parent pays 2x, two others pay x)
        x = math.ceil(total_needed / 4.0)
        x = ceil100(x)
        parent_pay = x * 2
        # display as child-parent
        rev_t = reverse_lookup(x, 'tsumo', False)
        results.append({'条件': 'ツモ（子）', 'rank': rev_t['rank'], 'points': rev_t['points'], 'detail': f'総必要:{total_needed} 子:{x} 親:{parent_pay}'})

    return results
