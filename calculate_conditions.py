"""
麻雀の逆転条件を計算するためのコアロジック。

このモジュールは、現在の点数状況に基づいて、麻雀のオーラスで
逆転するために必要な点数を決定する責務を持ちます。
直撃ロン、他家からのロン、ツモの3つの主要なシナリオで条件を計算します。
"""
from points_lookup import reverse_lookup, ceil100
import math

def calculate_conditions(scores, oya, tsumibo, kyotaku):
    """
    ユーザー（'自分'）の逆転条件を計算します。

    この関数は、トップのプレイヤーを追い越すために必要な手役の価値を、
    以下の3つの異なるシナリオで計算します。
    1. 直撃ロン: 現在のトップからロン和了する。
    2. 他家ロン: トップ以外のプレイヤーからロン和了する。
    3. ツモ: 自力で和了牌を引く（ツモ）。

    Args:
        scores (dict): 4人全員の点数を含む辞書。
        oya (str): 親であるプレイヤーの名前。
        tsumibo (int): 積み棒の本数。
        kyotaku (int): 供託棒の本数。

    Returns:
        dict: トップとの点差（'top_diff'）、トップのプレイヤー名（'leader'）、
              そして各シナリオの結果辞書のリスト（'results'）を含む辞書。
    """
    me = '自分'
    if me not in scores:
        raise ValueError('scores must include "自分"')

    leader = max(scores, key=lambda k: (scores[k], k))
    leader_score = scores[leader]
    my_score = scores[me]
    top_diff = leader_score - my_score + 1

    kyotaku_points = kyotaku * 1000
    ron_tsumibo_bonus = tsumibo * 300  # ロン和了では1本場につき300点。
    tsumo_bonus = tsumibo * 300 # ツモ和了では各プレイヤーから1本場につき100点（合計300点）。

    is_parent = (oya == me)
    role_str = "親" if is_parent else "子"

    results = []

    # --- 1. 直撃ロン（トップから） ---
    # トップが支払うため、点差は倍縮まる。
    need_direct = (top_diff - (kyotaku_points + ron_tsumibo_bonus)) / 2
    need_direct = ceil100(need_direct)
    rev_direct = reverse_lookup(need_direct, 'ron', is_parent)

    if isinstance(rev_direct['points'], int):
        total_points = rev_direct['points'] + kyotaku_points + ron_tsumibo_bonus
        opponent_loss = rev_direct['points']
        difference_points = rev_direct['points'] * 2
    else: # 満貫など
        total_points = rev_direct['points']
        opponent_loss = rev_direct['points']
        difference_points = rev_direct['points']

    results.append({
        '条件': f'直撃ロン（{leader}）',
        'need_points': need_direct,
        'rank': rev_direct['rank'],
        'display': rev_direct['points'],
        'total_points': total_points,
        'opponent_loss': opponent_loss,
        'difference_points': difference_points,
        'is_direct': True
    })

    # --- 2. 他家からのロン（トップ以外から） ---
    need_other = top_diff - (kyotaku_points + ron_tsumibo_bonus)
    need_other = ceil100(need_other)
    rev_other = reverse_lookup(need_other, 'ron', is_parent)
    
    if isinstance(rev_other['points'], int):
        total_points = rev_other['points'] + kyotaku_points + ron_tsumibo_bonus
        opponent_loss = rev_other['points']
        difference_points = rev_other['points']
    else: # 満貫など
        total_points = rev_other['points']
        opponent_loss = rev_other['points']
        difference_points = rev_other['points']

    results.append({
        '条件': f'他家放銃ロン',
        'need_points': need_other,
        'rank': rev_other['rank'],
        'display': rev_other['points'],
        'total_points': total_points,
        'opponent_loss': opponent_loss,
        'difference_points': difference_points,
        'is_direct': False
    })

    # --- 3. ツモ ---
    total_needed = top_diff - (kyotaku_points + tsumo_bonus)
    
    if is_parent:
        # 親のツモ: 各相手が総価値の1/3を支払う。
        per_person_needed = ceil100(total_needed / 3.0)
        rev_t = reverse_lookup(per_person_needed, 'tsumo', True)
        
        if 'オール' in str(rev_t['points']):
            per_person_actual = int(str(rev_t['points']).replace('オール', ''))
            total_points = per_person_actual * 3
        else:
            per_person_actual = rev_t['points']
            total_points = per_person_actual * 3
            
        opponent_loss = per_person_actual
        difference_points = total_points + opponent_loss # 近似値
    else:
        # 子のツモ: 親は他の子の約2倍を支払う。
        # 他の子からの支払いに基づいて検索。
        child_payment_needed = ceil100(total_needed / 4.0)
        rev_t = reverse_lookup(child_payment_needed, 'tsumo', False)

        if isinstance(rev_t['points'], str) and '-' in rev_t['points']:
            child_pay, parent_pay = map(int, rev_t['points'].split('-'))
            total_points = child_pay * 2 + parent_pay
            opponent_loss = f"子{child_pay}, 親{parent_pay}"
        else:
            child_pay = rev_t['points']
            parent_pay = child_pay * 2
            total_points = child_pay * 2 + parent_pay
            opponent_loss = f"子{child_pay}, 親{parent_pay}"
            
        difference_points = total_points # 近似値

    total_points_actual = total_points + kyotaku_points + tsumo_bonus

    results.append({
        '条件': f'ツモ',
        'need_points': total_needed,
        'rank': rev_t['rank'],
        'display': rev_t['points'],
        'total_points': total_points_actual,
        'opponent_loss': opponent_loss,
        'difference_points': difference_points,
        'is_direct': False
    })

    return {'top_diff': top_diff, 'leader': leader, 'results': results}
