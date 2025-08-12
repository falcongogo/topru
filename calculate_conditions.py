"""
Core logic for calculating Mahjong win conditions.

This module is responsible for determining the required points to win in the
final round of a Mahjong game based on the current scores and table situation.
It calculates conditions for three primary scenarios: winning by direct ron,
winning by ron from another player, and winning by tsumo.
"""
from points_lookup import reverse_lookup, ceil100
import math

def calculate_conditions(scores, oya, tsumibo, kyotaku):
    """
    Calculates the win conditions for the user ('自分').

    This function computes the necessary hand values to overtake the top player
    in three different scenarios:
    1. Direct Ron: Winning by ron from the current leader.
    2. Other Ron: Winning by ron from a player other than the leader.
    3. Tsumo: Winning by self-draw (tsumo).

    Args:
        scores (dict): A dictionary of scores for all four players.
        oya (str): The name of the player who is the dealer.
        tsumibo (int): The number of bonus sticks on the table.
        kyotaku (int): The number of riichi sticks on the table.

    Returns:
        dict: A dictionary containing the point difference to the top player
              ('top_diff'), the name of the leader ('leader'), and a list of
              result dictionaries ('results'), one for each scenario.
    """
    me = '自分'
    if me not in scores:
        raise ValueError('scores must include "自分"')

    leader = max(scores, key=lambda k: (scores[k], k))
    leader_score = scores[leader]
    my_score = scores[me]
    top_diff = leader_score - my_score + 1

    kyotaku_points = kyotaku * 1000
    ron_tsumibo_bonus = tsumibo * 300  # Ron winner gets 300 per stick.
    tsumo_bonus = tsumibo * 300 # Tsumo winner gets 100 per stick from each player (total 300).

    is_parent = (oya == me)
    role_str = "親" if is_parent else "子"

    results = []

    # --- 1. Direct Ron (from leader) ---
    # The point difference is doubled because the leader pays.
    need_direct = (top_diff - (kyotaku_points + ron_tsumibo_bonus)) / 2
    need_direct = ceil100(need_direct)
    rev_direct = reverse_lookup(need_direct, 'ron', is_parent)

    if isinstance(rev_direct['points'], int):
        total_points = rev_direct['points'] + kyotaku_points + ron_tsumibo_bonus
        opponent_loss = rev_direct['points']
        difference_points = rev_direct['points'] * 2
    else: # Mangan etc.
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

    # --- 2. Other Ron (from non-leader) ---
    need_other = top_diff - (kyotaku_points + ron_tsumibo_bonus)
    need_other = ceil100(need_other)
    rev_other = reverse_lookup(need_other, 'ron', is_parent)
    
    if isinstance(rev_other['points'], int):
        total_points = rev_other['points'] + kyotaku_points + ron_tsumibo_bonus
        opponent_loss = rev_other['points']
        difference_points = rev_other['points']
    else: # Mangan etc.
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

    # --- 3. Tsumo ---
    total_needed = top_diff - (kyotaku_points + tsumo_bonus)
    
    if is_parent:
        # Parent Tsumo: each opponent pays 1/3 of the total value.
        per_person_needed = ceil100(total_needed / 3.0)
        rev_t = reverse_lookup(per_person_needed, 'tsumo', True)
        
        if 'オール' in str(rev_t['points']):
            per_person_actual = int(str(rev_t['points']).replace('オール', ''))
            total_points = per_person_actual * 3
        else:
            per_person_actual = rev_t['points']
            total_points = per_person_actual * 3
            
        opponent_loss = per_person_actual
        difference_points = total_points + opponent_loss # Approximation
    else:
        # Child Tsumo: parent pays ~2x what other children pay.
        # Base the lookup on the payment from another child.
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
            
        difference_points = total_points # Approximation

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
