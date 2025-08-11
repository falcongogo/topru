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

    results = []

    # Direct Ron (from leader) - 直撃時は点差を倍縮まる（半分ではなく）
    need_direct = top_diff - kyotaku_points - tsumibo_points
    need_direct = max(0, need_direct)
    # 直撃時は点差が倍縮まる = 必要な点数が半分になる
    need_direct = need_direct / 2
    need_direct = ceil100(need_direct)
    rev_direct = reverse_lookup(need_direct, 'ron', is_parent)
    
    # 直撃時の合計点数と相手のマイナス点数を計算
    if isinstance(rev_direct['points'], int):
        total_points = rev_direct['points']
        opponent_loss = rev_direct['points']
    else:
        # 点数が文字列の場合は数値に変換
        total_points = rev_direct['points']
        opponent_loss = rev_direct['points']
    
    # 差分点数を計算（和了時の取得合計点数 + 相手から取得した点数）
    difference_points = total_points + opponent_loss
    
    results.append({
        '条件': f'直撃ロン（{leader}）（{role_str}）',
        'need_points': need_direct,
        'rank': rev_direct['rank'],
        'display': rev_direct['points'],
        'total_points': total_points,
        'opponent_loss': opponent_loss,
        'difference_points': difference_points,
        'is_direct': True
    })

    # Other Ron (no name)
    need_other = top_diff - kyotaku_points - tsumibo_points
    need_other = max(0, need_other)
    need_other = ceil100(need_other)
    rev_other = reverse_lookup(need_other, 'ron', is_parent)
    
    # 他家放銃時の合計点数と相手のマイナス点数
    if isinstance(rev_other['points'], int):
        total_points = rev_other['points']
        opponent_loss = rev_other['points']
    else:
        total_points = rev_other['points']
        opponent_loss = rev_other['points']
    
    # 差分点数を計算（和了時の取得合計点数 + 相手から取得した点数）
    difference_points = total_points + opponent_loss
    
    results.append({
        '条件': f'他家放銃ロン（{role_str}）',
        'need_points': need_other,
        'rank': rev_other['rank'],
        'display': rev_other['points'],
        'total_points': total_points,
        'opponent_loss': opponent_loss,
        'difference_points': difference_points,
        'is_direct': False
    })

    # Tsumo
    total_needed = top_diff - kyotaku_points - tsumo_tsumibo_points
    total_needed = max(0, total_needed)
    
    if is_parent:
        # 親ツモ：子3人から1倍ずつ = 合計3倍
        # 必要な合計点数を3で割って、1人あたりの支払い額を計算
        per_person = math.ceil(total_needed / 3.0)
        per_person = ceil100(per_person)
        rev_t = reverse_lookup(per_person, 'tsumo', True)
        
        # 親ツモの合計点数と相手のマイナス点数
        if isinstance(rev_t['points'], str) and 'オール' in rev_t['points']:
            # "4000オール" のような形式
            per_person_actual = int(rev_t['points'].replace('オール', ''))
            total_points = per_person_actual * 3
            opponent_loss = per_person_actual
        else:
            per_person_actual = rev_t['points']
            total_points = per_person_actual * 3
            opponent_loss = per_person_actual
        
        # もし実際に貰える点数が必要な点数より少ない場合は、より高い点数を検索
        if total_points < total_needed:
            # より高い点数で再検索
            higher_per_person = math.ceil(total_needed / 3.0) + 100
            higher_per_person = ceil100(higher_per_person)
            rev_t_higher = reverse_lookup(higher_per_person, 'tsumo', True)
            
            if isinstance(rev_t_higher['points'], str) and 'オール' in rev_t_higher['points']:
                per_person_actual = int(rev_t_higher['points'].replace('オール', ''))
                total_points = per_person_actual * 3
                opponent_loss = per_person_actual
                rev_t = rev_t_higher
            else:
                per_person_actual = rev_t_higher['points']
                total_points = per_person_actual * 3
                opponent_loss = per_person_actual
                rev_t = rev_t_higher
        
        # 差分点数を計算（和了時の取得合計点数 + 相手から取得した点数）
        difference_points = total_points + opponent_loss
        
        results.append({
            '条件': f'ツモ（{role_str}）',
            'need_points': total_points,  # 実際に必要な合計点数
            'rank': rev_t['rank'],
            'display': rev_t['points'],
            'total_points': total_points,
            'opponent_loss': opponent_loss,
            'difference_points': difference_points,
            'is_direct': False
        })
    else:
        # 子ツモ：親1人から2倍、子2人から1倍ずつ = 合計4倍
        # 必要な合計点数を4で割って、子の支払い額を基準に計算
        # ただし、親は子の2倍支払うので、実際の計算は複雑
        # まず子の支払い額を推定
        child_payment = math.ceil(total_needed / 4.0)
        child_payment = ceil100(child_payment)
        
        rev_t = reverse_lookup(child_payment, 'tsumo', False)
        
        # 子ツモの合計点数と相手のマイナス点数
        if isinstance(rev_t['points'], str) and '-' in rev_t['points']:
            # "2000-4000" のような形式
            child_pay, parent_pay = map(int, rev_t['points'].split('-'))
            total_points = child_pay * 2 + parent_pay  # 子2人 + 親1人
            opponent_loss = f"子{child_pay}点×2 + 親{parent_pay}点"
        else:
            total_points = rev_t['points'] * 4
            opponent_loss = rev_t['points']
        
        # もし実際に貰える点数が必要な点数より少ない場合は、より高い点数を検索
        if total_points < total_needed:
            # より高い点数で再検索
            higher_child_payment = math.ceil(total_needed / 4.0) + 100
            higher_child_payment = ceil100(higher_child_payment)
            rev_t_higher = reverse_lookup(higher_child_payment, 'tsumo', False)
            
            if isinstance(rev_t_higher['points'], str) and '-' in rev_t_higher['points']:
                child_pay, parent_pay = map(int, rev_t_higher['points'].split('-'))
                total_points = child_pay * 2 + parent_pay
                opponent_loss = f"子{child_pay}点×2 + 親{parent_pay}点"
                rev_t = rev_t_higher
            else:
                total_points = rev_t_higher['points'] * 4
                opponent_loss = rev_t_higher['points']
                rev_t = rev_t_higher
        
        # 差分点数を計算（和了時の取得合計点数 + 相手から取得した点数）
        difference_points = total_points + opponent_loss
        
        results.append({
            '条件': f'ツモ（{role_str}）',
            'need_points': total_points,  # 実際に必要な合計点数
            'rank': rev_t['rank'],
            'display': rev_t['points'],
            'total_points': total_points,
            'opponent_loss': opponent_loss,
            'difference_points': difference_points,
            'is_direct': False
        })

    return {'top_diff': top_diff, 'leader': leader, 'results': results}
