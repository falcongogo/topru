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

    is_parent = (oya == me)
    role_str = "親" if is_parent else "子"

    results = []

    # 1. まず逆転に必要な点数を計算（供託棒・積み棒なし）
    need_for_reverse = top_diff

    # 2. Direct Ron (from leader) - 直撃時は点差を倍縮まる
    need_direct = need_for_reverse / 2  # 直撃時は点差が倍縮まる
    need_direct = ceil100(need_direct)
    rev_direct = reverse_lookup(need_direct, 'ron', is_parent)
    
    # 3. 供託棒と積み棒を差し引く
    kyotaku_points = kyotaku * 1000
    tsumibo_points = tsumibo * 300 * 2  # ロン時の積み棒
    
    # 実際に必要な点数 = 逆転に必要な点数 - 供託棒 - 積み棒
    actual_need_direct = need_direct - kyotaku_points - tsumibo_points
    actual_need_direct = max(0, actual_need_direct)
    
    # 必要に応じてより高い点数を検索
    if actual_need_direct > 0:
        rev_direct = reverse_lookup(actual_need_direct, 'ron', is_parent)
    
    # 直撃時の合計点数と相手のマイナス点数を計算
    if isinstance(rev_direct['points'], (int, float)):
        total_points = int(rev_direct['points'])
        opponent_loss = int(rev_direct['points'])
    elif isinstance(rev_direct['points'], str):
        # 文字列の数値の場合
        try:
            points_val = float(rev_direct['points'])
            total_points = int(points_val)
            opponent_loss = int(points_val)
        except ValueError:
            # 文字列形式の場合はそのまま使用
            total_points = rev_direct['points']
            opponent_loss = rev_direct['points']
    else:
        total_points = rev_direct['points']
        opponent_loss = rev_direct['points']
    
    # 差分点数を計算
    if isinstance(opponent_loss, str):
        difference_points = total_points
    else:
        difference_points = total_points + opponent_loss
    
    results.append({
        '条件': f'直撃ロン（{leader}）（{role_str}）',
        'need_points': actual_need_direct,
        'rank': rev_direct['rank'],
        'display': rev_direct.get('display', rev_direct['points']),
        'total_points': total_points,
        'opponent_loss': opponent_loss,
        'difference_points': difference_points,
        'is_direct': True
    })

    # 4. Other Ron (no name)
    need_other = need_for_reverse  # 他家放銃時は点差そのまま
    need_other = ceil100(need_other)
    rev_other = reverse_lookup(need_other, 'ron', is_parent)
    
    # 供託棒と積み棒を差し引く
    actual_need_other = need_other - kyotaku_points - tsumibo_points
    actual_need_other = max(0, actual_need_other)
    
    # 必要に応じてより高い点数を検索
    if actual_need_other > 0:
        rev_other = reverse_lookup(actual_need_other, 'ron', is_parent)
    
    # 他家放銃時の合計点数と相手のマイナス点数
    if isinstance(rev_other['points'], (int, float)):
        total_points = int(rev_other['points'])
        opponent_loss = int(rev_other['points'])
    elif isinstance(rev_other['points'], str):
        # 文字列の数値の場合
        try:
            points_val = float(rev_other['points'])
            total_points = int(points_val)
            opponent_loss = int(points_val)
        except ValueError:
            # 文字列形式の場合はそのまま使用
            total_points = rev_other['points']
            opponent_loss = rev_other['points']
    else:
        total_points = rev_other['points']
        opponent_loss = rev_other['points']
    
    # 差分点数を計算
    if isinstance(opponent_loss, str):
        difference_points = total_points
    else:
        difference_points = total_points + opponent_loss
    
    results.append({
        '条件': f'他家放銃ロン（{role_str}）',
        'need_points': actual_need_other,
        'rank': rev_other['rank'],
        'display': rev_other.get('display', rev_other['points']),
        'total_points': total_points,
        'opponent_loss': opponent_loss,
        'difference_points': difference_points,
        'is_direct': False
    })

    # 5. Tsumo - ツモ時は積み棒の計算が異なる
    if is_parent:
        # 親ツモ：子3人から1倍ずつ = 合計3倍
        # まず逆転に必要な点数を3で割って、1人あたりの支払い額を計算
        per_person = math.ceil(need_for_reverse / 3.0)
        per_person = ceil100(per_person)
        rev_t = reverse_lookup(per_person, 'tsumo', True)
        
        # ツモ時の積み棒を差し引く（親ツモ：400点×積み棒数）
        tsumo_tsumibo_points = tsumibo * 400
        actual_need_tsumo = per_person - (tsumo_tsumibo_points / 3)  # 1人あたりの積み棒分
        actual_need_tsumo = max(0, actual_need_tsumo)
        
        # 必要に応じてより高い点数を検索
        if actual_need_tsumo > 0:
            rev_t = reverse_lookup(actual_need_tsumo, 'tsumo', True)
        
        # 親ツモの合計点数と相手のマイナス点数
        if isinstance(rev_t['points'], (int, float)):
            # 数値の場合は3倍（子3人）
            per_person_actual = int(rev_t['points'])
            total_points = per_person_actual * 3
            opponent_loss = per_person_actual
        elif isinstance(rev_t['points'], str):
            # 文字列の数値の場合
            try:
                points_val = float(rev_t['points'])
                per_person_actual = int(points_val)
                total_points = per_person_actual * 3
                opponent_loss = per_person_actual
            except ValueError:
                # 文字列形式の場合はそのまま使用
                per_person_actual = rev_t['points']
                total_points = per_person_actual * 3
                opponent_loss = per_person_actual
        else:
            per_person_actual = rev_t['points']
            total_points = per_person_actual * 3
            opponent_loss = per_person_actual
        
        # 差分点数を計算
        if isinstance(opponent_loss, str):
            difference_points = total_points
        else:
            difference_points = total_points + opponent_loss
        
        results.append({
            '条件': f'ツモ（{role_str}）',
            'need_points': actual_need_tsumo,
            'rank': rev_t['rank'],
            'display': rev_t.get('display', rev_t['points']),
            'total_points': total_points,
            'opponent_loss': opponent_loss,
            'difference_points': difference_points,
            'is_direct': False
        })
    else:
        # 子ツモ：親1人から2倍、子2人から1倍ずつ = 合計4倍
        # まず逆転に必要な点数を4で割って、子の支払い額を基準に計算
        child_payment = math.ceil(need_for_reverse / 4.0)
        child_payment = ceil100(child_payment)
        
        rev_t = reverse_lookup(child_payment, 'tsumo', False)
        
        # ツモ時の積み棒を差し引く（子ツモ：400点×積み棒数）
        tsumo_tsumibo_points = tsumibo * 400
        actual_need_tsumo = child_payment - (tsumo_tsumibo_points / 4)  # 1人あたりの積み棒分
        actual_need_tsumo = max(0, actual_need_tsumo)
        
        # 必要に応じてより高い点数を検索
        if actual_need_tsumo > 0:
            rev_t = reverse_lookup(actual_need_tsumo, 'tsumo', False)
        
        # 子ツモの合計点数と相手のマイナス点数
        if isinstance(rev_t['points'], (int, float)):
            # 数値の場合は4倍（子2人 + 親1人×2）
            total_points = int(rev_t['points'] * 4)
            opponent_loss = int(rev_t['points'])
        elif isinstance(rev_t['points'], str):
            # 文字列の数値の場合
            try:
                points_val = float(rev_t['points'])
                total_points = int(points_val * 4)
                opponent_loss = int(points_val)
            except ValueError:
                # 文字列形式の場合はそのまま使用
                total_points = rev_t['points'] * 4
                opponent_loss = rev_t['points']
        else:
            total_points = rev_t['points'] * 4
            opponent_loss = rev_t['points']
        
        # 差分点数を計算
        if isinstance(opponent_loss, str):
            difference_points = total_points
        else:
            difference_points = total_points + opponent_loss
        
        results.append({
            '条件': f'ツモ（{role_str}）',
            'need_points': actual_need_tsumo,
            'rank': rev_t['rank'],
            'display': rev_t.get('display', rev_t['points']),
            'total_points': total_points,
            'opponent_loss': opponent_loss,
            'difference_points': difference_points,
            'is_direct': False
        })

    return {'top_diff': top_diff, 'leader': leader, 'results': results}
