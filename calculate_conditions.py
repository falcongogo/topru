from points_table import POINTS_TABLE

def lookup_points(role, win_type, needed_points):
    for (fu, han), points in POINTS_TABLE[role][win_type].items():
        if win_type == 'ron':
            if points >= needed_points:
                if (role == 'child' and points >= 7700) or (role == 'parent' and points >= 11600):
                    return '満貫'
                return f"{fu}符{han}翻"
        else:
            pay_parent, pay_child = points
            if role == 'child':
                total = pay_parent + 2 * pay_child
            else:
                total = pay_parent * 3
            if total >= needed_points:
                if (role == 'child' and total >= 8000) or (role == 'parent' and total >= 12000):
                    return '満貫'
                return f"{fu}符{han}翻"
    return '不可能'

def calculate_conditions(scores, dealer, tsumi, kyotaku):
    names = ["自分", "下家", "対面", "上家"]
    my_idx = 0
    top_idx = max(range(4), key=lambda i: scores[i])
    role = 'parent' if dealer == my_idx else 'child'
    diff = scores[top_idx] - scores[my_idx] + 1

    diff -= tsumi * 300 + kyotaku * 1000

    ron_direct = diff
    ron_other = diff
    tsumo_needed = diff

    return {
        "直撃ロン": lookup_points(role, 'ron', ron_direct),
        "他家放銃ロン": lookup_points(role, 'ron', ron_other),
        "ツモ": lookup_points(role, 'tsumo', tsumo_needed)
    }
