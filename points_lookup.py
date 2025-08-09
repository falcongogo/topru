from points_table import POINTS_TABLE
import math

CHILD_RON_MANGAN = 7700
PARENT_RON_MANGAN = 11600
PARENT_TSUMO_MANGAN = 4000
CHILD_TSUMO_MANGAN = 2000

def ceil100(x):
    return int(math.ceil(x / 100.0) * 100)

def reverse_lookup(points, method, is_parent):
    if points <= 0:
        return {'rank': '不要', 'points': 0}

    # Mangan and higher first
    if method == 'ron':
        if (not is_parent) and points >= CHILD_RON_MANGAN:
            return {'rank': '満貫', 'points': points}
        if is_parent and points >= PARENT_RON_MANGAN:
            return {'rank': '満貫', 'points': points}
    else:
        if is_parent and points >= PARENT_TSUMO_MANGAN:
            return {'rank': '満貫', 'points': f"{points}オール"}
        if (not is_parent) and points >= CHILD_TSUMO_MANGAN:
            parent_pay = points * 2
            return {'rank': '満貫', 'points': f"{points}-{parent_pay}"}

    role = 'parent' if is_parent else 'child'
    table = POINTS_TABLE[role][method]

    candidates = []
    for (fu, han), val in table.items():
        if fu > 50:
            continue
        if method == 'ron':
            if val >= points:
                candidates.append((han, fu, val))
        else:
            if role == 'parent':
                if val >= points:
                    candidates.append((han, fu, val))
            else:
                child_pay, parent_pay = val
                if child_pay >= points and parent_pay >= points * 2:
                    candidates.append((han, fu, val))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2] if isinstance(x[2], int) else x[2][0]))
        han, fu, val = candidates[0][0], candidates[0][1], candidates[0][2]
        if method == 'ron':
            return {'rank': f"{fu}符{han}翻", 'points': val}
        else:
            if role == 'parent':
                return {'rank': f"{fu}符{han}翻", 'points': f"{val}オール"}
            else:
                child_pay, parent_pay = val
                return {'rank': f"{fu}符{han}翻", 'points': f"{child_pay}-{parent_pay}"}

    return {'rank': '不可能', 'points': points}
