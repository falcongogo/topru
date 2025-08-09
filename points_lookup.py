from points_table import POINTS_TABLE
import math

# thresholds: child ron 7700 -> mangan, parent ron 11600 -> mangan
def reverse_lookup(points, method, is_parent):
    '''
    points: for ron -> required ron point (int)
            for tsumo -> per-person payment (int)
    method: 'ron' or 'tsumo'
    is_parent: True if winner is parent
    Returns: dict with keys: 'rank' (display string), 'points' (displayable points)
    '''
    if points <= 0:
        return {'rank': '不要', 'points': 0}
    # Determine mangan thresholds
    if method == 'ron':
        if not is_parent and points >= 7700:
            return {'rank': '満貫', 'points': points}
        if is_parent and points >= 11600:
            return {'rank': '満貫', 'points': points}
    else:  # tsumo
        # parent per-person >=4000 => mangan
        if is_parent and points >= 4000:
            return {'rank': '満貫', 'points': f"{points}オール"}
        # child per-person >=2000 => mangan (child pays 2000, parent 4000 total 6000? we treat per-unit)
        if (not is_parent) and points >= 2000:
            # represent as child-parent pair
            parent_pay = points * 2
            return {'rank': '満貫', 'points': f"{points}-{parent_pay}"}

    role = 'parent' if is_parent else 'child'
    table = POINTS_TABLE[role][method]

    # Search for matching (fu,han) with fu <=50 only
    # For ron: compare table[(fu,han)] >= points
    # For tsumo parent: table value is per-person amount >= points
    # For tsumo child: table value is tuple (child_pay, parent_pay) compare child_pay >= points and parent_pay >= points*2
    candidates = []
    for (fu, han), val in table.items():
        if fu > 50:  # skip 60+ as per UI requirement
            continue
        if method == 'ron':
            if val >= points:
                candidates.append((fu, han, val))
        elif method == 'tsumo':
            if role == 'parent':
                if val >= points:
                    candidates.append((fu, han, val))
            else:
                child_pay, parent_pay = val
                # compute required parent pay based on per-unit points assumption
                if child_pay >= points and parent_pay >= points*2:
                    candidates.append((fu, han, val))
    # choose smallest (by han then fu then points)
    if candidates:
        # sort by han asc then fu asc then val asc
        candidates.sort(key=lambda x: (x[1], x[0], x[2]))
        fu, han, val = candidates[0]
        if method == 'ron':
            return {'rank': f"{fu}符{han}翻", 'points': val}
        else:
            if role == 'parent':
                return {'rank': f"{fu}符{han}翻", 'points': f"{val}オール"}
            else:
                child_pay, parent_pay = val
                return {'rank': f"{fu}符{han}翻", 'points': f"{child_pay}-{parent_pay}"}

    return {'rank': '不可能', 'points': points}
