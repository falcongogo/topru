from points_table import POINTS_TABLE
import math

CHILD_RON_MANGAN = 8000
PARENT_RON_MANGAN = 12000
PARENT_TSUMO_MANGAN = 4000
CHILD_TSUMO_MANGAN = 2000

def ceil100(x):
    return int(math.ceil(x / 100.0) * 100)

def reverse_lookup(points, method, is_parent):
    if points <= 0:
        return {'rank': '不要', 'points': 0, 'display': 0}

    # 満貫以上（跳満・倍満等も）はrankで明示
    if method == 'ron':
        if (not is_parent) and points >= CHILD_RON_MANGAN:
            if points >= 32000:
                return {'rank': '役満', 'points': points}
            elif points >= 24000:
                return {'rank': '三倍満', 'points': points}
            elif points >= 16000:
                return {'rank': '倍満', 'points': points}
            elif points >= 12000:
                return {'rank': '跳満', 'points': points}
            else:
                return {'rank': '満貫', 'points': points}
        if is_parent and points >= PARENT_RON_MANGAN:
            if points >= 48000:
                return {'rank': '役満', 'points': points}
            elif points >= 36000:
                return {'rank': '三倍満', 'points': points}
            elif points >= 24000:
                return {'rank': '倍満', 'points': points}
            elif points >= 18000:
                return {'rank': '跳満', 'points': points}
            else:
                return {'rank': '満貫', 'points': points}
    else:
        if is_parent and points >= PARENT_TSUMO_MANGAN:
            if points >= 16000:
                return {'rank': '役満', 'points': points, 'display': f"{points}オール"}
            elif points >= 12000:
                return {'rank': '三倍満', 'points': points, 'display': f"{points}オール"}
            elif points >= 8000:
                return {'rank': '倍満', 'points': points, 'display': f"{points}オール"}
            elif points >= 6000:
                return {'rank': '跳満', 'points': points, 'display': f"{points}オール"}
            elif points >= 4000:
                return {'rank': '満貫', 'points': points, 'display': f"{points}オール"}
        if (not is_parent) and points >= CHILD_TSUMO_MANGAN:
            parent_pay = points * 2
            if points >= 8000:
                return {'rank': '役満', 'points': points, 'display': f"{points}-{parent_pay}"}
            elif points >= 6000:
                return {'rank': '三倍満', 'points': points, 'display': f"{points}-{parent_pay}"}
            elif points >= 4000:
                return {'rank': '倍満', 'points': points, 'display': f"{points}-{parent_pay}"}
            elif points >= 3000:
                return {'rank': '跳満', 'points': points, 'display': f"{points}-{parent_pay}"}
            elif points >= 2000:
                return {'rank': '満貫', 'points': points, 'display': f"{points}-{parent_pay}"}

    role = 'parent' if is_parent else 'child'
    table = POINTS_TABLE[role][method]

    # 満貫未満は50符以下のみ、翻数制限
    candidates = []
    for (fu, han), val in table.items():
        if fu > 50:
            continue
        if han > 4:
            continue
        if method == 'ron' and fu == 20:
            continue  # ロンで20符は不可
        if method == 'ron':
            if val >= points:
                candidates.append((han, fu, val))
        else:
            if role == 'parent':
                if val >= points:
                    candidates.append((han, fu, val))
            else:
                child_pay, parent_pay = val
                # 子ツモ：親から2倍、子から1倍の点数をもらう
                # 必要な点数は子の支払い額以上で、親の支払い額は子の支払い額の2倍以上
                if child_pay >= points:
                    candidates.append((han, fu, val))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2] if isinstance(x[2], int) else x[2][0]))
        han, fu, val = candidates[0][0], candidates[0][1], candidates[0][2]
        if method == 'ron':
            return {'rank': f"{fu}符{han}翻", 'points': val, 'display': val}
        else:
            if role == 'parent':
                return {'rank': f"{fu}符{han}翻", 'points': val, 'display': f"{val}オール"}
            else:
                child_pay, parent_pay = val
                return {'rank': f"{fu}符{han}翻", 'points': child_pay, 'display': f"{child_pay}-{parent_pay}"}

    return {'rank': '不可能', 'points': points, 'display': points}