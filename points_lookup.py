"""
Reverse point lookup for Mahjong hands.

This module provides the functionality to find the minimum required hand (in terms
of fu and han, or by rank for Mangan+) to achieve a certain score. It is the
core of the "reverse lookup" feature of the application.
"""
from points_table import POINTS_TABLE
import math

# --- Constants for Mangan Thresholds ---
CHILD_RON_MANGAN = 8000
PARENT_RON_MANGAN = 12000
PARENT_TSUMO_MANGAN = 4000
CHILD_TSUMO_MANGAN = 2000

def ceil100(x: float) -> int:
    """
    Rounds a number up to the nearest 100.

    Args:
        x: The number to round up.

    Returns:
        The input number rounded up to the next multiple of 100.
    """
    return int(math.ceil(x / 100.0) * 100)

def reverse_lookup(points: int, method: str, is_parent: bool) -> dict:
    """
    Finds the minimum hand required to achieve a given number of points.

    This function performs a "reverse lookup" to find the smallest hand that
    is worth at least the target `points`. It first checks for Mangan-level
    hands and above. If the required points are lower, it searches the
    `POINTS_TABLE` for the best fu/han combination.

    Args:
        points: The minimum number of points the hand must be worth. For tsumo,
                this is the payment per player (or per child for child tsumo).
        method: The winning method, either 'ron' or 'tsumo'.
        is_parent: True if the player is the dealer (oya), False otherwise.

    Returns:
        A dictionary containing the 'rank' (e.g., "30符4翻", "満貫") and the
        actual 'points' value or display string for that hand. Returns
        '不要' (Unnecessary) if points <= 0, or '不可能' (Impossible) if
        no suitable hand can be found.
    """
    if points <= 0:
        return {'rank': '不要', 'points': 0}

    # --- Mangan and above ---
    # For high-point hands, we return a rank name instead of fu/han.
    mangan_value = PARENT_RON_MANGAN if is_parent else CHILD_RON_MANGAN
    if method == 'ron' and points >= mangan_value:
        yakuman_val = 48000 if is_parent else 32000
        if points >= yakuman_val: return {'rank': '役満', 'points': yakuman_val}
        if points >= yakuman_val * 0.75: return {'rank': '三倍満', 'points': int(yakuman_val * 0.75)}
        if points >= yakuman_val * 0.5: return {'rank': '倍満', 'points': int(yakuman_val * 0.5)}
        if points >= yakuman_val * 0.375: return {'rank': '跳満', 'points': int(yakuman_val * 0.375)}
        return {'rank': '満貫', 'points': mangan_value}

    mangan_tsumo_value = PARENT_TSUMO_MANGAN if is_parent else CHILD_TSUMO_MANGAN
    if method == 'tsumo' and points >= mangan_tsumo_value:
        yakuman_val = 16000 if is_parent else 8000
        if points >= yakuman_val: return {'rank': '役満', 'points': f"{yakuman_val}オール" if is_parent else f"{yakuman_val}-{yakuman_val*2}"}
        if points >= yakuman_val * 0.75: return {'rank': '三倍満', 'points': f"{int(yakuman_val*0.75)}オール" if is_parent else f"{int(yakuman_val*0.75)}-{int(yakuman_val*0.75)*2}"}
        if points >= yakuman_val * 0.5: return {'rank': '倍満', 'points': f"{int(yakuman_val*0.5)}オール" if is_parent else f"{int(yakuman_val*0.5)}-{int(yakuman_val*0.5)*2}"}
        if points >= yakuman_val * 0.375: return {'rank': '跳満', 'points': f"{int(yakuman_val*0.375)}オール" if is_parent else f"{int(yakuman_val*0.375)}-{int(yakuman_val*0.375)*2}"}
        return {'rank': '満貫', 'points': f"{mangan_tsumo_value}オール" if is_parent else f"{mangan_tsumo_value}-{mangan_tsumo_value*2}"}

    # --- Below Mangan ---
    # Search the points table for the best candidate.
    role = 'parent' if is_parent else 'child'
    table = POINTS_TABLE[role][method]

    candidates = []
    for (fu, han), val in table.items():
        # Rules: No 20fu ron, only up to 50fu/4han for this table
        if fu > 50 or han > 4: continue
        if method == 'ron' and fu == 20: continue

        required_val = val if method == 'ron' or is_parent else val[0]
        if required_val >= points:
            candidates.append((han, fu, val))

    if not candidates:
        return {'rank': '不可能', 'points': points}

    # Find the best candidate (lowest han, then lowest fu, then lowest score)
    candidates.sort(key=lambda x: (x[0], x[1], x[2] if isinstance(x[2], int) else x[2][0]))
    han, fu, val = candidates[0]

    if method == 'ron':
        return {'rank': f"{fu}符{han}翻", 'points': val}
    else: # tsumo
        if is_parent:
            return {'rank': f"{fu}符{han}翻", 'points': f"{val}オール"}
        else:
            child_pay, parent_pay = val
            return {'rank': f"{fu}符{han}翻", 'points': f"{child_pay}-{parent_pay}"}