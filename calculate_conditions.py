from points_lookup import reverse_lookup
import math

def calculate_conditions(scores, oya, tsumibo, kyotaku):
    me = "自分"
    diff_target = max([s for p, s in scores.items() if p != me]) - scores[me]
    if diff_target <= 0:
        return [{"条件": "全条件", "points": 0, "rank": "不要", "detail": "すでにトップ"}]

    diff_target -= kyotaku * 1000
    tsumibo_points = tsumibo * 300
    is_parent = (me == oya)
    results = []

    # 直撃ロン
    ron_points_direct = diff_target + 100
    ron_points_direct -= tsumibo_points
    ron_points_direct = math.ceil(ron_points_direct / 100) * 100
    rank_str, _ = reverse_lookup(ron_points_direct, "ron", is_parent)
    results.append({
        "条件": "直撃ロン",
        "points": ron_points_direct,
        "rank": rank_str,
        "detail": f"点差{diff_target} + 積み棒{tsumibo_points}"
    })

    # 他家放銃ロン
    ron_points_other = diff_target + 100
    ron_points_other -= tsumibo_points
    ron_points_other = math.ceil(ron_points_other / 100) * 100
    rank_str, _ = reverse_lookup(ron_points_other, "ron", is_parent)
    results.append({
        "条件": "他家放銃ロン",
        "points": ron_points_other,
        "rank": rank_str,
        "detail": f"点差{diff_target} + 積み棒{tsumibo_points}"
    })

    # ツモ
    if is_parent:
        tsumo_payment = (diff_target + 100 - tsumibo_points) / 3
    else:
        tsumo_payment = (diff_target + 100 - tsumibo_points) / 4
    tsumo_payment = math.ceil(tsumo_payment / 100) * 100
    rank_str, _ = reverse_lookup(tsumo_payment, "tsumo", is_parent)
    results.append({
        "条件": "ツモ",
        "points": tsumo_payment,
        "rank": rank_str,
        "detail": f"点差{diff_target} + 積み棒{tsumibo_points}"
    })

    return results
