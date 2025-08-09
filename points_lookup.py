def reverse_lookup(points, win_type, is_parent):
    table = [
        (2000, "20符1翻"), (2600, "30符1翻"), (3200, "40符1翻"), (3900, "50符1翻"),
        (4000, "満貫"), (5200, "跳満"), (8000, "倍満"), (12000, "三倍満"), (16000, "役満")
    ]
    for limit, label in table:
        if points <= limit:
            return label, limit
    return "役満以上", points
