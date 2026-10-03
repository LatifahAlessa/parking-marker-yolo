from app.core.grouping import parse_filename
from app.core.splitting import groupwise_split


def test_grouping_variable_digits():
    p = parse_filename("VI_1448_121803_001.jpeg")
    assert p.group_id == "VI_1448_121803"
    assert p.view_id == "001"
    p = parse_filename("IV_00184_12.jpg")
    assert p.group_id == "IV_00184"
    assert p.view_id == "12"


def test_onedrive_suffix():
    p = parse_filename("VI_1448_163042_003__OneDrive_2_9-1-2026-2.jpeg")
    assert p.group_id == "VI_1448_163042"
    assert p.view_id == "003"


def test_groupwise_split_no_leakage_and_approx_ratio():
    sizes = {f"g{i}": 4 for i in range(20)}
    split = groupwise_split(sizes, seed=42)
    assert set(split) == set(sizes)
    counts = {k: 0 for k in ("train", "val", "test")}
    for group, part in split.items():
        counts[part] += sizes[group]
    total = sum(counts.values())
    assert abs(counts["train"] / total - .70) <= .10
    assert abs(counts["val"] / total - .15) <= .10
    assert abs(counts["test"] / total - .15) <= .10
