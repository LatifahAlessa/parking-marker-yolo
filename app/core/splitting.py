from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupInfo:
    group_id: str
    image_count: int


def groupwise_split(
    group_sizes: dict[str, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Split whole groups while approximately balancing image counts.

    No group can appear in more than one split. A seeded greedy assignment keeps
    the result reproducible while targeting the requested image ratios.
    """
    if not group_sizes:
        raise ValueError("No groups found")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    groups = [GroupInfo(k, v) for k, v in group_sizes.items()]
    if len(groups) < 3:
        raise ValueError("At least 3 vehicle groups are required for train/val/test splitting")

    rng = random.Random(seed)
    rng.shuffle(groups)
    # Large groups first improves image-count balance, while the shuffle breaks ties.
    groups.sort(key=lambda g: g.image_count, reverse=True)

    total = sum(g.image_count for g in groups)
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * test_ratio,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assignment: dict[str, str] = {}

    # Seed each split with one group so tiny datasets do not leave a split empty.
    initial_order = ["train", "val", "test"]
    for split, group in zip(initial_order, groups[:3]):
        assignment[group.group_id] = split
        counts[split] += group.image_count

    for group in groups[3:]:
        def score(split: str) -> tuple[float, float]:
            # Prefer the split furthest below its target proportionally.
            deficit = (targets[split] - counts[split]) / max(targets[split], 1.0)
            # A second term discourages overshooting very small target sets.
            projected_error = abs((counts[split] + group.image_count) - targets[split]) / max(targets[split], 1.0)
            return (deficit, -projected_error)

        split = max(("train", "val", "test"), key=score)
        assignment[group.group_id] = split
        counts[split] += group.image_count

    return assignment
