from __future__ import annotations

from app.core.schemas import ImageResult, QuestionResult, TriState


def aggregate_question(
    question: str,
    images: list[ImageResult],
    decisive_threshold: float = 0.50,
) -> QuestionResult:
    """
    Q2 aggregation.

    A decisive negative spatial observation is enough to establish
    incorrect parking. A different view that does not expose the
    violation must not cancel it.
    """
    items = [getattr(x, question) for x in images]

    neg = [
        x for x in items
        if x.status == TriState.NEGATIVE
        and x.confidence >= decisive_threshold
    ]

    pos = [
        x for x in items
        if x.status == TriState.POSITIVE
        and x.confidence >= decisive_threshold
    ]

    if neg:
        return max(neg, key=lambda x: x.confidence)

    if pos:
        return max(pos, key=lambda x: x.confidence)

    return max(items, key=lambda x: x.confidence)


def aggregate_presence(
    question: str,
    images: list[ImageResult],
    decisive_threshold: float = 0.50,
) -> QuestionResult:
    """
    Q1 and Q3 are existence questions.

    A decisive positive view establishes presence.
    Other views failing to observe the same feature must not cancel it.
    """
    items = [getattr(x, question) for x in images]

    pos = [
        x for x in items
        if x.status == TriState.POSITIVE
        and x.confidence >= decisive_threshold
    ]

    if pos:
        return max(pos, key=lambda x: x.confidence)

    neg = [
        x for x in items
        if x.status == TriState.NEGATIVE
        and x.confidence >= decisive_threshold
    ]

    if neg:
        return max(neg, key=lambda x: x.confidence)

    return max(items, key=lambda x: x.confidence)


def aggregate_group(images: list[ImageResult]) -> dict:
    q1 = aggregate_presence("q1", images)
    q3 = aggregate_presence("q3", images)

    boundary_views = [
        image
        for image in images
        if image.q1.status == TriState.POSITIVE
    ]

    if boundary_views:
        q2 = aggregate_question("q2", boundary_views)
    else:
        q2 = max(
            (image.q2 for image in images),
            key=lambda x: x.confidence,
        )

    if q2.status == TriState.NEGATIVE:
        final = "irregular"
        final_reason = (
            "A decisive view shows the vehicle incorrectly positioned "
            "relative to a visible boundary"
        )

    elif (
        q1.status == TriState.POSITIVE
        and q2.status == TriState.POSITIVE
    ):
        final = "regular"
        final_reason = (
            "A decisive view exposes a usable boundary and places "
            "the vehicle correctly relative to it"
        )

    else:
        final = "undetermined"
        final_reason = (
            "The available views do not support a reliable "
            "regular/irregular decision"
        )

    return {
        "group_id": images[0].group_id,
        "final_decision": final,
        "final_reason": final_reason,
        "q1": q1.to_dict(),
        "q2": q2.to_dict(),
        "q3": q3.to_dict(),
        "images": [x.to_dict() for x in images],
    }
