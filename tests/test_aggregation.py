from app.aggregation import aggregate_group
from app.core.schemas import ImageResult, QuestionResult, TriState


def qr(s, c, image="x.jpg"):
    return QuestionResult(TriState(s), c, "test", image)


def image(name, q1, q2, q3):
    return ImageResult(name, "g", q1, q2, q3)


def test_decisive_view_beats_unclear_views():
    items = [
        image("a", qr("unclear", .48), qr("unclear", .48), qr("unclear", .48)),
        image("b", qr("unclear", .45), qr("unclear", .45), qr("unclear", .45)),
        image("c", qr("positive", .88), qr("positive", .82), qr("negative", .70)),
    ]
    out = aggregate_group(items)
    assert out["q1"]["status"] == "positive"
    assert out["final_decision"] == "regular"
