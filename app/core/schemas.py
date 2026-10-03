from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TriState(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCLEAR = "unclear"


@dataclass
class QuestionResult:
    status: TriState
    confidence: float
    reason: str
    evidence_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        out["confidence"] = round(float(self.confidence), 4)
        return out


@dataclass
class ImageResult:
    filename: str
    group_id: str
    q1: QuestionResult
    q2: QuestionResult
    q3: QuestionResult
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "group_id": self.group_id,
            "q1": self.q1.to_dict(),
            "q2": self.q2.to_dict(),
            "q3": self.q3.to_dict(),
            "meta": self.meta,
        }
