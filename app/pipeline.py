from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np

from app.aggregation import aggregate_group
from app.core.grouping import parse_filename, sha256_bytes
from app.core.schemas import ImageResult, QuestionResult, TriState
from app.vision.geometry import cat_eye_line_mask, decide_q2, ground_visibility
from app.vision.yolo_models import ModelFilesMissing, YoloModels


class ParkingCompliancePipeline:
    def __init__(self) -> None:
        self.models = YoloModels()

    def model_status(self) -> dict:
        return self.models.status()

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid or unreadable image")
        return image

    @staticmethod
    def _to_tri(value: str) -> TriState:
        return {"positive": TriState.POSITIVE, "negative": TriState.NEGATIVE}.get(value, TriState.UNCLEAR)

    def analyze_image(self, filename: str, data: bytes) -> ImageResult:
        parsed = parse_filename(filename)
        image = self._decode(data)
        h, w = image.shape[:2]

        vehicle = self.models.vehicle(image)
        boundaries = self.models.boundaries(image)
        cat_eyes = self.models.cat_eyes(image)
        ground_vis = ground_visibility(vehicle, (h, w))

        # Q3: direct object detection. A single high-confidence cat eye or multiple
        # moderate-confidence detections are enough to establish presence.
        high = [x for x in cat_eyes if x.confidence >= 0.62]
        moderate = [x for x in cat_eyes if x.confidence >= 0.38]
        if high or len(moderate) >= 2:
            q3_conf = max([x.confidence for x in cat_eyes], default=0.0)
            q3 = QuestionResult(
                TriState.POSITIVE,
                max(q3_conf, 0.65),
                f"YOLO detected {len(cat_eyes)} cat-eye reflector candidate(s)",
                filename,
            )
        elif ground_vis >= 0.42:
            q3 = QuestionResult(TriState.NEGATIVE, 0.62, "Relevant road area is visible and no cat-eye reflector was detected", filename)
        else:
            q3 = QuestionResult(TriState.UNCLEAR, 0.45, "No reliable cat-eye detection and the relevant road area is not sufficiently visible", filename)

        # Q1: boundary classes come from the custom segmentation model. A row of
        # >=3 detected cat eyes may also provide boundary evidence.
        strong_boundaries = [x for x in boundaries if x.confidence >= 0.35 and int(x.mask.sum()) >= max(80, int(0.00015 * h * w))]
        cat_line = cat_eye_line_mask(moderate, (h, w), min_points=3)
        if strong_boundaries or cat_line is not None:
            names = sorted({x.class_name for x in strong_boundaries})
            evidence = names + (["cat_eye_row"] if cat_line is not None else [])
            confs = [x.confidence for x in strong_boundaries] + ([max((x.confidence for x in moderate), default=0.55)] if cat_line is not None else [])
            q1 = QuestionResult(
                TriState.POSITIVE,
                min(0.98, max(confs, default=0.60)),
                "Usable parking boundary evidence detected: " + ", ".join(evidence),
                filename,
            )
        elif ground_vis >= 0.42:
            q1 = QuestionResult(TriState.NEGATIVE, 0.60, "Ground around the target vehicle is visible but no usable parking boundary was detected", filename)
        else:
            q1 = QuestionResult(TriState.UNCLEAR, 0.46, "Not enough visible ground/boundary geometry to determine the parking space", filename)

        if q1.status != TriState.POSITIVE:
            q2 = QuestionResult(TriState.UNCLEAR, q1.confidence, "Q2 cannot be determined because Q1 did not establish a usable boundary", filename)
            q2_details = {}
        else:
            decision = decide_q2(vehicle, strong_boundaries, moderate, (h, w))
            q2 = QuestionResult(self._to_tri(decision.status), decision.confidence, decision.reason, filename)
            q2_details = decision.details

        return ImageResult(
            filename=filename,
            group_id=parsed.group_id,
            q1=q1,
            q2=q2,
            q3=q3,
            meta={
                "image_size": [w, h],
                "vehicle_detected": vehicle is not None,
                "vehicle_confidence": round(vehicle.confidence, 4) if vehicle else None,
                "boundary_detections": [
                    {"class": x.class_name, "confidence": round(x.confidence, 4), "bbox": list(x.bbox)} for x in boundaries
                ],
                "cat_eye_detections": [
                    {"confidence": round(x.confidence, 4), "bbox": list(x.bbox)} for x in cat_eyes
                ],
                "ground_visibility": round(ground_vis, 4),
                "q2_geometry": q2_details,
            },
        )

    def analyze_files(self, files: Iterable[tuple[str, bytes]]) -> dict:
        self.models.ensure_ready()
        seen: dict[str, str] = {}
        duplicates: list[dict[str, str]] = []
        grouped: dict[str, list[ImageResult]] = defaultdict(list)
        received = 0

        for filename, data in files:
            received += 1
            digest = sha256_bytes(data)
            if digest in seen:
                duplicates.append({"filename": filename, "duplicate_of": seen[digest]})
                continue
            seen[digest] = filename
            result = self.analyze_image(filename, data)
            grouped[result.group_id].append(result)

        if not grouped:
            raise ValueError("No valid unique images were processed")

        groups = [aggregate_group(sorted(items, key=lambda x: x.filename)) for _, items in sorted(grouped.items())]
        return {
            "received_images": received,
            "unique_images": len(seen),
            "duplicates": duplicates,
            "groups": groups,
        }
