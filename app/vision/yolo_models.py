from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

BOUNDARY_CLASSES = {0: "white_line", 1: "allowed_curb", 2: "forbidden_curb", 3: "hatched_area"}
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck


@dataclass
class SegDetection:
    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class BoxDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class ModelFilesMissing(RuntimeError):
    pass


def _preferred_model(folder: Path, base_name: str = "best") -> Path | None:
    onnx = folder / f"{base_name}.onnx"
    pt = folder / f"{base_name}.pt"
    if onnx.exists():
        return onnx
    if pt.exists():
        return pt
    return None


class YoloModels:
    def __init__(self) -> None:
        self.vehicle_path = Path(os.getenv("VEHICLE_MODEL", ROOT / "models" / "vehicle" / "yolo11n-seg.pt"))
        self.boundary_path = Path(os.getenv("BOUNDARY_MODEL", "")) if os.getenv("BOUNDARY_MODEL") else _preferred_model(ROOT / "models" / "boundary")
        self.cat_eye_path = Path(os.getenv("CAT_EYE_MODEL", "")) if os.getenv("CAT_EYE_MODEL") else _preferred_model(ROOT / "models" / "cat_eye")
        self._vehicle = None
        self._boundary = None
        self._cat_eye = None

    def status(self) -> dict[str, Any]:
        return {
            "vehicle_model": str(self.vehicle_path),
            "vehicle_ready": self.vehicle_path.exists(),
            "boundary_model": str(self.boundary_path) if self.boundary_path else None,
            "boundary_ready": bool(self.boundary_path and self.boundary_path.exists()),
            "cat_eye_model": str(self.cat_eye_path) if self.cat_eye_path else None,
            "cat_eye_ready": bool(self.cat_eye_path and self.cat_eye_path.exists()),
        }

    def ensure_ready(self) -> None:
        status = self.status()
        missing = [k.replace("_ready", "") for k, v in status.items() if k.endswith("_ready") and not v]
        if missing:
            raise ModelFilesMissing(
                "Missing YOLO model files: " + ", ".join(missing) + ". "
                "Run scripts/download_vehicle_model.py, then train the boundary and cat-eye models."
            )

    @staticmethod
    def _load(path: Path):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from e
        return YOLO(str(path))

    def _vehicle_model(self):
        if self._vehicle is None:
            self._vehicle = self._load(self.vehicle_path)
        return self._vehicle

    def _boundary_model(self):
        if self._boundary is None:
            assert self.boundary_path is not None
            self._boundary = self._load(self.boundary_path)
        return self._boundary

    def _cat_eye_model(self):
        if self._cat_eye is None:
            assert self.cat_eye_path is not None
            self._cat_eye = self._load(self.cat_eye_path)
        return self._cat_eye

    @staticmethod
    def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        return mask > 0.5

    def vehicle(self, image: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> SegDetection | None:
        h, w = image.shape[:2]
        result = self._vehicle_model().predict(image, conf=conf, imgsz=imgsz, verbose=False, device="cpu")[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        center = np.array([w / 2.0, h / 2.0])
        diag = max(np.linalg.norm(np.array([w, h])), 1.0)
        candidates: list[tuple[float, SegDetection]] = []

        for i, box in enumerate(result.boxes):
            cls = int(box.cls.item())
            if cls not in VEHICLE_CLASS_IDS:
                continue
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            center_score = 1.0 - min(float(np.linalg.norm(box_center - center) / diag), 1.0)
            area_score = min(area / max(w * h, 1), 1.0)
            score = 0.62 * area_score + 0.38 * center_score

            if masks is not None and i < len(masks):
                mask = self._resize_mask(masks[i], (h, w))
            else:
                mask = np.zeros((h, w), dtype=bool)
                mask[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = True

            candidates.append((score, SegDetection(cls, result.names.get(cls, str(cls)), confidence, mask, (x1, y1, x2, y2))))

        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    def boundaries(self, image: np.ndarray, conf: float = 0.30, imgsz: int = 768) -> list[SegDetection]:
        h, w = image.shape[:2]
        result = self._boundary_model().predict(image, conf=conf, imgsz=imgsz, verbose=False, device="cpu")[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        out: list[SegDetection] = []
        for i, box in enumerate(result.boxes):
            cls = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            if masks is not None and i < len(masks):
                mask = self._resize_mask(masks[i], (h, w))
            else:
                mask = np.zeros((h, w), dtype=bool)
                mask[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = True
            out.append(SegDetection(cls, BOUNDARY_CLASSES.get(cls, result.names.get(cls, str(cls))), confidence, mask, (x1, y1, x2, y2)))
        return out

    def cat_eyes(self, image: np.ndarray, conf: float = 0.30, imgsz: int = 960) -> list[BoxDetection]:
        result = self._cat_eye_model().predict(image, conf=conf, imgsz=imgsz, verbose=False, device="cpu")[0]
        if result.boxes is None:
            return []
        out: list[BoxDetection] = []
        for box in result.boxes:
            cls = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            out.append(BoxDetection(cls, result.names.get(cls, "cat_eye"), confidence, (x1, y1, x2, y2)))
        return out
