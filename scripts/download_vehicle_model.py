from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "models" / "vehicle" / "yolo11n-seg.pt"
out.parent.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n-seg.pt")
src = Path(getattr(model, "ckpt_path", "yolo11n-seg.pt"))
if src.resolve() != out.resolve():
    shutil.copy2(src, out)
print(f"Vehicle model ready: {out}")
