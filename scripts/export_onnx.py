from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

for folder, imgsz in (("boundary", 768), ("cat_eye", 960)):
    src = ROOT / "models" / folder / "best.pt"
    if not src.exists():
        print(f"Skipping {folder}: {src} does not exist")
        continue
    exported = Path(YOLO(str(src)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True))
    out = ROOT / "models" / folder / "best.onnx"
    if exported.resolve() != out.resolve():
        shutil.copy2(exported, out)
    print(f"Exported {folder}: {out}")
