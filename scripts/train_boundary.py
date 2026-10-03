from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "prepared_data" / "boundary" / "boundary.yaml"


def count_nonempty_labels() -> int:
    return sum(1 for p in (ROOT / "prepared_data" / "boundary" / "labels" / "train").glob("*.txt") if p.read_text().strip())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0", help="0 for GPU, cpu for CPU")
    args = p.parse_args()

    if not DATA.exists():
        raise SystemExit("Run scripts/first_run.py first")
    if count_nonempty_labels() == 0:
        raise SystemExit("No non-empty boundary segmentation labels found. See ANNOTATION_GUIDE.md")

    model = YOLO("yolo11n-seg.pt")
    result = model.train(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, patience=12, seed=42, project=str(ROOT / "runs"), name="boundary_seg",
    )
    best = Path(model.trainer.best)
    out = ROOT / "models" / "boundary" / "best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"Best boundary model copied to: {out}")


if __name__ == "__main__":
    main()
