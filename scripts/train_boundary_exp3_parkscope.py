from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "prepared_data" / "boundary" / "boundary.yaml"
PARKSCOPE = ROOT / "models" / "parkscope" / "parkscope_best.pt"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")

    args = parser.parse_args()

    if not PARKSCOPE.exists():
        raise FileNotFoundError(
            f"ParkScope model not found: {PARKSCOPE}"
        )

    print("Loading ParkScope pretrained model:")
    print(PARKSCOPE)

    model = YOLO(str(PARKSCOPE))

    print("ParkScope model loaded.")
    print("Fine-tuning on our boundary classes...")

    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=12,
        seed=42,

        project=str(ROOT / "runs"),
        name="boundary_seg_exp3_parkscope",

        # Match Experiment 1 settings
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        translate=0.10,
        scale=0.50,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    best = Path(model.trainer.best)

    output = (
        ROOT
        / "models"
        / "boundary"
        / "exp3_parkscope_best.pt"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(best, output)

    print()
    print("✅ Experiment 3 completed")
    print(f"Best model copied to: {output}")


if __name__ == "__main__":
    main()
