from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "prepared_data" / "boundary"
EXP = ROOT / "prepared_data" / "boundary_exp2"


def copy_image_and_label(image: Path, label: Path, out_images: Path, out_labels: Path):
    shutil.copy2(image, out_images / image.name)
    shutil.copy2(label, out_labels / label.name)


def prepare_dataset():
    """
    Experiment 2:
    - exact same original train/val split
    - validation remains completely unchanged
    - every positive training image is duplicated once
    - background images are not duplicated

    YOLO will independently augment the copies during training.
    """

    if EXP.exists():
        shutil.rmtree(EXP)

    for split in ["train", "val"]:
        (EXP / "images" / split).mkdir(parents=True, exist_ok=True)
        (EXP / "labels" / split).mkdir(parents=True, exist_ok=True)

    # --------------------------
    # TRAIN
    # --------------------------
    train_images = SOURCE / "images" / "train"
    train_labels = SOURCE / "labels" / "train"

    originals = 0
    positives = 0
    duplicates = 0

    for image in sorted(train_images.iterdir()):
        if (
            not image.is_file()
            or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}
        ):
            continue

        label = train_labels / f"{image.stem}.txt"

        if not label.exists():
            raise RuntimeError(f"Missing label for {image.name}")

        # Original
        copy_image_and_label(
            image,
            label,
            EXP / "images" / "train",
            EXP / "labels" / "train",
        )
        originals += 1

        # Duplicate each positive image exactly once
        if label.read_text().strip():
            positives += 1

            copy_stem = image.stem + "__poscopy"
            copy_image = EXP / "images" / "train" / (
                copy_stem + image.suffix.lower()
            )
            copy_label = EXP / "labels" / "train" / (
                copy_stem + ".txt"
            )

            shutil.copy2(image, copy_image)
            shutil.copy2(label, copy_label)
            duplicates += 1

    # --------------------------
    # VAL — unchanged
    # --------------------------
    val_images = SOURCE / "images" / "val"
    val_labels = SOURCE / "labels" / "val"

    val_count = 0

    for image in sorted(val_images.iterdir()):
        if (
            not image.is_file()
            or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}
        ):
            continue

        label = val_labels / f"{image.stem}.txt"

        if not label.exists():
            raise RuntimeError(f"Missing validation label for {image.name}")

        copy_image_and_label(
            image,
            label,
            EXP / "images" / "val",
            EXP / "labels" / "val",
        )
        val_count += 1

    data = {
        "path": str(EXP),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "white_line",
            1: "allowed_curb",
            2: "forbidden_curb",
            3: "hatched_area",
        },
    }

    with open(EXP / "boundary_exp2.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("\nExperiment 2 dataset:")
    print(f"  Original train images : {originals}")
    print(f"  Positive train images : {positives}")
    print(f"  Added positive copies : {duplicates}")
    print(f"  Total train images    : {originals + duplicates}")
    print(f"  Validation images     : {val_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    prepare_dataset()

    model = YOLO("yolo11n-seg.pt")

    model.train(
        data=str(EXP / "boundary_exp2.yaml"),

        # Same core experiment settings
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=12,
        seed=42,

        project=str(ROOT / "runs"),
        name="boundary_seg_exp2",

        # ----------------------------------
        # Conservative parking augmentation
        # ----------------------------------
        fliplr=0.5,
        flipud=0.0,

        degrees=2.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,

        hsv_h=0.005,
        hsv_s=0.30,
        hsv_v=0.25,

        # Default was 1.0; use mosaic less often
        mosaic=0.30,

        mixup=0.0,
        copy_paste=0.0,
    )

    best = Path(model.trainer.best)

    out = ROOT / "models" / "boundary" / "exp2_best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best, out)

    print(f"\n✅ Experiment 2 model copied to: {out}")


if __name__ == "__main__":
    main()
