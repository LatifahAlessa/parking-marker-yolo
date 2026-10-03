from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.grouping import IMAGE_EXTENSIONS, canonical_stem, parse_filename, sha256_file
from app.core.splitting import groupwise_split


def discover_default_source() -> Path:
    box = ROOT / "input_dataset"
    items = [p for p in box.iterdir() if not p.name.startswith(".")]
    if len(items) == 1:
        return items[0]
    if not items:
        raise SystemExit(
            "No dataset found. Put ONE dataset folder or .zip inside input_dataset/, "
            "or pass --source /path/to/dataset"
        )
    raise SystemExit("input_dataset/ contains more than one item. Pass --source explicitly.")


def resolve_source(source: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if source.is_dir():
        return source, None
    if source.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="parking_yolo_")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp.name)
        return Path(tmp.name), tmp
    raise SystemExit("--source must be a directory or .zip file")


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def label_index(root: Path, keyword: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.rglob("*.txt"):
        parent_text = str(p.parent).lower()
        if keyword in parent_text:
            out[canonical_stem(p.name)] = p
    return out


def copy_label(index: dict[str, Path], stem: str, dst: Path) -> bool:
    src = index.get(stem)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src and src.exists():
        shutil.copy2(src, dst)
        return True
    # Empty label is valid for a negative/background image. If the user has not
    # annotated yet, this also leaves the expected file in place.
    dst.write_text("", encoding="utf-8")
    return False


def write_yaml(path: Path, dataset_root: Path, names: list[str]) -> None:
    text = [
        f"path: {dataset_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    text += [f"  {i}: {name}" for i, name in enumerate(names)]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="First-run dataset grouping and group-wise 70/15/15 split")
    parser.add_argument("--source", type=Path, help="Dataset folder or .zip. If omitted, uses the only item in input_dataset/.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace an existing prepared_data directory")
    args = parser.parse_args()

    source = args.source or discover_default_source()
    source_root, tmp = resolve_source(source)
    try:
        images = find_images(source_root)
        if not images:
            raise SystemExit("No images found in the dataset")

        boundary_labels = label_index(source_root, "boundary")
        cat_labels = label_index(source_root, "cat")

        unique_images: list[tuple[Path, str, str]] = []
        duplicates: list[tuple[str, str]] = []
        seen_hash: dict[str, str] = {}
        group_sizes: Counter[str] = Counter()

        for img in images:
            try:
                parsed = parse_filename(img.name)
            except ValueError as e:
                raise SystemExit(str(e)) from e
            digest = sha256_file(img)
            if digest in seen_hash:
                duplicates.append((img.name, seen_hash[digest]))
                continue
            seen_hash[digest] = img.name
            unique_images.append((img, parsed.group_id, parsed.canonical_stem))
            group_sizes[parsed.group_id] += 1

        assignment = groupwise_split(dict(group_sizes), seed=args.seed)
        out = ROOT / "prepared_data"
        if out.exists():
            if not args.force:
                raise SystemExit("prepared_data/ already exists. Use --force to rebuild it.")
            shutil.rmtree(out)

        boundary_root = out / "boundary"
        cat_root = out / "cat_eye"
        for root in (boundary_root, cat_root):
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True, exist_ok=True)
                (root / "labels" / split).mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        copied_boundary = 0
        copied_cat = 0
        split_counts = Counter()
        split_groups: dict[str, set[str]] = defaultdict(set)

        for img, group_id, stem in unique_images:
            split = assignment[group_id]
            split_counts[split] += 1
            split_groups[split].add(group_id)
            image_name = f"{stem}{img.suffix.lower()}"

            for ds_root in (boundary_root, cat_root):
                shutil.copy2(img, ds_root / "images" / split / image_name)

            copied_boundary += int(copy_label(boundary_labels, stem, boundary_root / "labels" / split / f"{stem}.txt"))
            copied_cat += int(copy_label(cat_labels, stem, cat_root / "labels" / split / f"{stem}.txt"))
            manifest_rows.append({"filename": image_name, "group_id": group_id, "view_stem": stem, "split": split})

        write_yaml(boundary_root / "boundary.yaml", boundary_root, ["white_line", "allowed_curb", "forbidden_curb", "hatched_area"])
        write_yaml(cat_root / "cat_eye.yaml", cat_root, ["cat_eye"])

        with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "group_id", "view_stem", "split"])
            writer.writeheader()
            writer.writerows(manifest_rows)

        with (out / "duplicates.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "duplicate_of"])
            writer.writerows(duplicates)

        total = len(unique_images)
        print("\nPrepared dataset successfully")
        print(f"Source images found: {len(images)}")
        print(f"Unique images kept: {total}")
        print(f"Exact duplicates skipped: {len(duplicates)}")
        print(f"Vehicle groups: {len(group_sizes)}")
        for split in ("train", "val", "test"):
            pct = 100.0 * split_counts[split] / max(total, 1)
            print(f"{split:>5}: {split_counts[split]:>4} images ({pct:5.1f}%), {len(split_groups[split])} groups")
        print(f"Boundary label files found/copied: {copied_boundary}/{total}")
        print(f"Cat-eye label files found/copied: {copied_cat}/{total}")
        if copied_boundary == 0 or copied_cat == 0:
            print("\nIMPORTANT: YOLO cannot learn custom parking markers from images alone.")
            print("Annotate the generated train/val/test images using ANNOTATION_GUIDE.md, then place/save labels in the matching labels folders.")
        print("\nGenerated:")
        print(f"  {boundary_root / 'boundary.yaml'}")
        print(f"  {cat_root / 'cat_eye.yaml'}")
        print(f"  {out / 'split_manifest.csv'}")
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
