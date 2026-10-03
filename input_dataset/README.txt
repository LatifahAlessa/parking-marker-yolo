Put ONE dataset folder or .zip in this directory, then run:

    python scripts/first_run.py

The script recursively finds images, groups them by filename, removes exact duplicates,
and splits whole vehicle groups approximately 70% / 15% / 15%.

Images may be unlabeled. If you already have YOLO labels, use this source layout:

source_dataset/
  images/                    # any nested layout is okay
  boundary_labels/           # YOLO segmentation .txt files
  cat_eye_labels/            # YOLO detection .txt files

Label filenames must match image stems.
