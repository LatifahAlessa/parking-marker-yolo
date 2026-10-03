# Annotation guide

The custom YOLO models require labels. Images alone cannot train the parking-marker classes.

## Boundary segmentation model (`yolo11n-seg`)

Use polygons and these exact class IDs:

| ID | Class | Label this |
|---:|---|---|
| 0 | `white_line` | white painted parking-space boundary lines |
| 1 | `allowed_curb` | blue or blue/black curb / wheel-stop indicating allowed parking |
| 2 | `forbidden_curb` | yellow or yellow/black curb indicating prohibited parking |
| 3 | `hatched_area` | white diagonal hatched / prohibited painted area |

YOLO segmentation line format:

`class_id x1 y1 x2 y2 x3 y3 ...`

All coordinates are normalized to 0–1.

## Cat-eye detector (`yolo11n`)

Use bounding boxes and one class:

| ID | Class |
|---:|---|
| 0 | `cat_eye` |

YOLO detection line format:

`0 x_center y_center width height`

All coordinates are normalized to 0–1.

## Important labeling rules

- Label only visible evidence. Do not hallucinate hidden parts behind a vehicle.
- Keep the same interpretation across day/night images.
- Do **not** label car reflections or round bolts as cat eyes.
- Negative/background images should have an empty `.txt` file.
- Never move different views of the same vehicle to different splits. `scripts/first_run.py` already prevents this.
- Do not tune thresholds on the test split. Use train for fitting, validation for tuning, and test only for final evaluation.

You can annotate with CVAT, Roboflow, Label Studio, or another tool that exports YOLO segmentation/detection labels.
