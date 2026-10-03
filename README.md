# Parking Compliance Detector Using YOLO

This project detects parking boundaries, vehicles, and cat-eye reflectors from multiple images of the same parked vehicle and determines whether the parking situation is regular, irregular, or undetermined.

I designed the system as a hybrid computer-vision pipeline. YOLO models provide visual detections and segmentation masks, while deterministic geometry and evidence-based aggregation are used to answer the parking-compliance questions.

The final application runs locally, requires no external API, and performs inference on CPU.

---

## 1. Problem Definition

For each vehicle group, the system answers three questions:

- **Q1:** Are parking boundaries sufficiently visible?
- **Q2:** Is the vehicle correctly positioned relative to the visible parking boundaries?
- **Q3:** Are cat-eye reflectors present?

The final result is one of:

- `REGULAR`
- `IRREGULAR`
- `UNDETERMINED`

Q2 is only meaningful when Q1 establishes usable parking-boundary evidence.

The system considers:

- white parking-space lines;
- blue or blue/black allowed curbs;
- yellow/black forbidden curbs;
- prohibited white diagonal hatching;
- rows of cat-eye reflectors that may define a parking boundary.

---

## 2. Dataset

The dataset contains **95 unique images** representing **22 vehicle groups**.

Each vehicle can have several images taken from different viewpoints. For example:

```text
VI_1448_121803_001.jpeg
VI_1448_121803_002.jpeg
VI_1448_121803_003.jpeg
```

These images all belong to:

```text
VI_1448_121803
```

I treat all views of the same vehicle as one group.

---

## 3. Train, Validation, and Test Split

I used a group-aware split rather than randomly splitting individual images.

| Split | Images |
|---|---:|
| Train | 64 |
| Validation | 17 |
| Test | 14 |
| **Total** | **95** |

The target was approximately 70% training, 15% validation, and 15% test.

The exact percentages differ slightly because I never split images belonging to the same vehicle across different partitions.

After selecting the final split, I kept it fixed across all boundary-model experiments so the experiments could be compared fairly.

---

## 4. Boundary Classes

The boundary segmentation model uses four classes:

```text
0 - white_line
1 - allowed_curb
2 - forbidden_curb
3 - hatched_area
```

The training split contains:

| Class | Instances |
|---|---:|
| white_line | 11 |
| allowed_curb | 20 |
| forbidden_curb | 6 |
| hatched_area | 4 |

The training dataset is highly imbalanced.

Of the 64 training images, approximately **44 are background images** without an annotated positive parking-boundary class.

`hatched_area` is particularly rare, so the available example group was kept in the training split.

The class imbalance and large number of background images motivated several of the experiments described below.

---

## 5. System Architecture

The final system contains three visual perception components.

### 5.1 Vehicle Segmentation

I use a pretrained YOLO11n-seg model:

```text
models/vehicle/yolo11n-seg.pt
```

It detects the main vehicle and supplies the vehicle mask and bounding box used by the geometric reasoning stage.

### 5.2 Parking-Boundary Segmentation

I use a custom YOLO11n-seg model trained for:

- `white_line`
- `allowed_curb`
- `forbidden_curb`
- `hatched_area`

The final model is:

```text
models/boundary/best.pt
```

This is the Experiment 3 model.

### 5.3 Cat-Eye Detection

Cat-eye reflectors are very small objects, so I trained a separate YOLO11n object detector for:

```text
cat_eye
```

The final model is:

```text
models/cat_eye/best.pt
```

---

## 6. Why Q2 Is Not Another YOLO Class

I did not train YOLO to directly predict `regular` or `irregular`.

Whether a car is correctly parked depends on the spatial relationship between the vehicle and the parking boundaries.

For example, the system must determine whether:

- the vehicle crosses a white line;
- the vehicle remains inside a parking bay;
- the vehicle is correctly positioned behind an allowed curb;
- the vehicle overlaps a forbidden curb;
- the vehicle enters a hatched area.

YOLO performs the visual perception, while deterministic geometry evaluates these spatial relationships.

This approach also makes the result more traceable because the system can explain which detected evidence caused the final decision.

---

# 7. Boundary Model Experiments

Because the dataset is small and imbalanced, I performed three main experiments.

I kept the same validation split throughout the experiments.

---

## 7.1 Experiment 1 - Baseline YOLO11n-seg

Experiment 1 was the baseline boundary segmentation model.

Main settings:

```text
Image size: 768
Batch size: 4
Epochs: 60
Training images: 64
Validation images: 17
```

The training used horizontal flipping, translation, scaling, HSV augmentation, and mosaic augmentation.

### Validation Results

| Class | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|
| All | 0.498 | 0.253 |
| white_line | 0.000 | 0.000 |
| allowed_curb | 0.500 | 0.263 |
| forbidden_curb | 0.995 | 0.497 |

Experiment 1 showed that the model could learn the curb classes, but it did not generalize to the validation white-line examples.

This motivated Experiment 2.

---

## 7.2 Experiment 2 - Positive-Image Oversampling

The original training split contains many background images compared with positive boundary examples.

I therefore tested positive-image oversampling.

Each positive training image was duplicated once.

The effective training set increased from:

```text
64 images
```

to:

```text
84 images
```

The 84 images consisted of the original 64 images plus 20 duplicated positive examples.

I also used more conservative augmentation.

The goal was to increase the model's exposure to positive boundary examples and improve recall.

### Validation Results

| Class | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|
| All | 0.331 | 0.206 |
| white_line | 0.000 | 0.000 |
| allowed_curb | 0.444 | 0.233 |
| forbidden_curb | 0.551 | 0.385 |

Although some recall behavior improved, overall segmentation performance decreased and white-line detection remained unsuccessful on validation.

I therefore did not select Experiment 2.

---

## 7.3 Experiment 3 - Parking-Domain Pretraining

For Experiment 3, I investigated whether parking-specific pretraining could improve performance.

I initialized YOLO11n-seg using a checkpoint from the **ParkScope** parking-related project and then fine-tuned it on my own four parking-boundary classes.

I returned to the original frozen split:

```text
Training: 64 images
Validation: 17 images
```

I did not use the positive-image duplication from Experiment 2.

The purpose was to test whether features learned from parking scenes would transfer better to this task.

### Validation Results

| Class | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.533 | 0.318 | 0.534 | 0.335 |
| white_line | 0.116 | 0.289 | 0.103 | 0.010 |
| allowed_curb | 0.484 | 0.667 | 0.670 | 0.446 |
| forbidden_curb | 1.000 | 0.000 | 0.828 | 0.547 |

Experiment 3 achieved the best overall segmentation performance.

It was also the first experiment to achieve non-zero validation mAP50 for `white_line`.

White-line validation mAP50 changed from:

```text
Experiment 1: 0.000
Experiment 2: 0.000
Experiment 3: 0.103
```

Allowed-curb performance also improved substantially.

I therefore selected Experiment 3 as the final parking-boundary model.

---

## 8. Experiment Comparison

| Experiment | Main Change | Overall Mask mAP50 | Mask mAP50-95 | White-Line mAP50 | Allowed-Curb mAP50 |
|---|---|---:|---:|---:|---:|
| Experiment 1 | Baseline | 0.498 | 0.253 | 0.000 | 0.500 |
| Experiment 2 | Positive-image oversampling | 0.331 | 0.206 | 0.000 | 0.444 |
| **Experiment 3** | **ParkScope domain pretraining** | **0.534** | **0.335** | **0.103** | **0.670** |

Experiment 3 is used by the final application.

---

## 9. Cat-Eye Detector

I trained a separate YOLO11n detector because cat-eye reflectors occupy very small regions in the image.

The best validation results were approximately:

```text
Precision:  0.752
Recall:     0.329
mAP50:      0.408
mAP50-95:   0.200
```

Precision was stronger than recall.

Because small cat eyes can easily be missed, I use cat-eye absence conservatively rather than automatically treating a missed detection as definite proof that cat eyes do not exist.

---

## 10. Q1 - Parking-Boundary Visibility

Q1 determines whether enough parking-boundary information is visible to reason about the parking position.

Positive evidence can come from:

- white parking lines;
- an allowed curb;
- a forbidden curb;
- a hatched area;
- a sufficiently structured row of cat-eye reflectors.

A single isolated or heavily cropped marking is not automatically treated as sufficient evidence.

---

## 11. Q2 - Vehicle Position

Q2 is evaluated only after Q1 establishes usable boundary evidence.

The geometry module evaluates situations such as:

- white-line crossing;
- a white boundary continuing underneath the projected vehicle footprint;
- correct positioning behind an allowed curb;
- interaction with a forbidden curb;
- interaction with a hatched region;
- positioning relative to cat-eye boundary evidence;
- positioning between two white parking lines.

An allowed curb is **not required** for a parking position to be correct.

If two white parking boundaries form a valid bay and the vehicle is positioned between them without crossing either boundary, Q2 can be positive.

If the vehicle crosses a valid parking boundary, Q2 can be negative.

---

## 12. Q3 - Cat-Eye Presence

Q3 determines whether cat-eye reflectors are present.

The detector provides individual cat-eye detections, and geometric reasoning can also determine whether multiple detections form a meaningful row.

Because the detector has limited recall, the system avoids treating every missing detection as strong negative evidence.

---

## 13. Multi-View Processing

Each vehicle can have several images.

A single image may hide important evidence because of:

- occlusion;
- perspective;
- cropping;
- shadows;
- distance;
- camera angle.

I therefore analyze every available view individually before producing the group result.

---

## 14. Why I Do Not Use Majority Voting

I do not use simple majority voting across images.

Different views contain different amounts of useful information.

For example, several views may be unclear while one view clearly shows the vehicle crossing a parking boundary.

In that case, the decisive view should not be cancelled simply because more images were unclear.

The aggregation stage is therefore evidence-based rather than vote-based.

For Q1 and Q3, decisive positive evidence can establish presence.

For Q2, only views in which Q1 has established usable boundary evidence are used.

This makes the final decision better suited to multi-view parking inspection.

---

## 15. Final Decision Logic

The basic final logic is:

```text
Q1 = positive
Q2 = positive
        |
        v
     REGULAR
```

```text
Q1 = positive
Q2 = negative
        |
        v
    IRREGULAR
```

If the visual evidence is insufficient:

```text
UNDETERMINED
```

The system intentionally avoids forcing a regular or irregular decision when there is not enough reliable evidence.

---

## 16. Validation and Test Usage

I used:

- the training split to fit model parameters;
- the validation split to compare experiments and choose the final model;
- the held-out test split for additional evaluation and diagnostic checks.

The test split contains 14 images.

During development, I inspected some test examples while debugging the complete end-to-end system. Because of this, I do not present repeated development observations on those images as a completely untouched final benchmark.

The external evaluator remains the strongest source of unseen evaluation data.

---

## 17. Known Limitations

The largest limitation is the small dataset.

Some classes contain very few positive examples, especially:

- white lines;
- forbidden curbs;
- hatched areas.

White-line performance improved with Experiment 3 but remains weaker than allowed-curb detection.

The cat-eye detector also has limited recall because cat eyes are very small objects.

Another limitation is that not every valid parking configuration exists in the dataset.

For example, the geometry supports a car correctly positioned between two white lines without an allowed curb, but the available dataset does not contain a clear real example of that exact configuration.

Therefore, I do not claim real-dataset validation for that specific case.

---

## 18. CPU-Only and Offline Inference

The final application performs inference using CPU.

The vehicle, boundary, and cat-eye model prediction calls explicitly use:

```python
device="cpu"
```

No external inference API is required.

The `/health` endpoint reports:

```text
runtime: YOLO / CPU
external_api: false
```

This allows the application to run locally and offline once its Python dependencies and model files are installed.

---

## 19. Local Installation and Run (Without Docker)

These steps are only required when running the application directly with Python. They are **not required when using Docker**.

Python 3.11 is recommended.

Create a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8001
```

Open:

```text
http://localhost:8001
```

Health check:

```text
http://localhost:8001/health
```

---

## 20. Required Models

The application expects:

```text
models/
  vehicle/
    yolo11n-seg.pt
  boundary/
    best.pt
  cat_eye/
    best.pt
```

The final:

```text
models/boundary/best.pt
```

contains the Experiment 3 model.

---

## 21. Docker Deployment

Docker handles Python and package installation inside the container, so no local virtual environment or `pip install` step is required.

Build and start the Docker application:

```bash
docker compose up --build
```

The container runs the application internally on port 8000 and maps it to port 8001 on the host.

Open:

```text
http://localhost:8001
```

Health check:

```text
http://localhost:8001/health
```

I verified the application successfully inside Docker using CPU inference.

---

## 22. Project Structure

```text
parking_marker_yolo/
|
|-- app/
|   |-- api.py
|   |-- pipeline.py
|   |-- aggregation.py
|   |-- core/
|   |   |-- grouping.py
|   |   |-- schemas.py
|   |   `-- splitting.py
|   `-- vision/
|       |-- geometry.py
|       `-- yolo_models.py
|
|-- prepared_data/
|   |-- boundary/
|   |   |-- images/
|   |   |   |-- train/
|   |   |   |-- val/
|   |   |   `-- test/
|   |   |-- labels/
|   |   |   |-- train/
|   |   |   |-- val/
|   |   |   `-- test/
|   |   `-- boundary.yaml
|   |
|   |-- cat_eye/
|   |   |-- images/
|   |   |   |-- train/
|   |   |   |-- val/
|   |   |   `-- test/
|   |   |-- labels/
|   |   |   |-- train/
|   |   |   |-- val/
|   |   |   `-- test/
|   |   `-- cat_eye.yaml
|   |
|   |-- split_manifest.csv
|   `-- duplicates.csv
|
|-- models/
|   |-- vehicle/
|   |   `-- yolo11n-seg.pt
|   |-- boundary/
|   |   `-- best.pt
|   `-- cat_eye/
|       `-- best.pt
|
|-- scripts/
|   |-- first_run.py
|   |-- download_vehicle_model.py
|   |-- predict_folder.py
|   |-- train_boundary.py
|   |-- train_boundary_exp2.py
|   |-- train_boundary_exp3_parkscope.py
|   `-- train_cat_eyes.py
|
|-- metadata/
|   |-- split_manifest.csv
|   `-- duplicates.csv
|
|-- THIRD_PARTY_LICENSES/
|   `-- ParkScope_LICENSE
|
|-- input_dataset/
|   `-- README.txt
|
|-- web/
|   `-- index.html
|
|-- tests/
|   |-- test_aggregation.py
|   `-- test_grouping_and_split.py
|
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- requirements-dev.txt
|-- ANNOTATION_GUIDE.md
`-- README.md
```

---

## 23. Reproducing the Experiments

### Experiment 1

```bash
python scripts/train_boundary.py --device mps --epochs 60 --imgsz 768
```

### Experiment 2

```bash
python scripts/train_boundary_exp2.py --device mps
```

### Experiment 3

The ParkScope checkpoint used for Experiment 3 should be available at:

```text
models/parkscope/parkscope_best.pt
```

Run:

```bash
python scripts/train_boundary_exp3_parkscope.py --device mps --batch 4
```

On systems without Apple MPS, another supported training device can be selected.

The deployed application still performs inference on CPU.

---

## 24. Final Model Selection

I selected Experiment 3 because it achieved:

```text
Overall mask mAP50:     0.534
Overall mask mAP50-95:  0.335
Allowed-curb mAP50:     0.670
White-line mAP50:       0.103
```

It achieved the strongest overall validation result and was the only boundary experiment to produce non-zero validation mAP50 for white-line detection.

For these reasons, Experiment 3 became the final boundary model.

---

## 25. Summary

The final system combines:

- YOLO11n-seg vehicle segmentation;
- custom parking-boundary segmentation;
- a dedicated cat-eye detector;
- deterministic geometric reasoning;
- multi-view evidence aggregation;
- CPU-only inference;
- FastAPI;
- Docker deployment.

Instead of directly training a single regular/irregular classifier on a very small dataset, I separate visual perception from spatial reasoning.

This provides a more interpretable result and allows the application to expose the evidence behind Q1, Q2, Q3, and the final parking diagnosis.

---

## 26. Third-Party Attribution

Experiment 3 used pretrained weights from the ParkScope project as parking-domain initialization before fine-tuning on my parking-boundary dataset.

ParkScope repository:
https://github.com/Nanasaki-Ai/ParkScope

The original ParkScope license is included with the submission under `THIRD_PARTY_LICENSES/ParkScope_LICENSE`.

The ParkScope model was not used directly for the final parking decisions. I fine-tuned the model on my own boundary classes, and the final application combines the resulting detections with my deterministic geometry and multi-view aggregation logic.

---

## 27. Annotation Process

I manually annotated the dataset using **CVAT (Computer Vision Annotation Tool)**.

For the parking-boundary segmentation task, I labeled the following classes:

```text
white_line
allowed_curb
forbidden_curb
hatched_area
```

For the cat-eye detection task, I labeled:

```text
cat_eye
```

I exported the annotations in YOLO-compatible format and used them to create the final train, validation, and test datasets.

I kept the same group-aware split for both the boundary and cat-eye tasks so that images of the same vehicle did not appear in different dataset partitions.

