# CTDD Dataset Documentation

## Overview

The Copper Tube Defect Dataset (CTDD) is used in this work as a single-class bounding-box detection benchmark for copper-tube surface micro-defects. It is designed for industrial cases where defect regions are small and visually similar to the surrounding surface texture.

Current benchmark summary:

| Item | Value |
|---|---:|
| Images | 1,847 |
| Bounding-box defect instances | 4,898 |
| Detection classes | 1 |
| Class name | `defect` |
| Annotation type | YOLO-format bounding boxes |
| Task | Single-class object detection |

## Access

CTDD contains industrial production data. The full images are therefore not hosted directly in this repository, and some acquisition details are not publicly disclosed.

For academic research use, request access by email:

```text
2025388032@stu.zjhu.edu.cn
```

Please include:

- Name and institution.
- Intended academic use.
- Whether redistribution or derivative release is planned.

## Expected Directory Structure

After access is granted or after preparing a compatible dataset, organize the files as:

```text
data/CTDD/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each image should have a same-stem `.txt` label file in YOLO detection format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to image width and height. CTDD uses one class:

```text
0 defect
```

The default dataset YAML is:

```text
ultralytics/cfg/datasets/defect.yaml
```

If the dataset is stored outside this repository, edit the `path` field in that YAML file.

## Split Protocol

The revised manuscript and response letter refer to the CTDD no-leak protocol. The repository records the benchmark-use protocol but does not expose production-line identifiers that may reveal confidential acquisition information.

Recommended reporting fields for each split are:

| Split | Required information |
|---|---|
| Train | Image count, instance count, class definition, annotation format |
| Validation | Image count, instance count, class definition, annotation format |
| Test | Image count, instance count, class definition, annotation format |

Where available, ensure that samples from the same tube, production batch, or acquisition sequence do not leak across training and test sets. Some production-line metadata are confidential and are not released in this repository.

## Public External Validation: NEU-DET

For external validation, the revised response uses NEU-DET as a public industrial surface-defect dataset. Because CF-YOLO is formulated as single-class bounding-box detection, the six original NEU-DET categories are merged into one `defect` class.

Reported NEU-DET setting:

| Item | Value |
|---|---:|
| Images | 1,800 |
| Defect instances | 4,189 |
| Train images | 1,260 |
| Validation images | 270 |
| Test images | 270 |
| Split seed | 0 |
| Classes after conversion | 1 (`defect`) |

## Metric Definitions

CTDD is evaluated as bounding-box detection.

- `AP50`: average precision at IoU = 0.50.
- `AP75`: average precision at IoU = 0.75.
- `AP95`: average precision at IoU = 0.95. This is not COCO-style AP@[.50:.95].
- `Precision`: TP / (TP + FP).
- `Recall`: TP / (TP + FN).
- `F1`: harmonic mean of Precision and Recall.
- `F0.5`: F-score that weights Precision more heavily than Recall.
- `mIoU`: box-level IoU between predicted bounding boxes and ground-truth bounding boxes. It is not a pixel-level segmentation metric.
- `Avg`: descriptive composite score used for table-level comparison only. Main conclusions should be based on individual metrics such as AP50, Precision, F1, and F0.5.

## Confidentiality Notes

The following categories may be partially withheld or generalized because they involve industrial production details:

- Imaging device details.
- Production-line acquisition protocol.
- Lighting hardware and precise setup.
- Factory, tube batch, or acquisition-sequence metadata.
- Ambiguous samples that may expose production context.

The repository provides the dataset structure, class definition, annotation format, split-use instructions, metric definitions, and reproduction commands needed to understand the benchmark protocol.
