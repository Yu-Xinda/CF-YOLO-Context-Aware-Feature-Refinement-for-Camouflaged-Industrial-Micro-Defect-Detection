# CF-YOLO: Context-Aware Feature Refinement for Camouflaged Industrial Micro-Defect Detection

This repository contains the public implementation of **CF-YOLO: Context-Aware Feature Refinement for Camouflaged Industrial Micro-Defect Detection**.

CF-YOLO targets single-class bounding-box detection of weak, background-similar industrial micro-defects. The model modifies the YOLO11 detection pipeline with:

1. **Context-Perception Aggregation Module (CPAM)**: a large-kernel perception branch generates context-dependent aggregation weights that guide local small-kernel aggregation.
2. **Feature Additive Refinement Module (FARM)**: a linear-complexity additive token-mixing refinement module inserted into the detection pipeline before multi-scale prediction.

The repository provides the implementation code, model configuration, training/evaluation scripts, environment requirements, and benchmark documentation needed to reproduce the reported CTDD protocol where data access is available.

<p align="center">
  <img src="docs/model.png" width="800" alt="CF-YOLO architecture">
</p>

## Repository Contents

- `ultralytics/nn/modules/modify/CPAM.py`: CPAM implementation.
- `ultralytics/nn/modules/FARM.py`: FARM implementation.
- `ultralytics/cfg/models/11/yolo11_CPAM_backbone_FARM.yaml`: CF-YOLO model configuration.
- `ultralytics/cfg/datasets/defect.yaml`: CTDD-style single-class detection dataset configuration.
- `ultralytics/cfg/datasets/neu-det-single.yaml`: NEU-DET single-class external-validation dataset template.
- `train.py`: portable training entry point.
- `test.py`: validation and prediction entry point.
- `DATASET.md`: CTDD dataset description, access note, split format, and metric definitions.
- `REPRODUCIBILITY.md`: commands, protocol details, and reported CTDD/NEU-DET/ablation results.
- `requirements.txt` and `requirement.txt`: Python dependencies.
- `LICENSE`: AGPL-3.0 license.

## Installation

```bash
git clone https://github.com/Yu-Xinda/CF-YOLO-Context-Aware-Feature-Refinement-for-Camouflaged-Industrial-Micro-Defect-Detection.git
cd CF-YOLO-Context-Aware-Feature-Refinement-for-Camouflaged-Industrial-Micro-Defect-Detection

conda create -n cfyolo python=3.10 -y
conda activate cfyolo
pip install -r requirements.txt
```

PyTorch and CUDA compatibility depends on the local machine. If needed, install the PyTorch build matching your CUDA version before installing the remaining requirements.

## Data Preparation

CTDD is an annotated single-class copper-tube defect benchmark with **1,847 images** and **4,898 bounding-box defect instances**. The dataset contains industrial production data, so the full image data and some acquisition details are not hosted directly in this repository.

Researchers may request access for academic research use by contacting:

```text
2025388032@stu.zjhu.edu.cn
```

Please include your affiliation and a short description of the intended use. After obtaining or preparing the dataset, organize it as:

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

Then edit `path` in `ultralytics/cfg/datasets/defect.yaml` if your CTDD root differs from `data/CTDD`.

More dataset and metric details are in `DATASET.md`.

## Training

```bash
python train.py \
  --data ultralytics/cfg/datasets/defect.yaml \
  --model ultralytics/cfg/models/11/yolo11_CPAM_backbone_FARM.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --seed 0
```

The script uses the CF-YOLO model YAML and CTDD-style dataset YAML by default, so the short form is:

```bash
python train.py
```

## Evaluation and Prediction

Validate a trained checkpoint:

```bash
python test.py --weights runs/detect/cf_yolo_ctdd/weights/best.pt
```

Run prediction on images or a folder:

```bash
python test.py \
  --weights runs/detect/cf_yolo_ctdd/weights/best.pt \
  --source path/to/images
```

## Main CTDD Results

The table reports the CTDD no-leak protocol used in the revised manuscript. Claims are limited to the evaluated methods on CTDD.

| Model | P | AP50 | AP95 | AP75 | mIoU | F1 | F0.5 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cascade R-CNN | 0.594 | 0.732 | 0.479 | 0.334 | 0.755 | 0.661 | 0.619 | 0.596 |
| Faster R-CNN | 0.500 | 0.746 | 0.471 | 0.265 | 0.613 | 0.537 | 0.737 | 0.553 |
| CenterNet | 0.853 | 0.667 | 0.404 | 0.221 | 0.753 | 0.720 | 0.798 | 0.631 |
| ATSS | 0.832 | 0.483 | 0.402 | 0.081 | 0.706 | 0.638 | 0.471 | 0.516 |
| RetinaNet | 0.448 | 0.684 | 0.445 | 0.152 | 0.704 | 0.568 | 0.489 | 0.499 |
| Dynamic R-CNN | 0.547 | 0.695 | 0.450 | 0.259 | 0.731 | 0.644 | 0.582 | 0.558 |
| FoveaNet | 0.479 | 0.659 | 0.428 | 0.131 | 0.705 | 0.577 | 0.514 | 0.499 |
| ETDNet | 0.675 | 0.788 | **0.560** | 0.411 | 0.775 | 0.653 | 0.666 | 0.647 |
| Conditional DETR | 0.546 | 0.582 | 0.268 | 0.291 | 0.663 | 0.600 | 0.566 | 0.502 |
| RF-DETR | 0.843 | 0.794 | 0.441 | 0.412 | **0.789** | 0.790 | 0.778 | 0.692 |
| RT-DETRv2 | 0.801 | 0.787 | 0.435 | 0.413 | 0.782 | 0.770 | 0.766 | 0.679 |
| YOLOv11n | 0.843 | 0.801 | 0.464 | 0.429 | 0.780 | 0.771 | 0.806 | 0.699 |
| CF-YOLO | **0.882** | **0.823** | 0.488 | **0.455** | 0.769 | **0.796** | **0.846** | **0.723** |

`AP95` denotes AP at IoU = 0.95, not COCO-style AP@[.50:.95]. `mIoU` is box-level IoU between predicted and ground-truth bounding boxes, not mask-level segmentation IoU. See `DATASET.md` for metric definitions and `REPRODUCIBILITY.md` for additional NEU-DET and ablation results.

## Data and Code Availability

The code, model configuration, scripts, environment files, and benchmark-use documentation are available in this repository. CTDD image data require access approval because the dataset contains industrial production data. Public external validation uses NEU-DET, which can be prepared in single-class YOLO detection format with `ultralytics/cfg/datasets/neu-det-single.yaml` as described in `REPRODUCIBILITY.md`.

## Notes on Reproduction Scope

- CF-YOLO is evaluated as a single-class bounding-box detector.
- The word "camouflaged" describes weak target-background contrast and texture similarity in copper-tube inspection; CTDD is not positioned as a generic COD benchmark.
- Edge deployment remains a future optimization target. This repository reports model complexity and FPS where measured, but it does not provide a dedicated quantized edge-deployment model.
- YOLO-MSD, Detail-Enhanced YOLO, AFF-YOLO, and YOLO-RFF were not included in same-protocol CTDD quantitative comparisons because public source code, trained weights, or complete reproduction settings were not available to us.

## Acknowledgement

This project builds on the open-source [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO codebase. The inherited Ultralytics code and this derivative repository are distributed under the AGPL-3.0 license.
