# Reproducibility Notes

This file records the benchmark protocol and reported results used in the revised manuscript and response letter. It is intended to make the repository easier to audit and reproduce where data access is available.

## Environment

Recommended setup:

```bash
conda create -n cfyolo python=3.10 -y
conda activate cfyolo
pip install -r requirements.txt
```

If your CUDA version requires a specific PyTorch build, install PyTorch from the official PyTorch index first, then install the remaining packages.

## Model and Dataset Files

CF-YOLO model YAML:

```text
ultralytics/cfg/models/11/yolo11_CPAM_backbone_FARM.yaml
```

Default CTDD dataset YAML:

```text
ultralytics/cfg/datasets/defect.yaml
```

NEU-DET single-class external-validation dataset YAML:

```text
ultralytics/cfg/datasets/neu-det-single.yaml
```

Edit the `path` field in `defect.yaml` to point to the local CTDD root.

## Training

Default CF-YOLO training:

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

Short form:

```bash
python train.py
```

The output checkpoint is saved under `runs/detect/<run-name>/weights/`.

## Validation

```bash
python test.py \
  --weights runs/detect/cf_yolo_ctdd/weights/best.pt \
  --data ultralytics/cfg/datasets/defect.yaml \
  --device 0
```

Prediction on images:

```bash
python test.py \
  --weights runs/detect/cf_yolo_ctdd/weights/best.pt \
  --source path/to/images \
  --device 0
```

## CTDD Main Comparison

The table below follows the revised CTDD no-leak protocol. The performance claim is limited to these evaluated baselines on CTDD.

| Model | P | AP50 | AP95 | AP75 | mIoU | F1 | F0.5 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cascade R-CNN | 0.594 | 0.732 | 0.479 | 0.334 | 0.755 | 0.661 | 0.619 | 0.596 |
| Faster R-CNN | 0.500 | 0.746 | 0.471 | 0.265 | 0.613 | 0.537 | 0.737 | 0.553 |
| CenterNet | 0.853 | 0.667 | 0.404 | 0.221 | 0.753 | 0.720 | 0.798 | 0.631 |
| ATSS | 0.832 | 0.483 | 0.402 | 0.081 | 0.706 | 0.638 | 0.471 | 0.516 |
| RetinaNet | 0.448 | 0.684 | 0.445 | 0.152 | 0.704 | 0.568 | 0.489 | 0.499 |
| Dynamic R-CNN | 0.547 | 0.695 | 0.450 | 0.259 | 0.731 | 0.644 | 0.582 | 0.558 |
| FoveaNet | 0.479 | 0.659 | 0.428 | 0.131 | 0.705 | 0.577 | 0.514 | 0.499 |
| ETDNet | 0.675 | 0.788 | 0.560 | 0.411 | 0.775 | 0.653 | 0.666 | 0.647 |
| Conditional DETR | 0.546 | 0.582 | 0.268 | 0.291 | 0.663 | 0.600 | 0.566 | 0.502 |
| RF-DETR | 0.843 | 0.794 | 0.441 | 0.412 | 0.789 | 0.790 | 0.778 | 0.692 |
| RT-DETRv2 | 0.801 | 0.787 | 0.435 | 0.413 | 0.782 | 0.770 | 0.766 | 0.679 |
| YOLOv11n | 0.843 | 0.801 | 0.464 | 0.429 | 0.780 | 0.771 | 0.806 | 0.699 |
| CF-YOLO | 0.882 | 0.823 | 0.488 | 0.455 | 0.769 | 0.796 | 0.846 | 0.723 |

## External Validation: NEU-DET

NEU-DET was converted to single-class bounding-box detection by merging its six categories into `defect`.

| Dataset | Model | AP50 | AP50-95 | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| NEU-DET | YOLOv11n | 0.786 | 0.468 | 0.712 | 0.774 | 0.660 |
| NEU-DET | CF-YOLO | 0.794 | 0.465 | 0.732 | 0.760 | 0.707 |

Reported split:

| Item | Value |
|---|---:|
| Images | 1,800 |
| Defect instances | 4,189 |
| Train images | 1,260 |
| Validation images | 270 |
| Test images | 270 |
| Seed | 0 |

Example command after converting NEU-DET to one `defect` class:

```bash
python train.py \
  --data ultralytics/cfg/datasets/neu-det-single.yaml \
  --model ultralytics/cfg/models/11/yolo11_CPAM_backbone_FARM.yaml \
  --name cf_yolo_neu_det_seed0 \
  --seed 0
```

## Component Ablation and Efficiency

| Model | AP50 | AP50-95 | F1 | Precision | Recall | Params | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv11n | 0.750 | 0.465 | 0.725 | 0.780 | 0.678 | 2.59M | 6.44 | 119 |
| CPAM-only | 0.768 | 0.466 | 0.742 | 0.810 | 0.685 | 17.29M | 87.07 | 95 |
| FARM-only | 0.735 | 0.461 | 0.712 | 0.775 | 0.658 | 2.86M | 6.99 | 113 |
| CF-YOLO | 0.823 | 0.488 | 0.796 | 0.882 | 0.710 | 17.56M | 87.23 | 85 |

## FARM Variant Ablation

| Variant | AP50 | AP50-95 | F1 | Precision |
|---|---:|---:|---:|---:|
| Full FARM | 0.823 | 0.488 | 0.796 | 0.882 |
| No spatial branch | 0.772 | 0.467 | 0.748 | 0.825 |
| No channel branch | 0.785 | 0.476 | 0.765 | 0.825 |
| Conv-only | 0.775 | 0.472 | 0.752 | 0.815 |

## FARM Placement Ablation

| Configuration | AP50 | AP50-95 | F1 | Params | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FARM in neck | 0.823 | 0.488 | 0.796 | 17.56M | 87.23 | 85 |
| FARM before neck | 0.760 | 0.469 | 0.738 | 17.56M | 87.68 | 82 |
| FARM after neck | 0.770 | 0.474 | 0.748 | 17.65M | 88.29 | 78 |

## CPAM Placement and Depth Ablation

| Configuration | AP50 | AP50-95 | F1 | Params | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|
| CPAM mid-backbone | 0.823 | 0.488 | 0.796 | 17.56M | 87.23 | 85 |
| CPAM shallow | 0.730 | 0.462 | 0.705 | 6.36M | 78.72 | 84 |
| CPAM deep | 0.718 | 0.460 | 0.695 | 14.04M | 15.82 | 113 |
| Fewer CPAM blocks | 0.770 | 0.468 | 0.745 | 16.85M | 51.75 | 70 |

## CPAM Dual-Kernel Ablation

`K_L` denotes the Large-Kernel Perception branch. `K_S` denotes the Small-Kernel Aggregation branch.

| K_S | K_L | Precision | F1 | AP50 |
|---:|---:|---:|---:|---:|
| 3 | 3 | 0.823 | 0.703 | 0.751 |
| 3 | 5 | 0.854 | 0.786 | 0.804 |
| 3 | 7 | 0.846 | 0.785 | 0.814 |
| 5 | 7 | 0.564 | 0.494 | 0.462 |
| 5 | 11 | 0.850 | 0.784 | 0.809 |
| 11 | 11 | 0.844 | 0.744 | 0.783 |
| 7 | 11 | 0.882 | 0.796 | 0.823 |

The adopted setting is `K_S = 7` and `K_L = 11`.

## Repeat-Run Stability

| Run | AP50 | AP50-95 | F1 |
|---|---:|---:|---:|
| YOLOv11n main | 0.750 | 0.465 | 0.725 |
| YOLOv11n seed=0 | 0.747 | 0.464 | 0.723 |
| CF-YOLO main | 0.823 | 0.488 | 0.796 |
| CF-YOLO seed=0 | 0.765 | 0.471 | 0.740 |

## Metrics

Metric definitions are listed in `DATASET.md`. In brief, `AP95` is AP at IoU = 0.95, and `mIoU` is bounding-box IoU, not segmentation-mask IoU.

## Scope Notes

- This project does not claim unrestricted state-of-the-art performance across all industrial defect or camouflaged object benchmarks.
- CTDD is a copper-tube micro-defect dataset with one detection class.
- Edge deployment and quantized models are future optimization targets.
- YOLO-MSD, Detail-Enhanced YOLO, AFF-YOLO, and YOLO-RFF were not included in same-protocol quantitative CTDD tables because public source code, trained weights, or complete reproduction settings were not available to us.
