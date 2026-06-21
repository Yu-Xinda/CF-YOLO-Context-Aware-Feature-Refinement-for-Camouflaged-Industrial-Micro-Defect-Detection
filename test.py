"""Evaluate or run inference with a trained CF-YOLO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/defect.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CF-YOLO or run prediction on images/videos.")
    parser.add_argument("--weights", required=True, help="Path to trained weights, e.g. runs/detect/cf_yolo_ctdd/weights/best.pt.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path for validation.")
    parser.add_argument("--source", default=None, help="Optional image/video/folder source for prediction mode.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Prediction IoU threshold.")
    parser.add_argument("--project", default=str(ROOT / "runs/detect"), help="Output directory.")
    parser.add_argument("--name", default="cf_yolo_eval", help="Run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    if args.source:
        model.predict(
            source=args.source,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            save=True,
            project=args.project,
            name=args.name,
            exist_ok=True,
        )
    else:
        model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )


if __name__ == "__main__":
    main()
