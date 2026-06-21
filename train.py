"""Train CF-YOLO with portable command-line arguments."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/11/yolo11_CPAM_backbone_FARM.yaml"
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/defect.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CF-YOLO on CTDD or another YOLO-format detection dataset.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model YAML or checkpoint path.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or comma-separated device ids.")
    parser.add_argument("--name", default="cf_yolo_ctdd", help="Run name under runs/detect.")
    parser.add_argument("--project", default=str(ROOT / "runs/detect"), help="Output directory.")
    parser.add_argument("--patience", type=int, default=50, help="Early-stopping patience.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by Ultralytics.")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional pretrained checkpoint. If omitted, the model YAML is trained with Ultralytics defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
        patience=args.patience,
        seed=args.seed,
    )
    if args.pretrained:
        train_kwargs["pretrained"] = args.pretrained
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
