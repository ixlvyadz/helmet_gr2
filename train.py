"""Fine-tune YOLO on the prepared dataset (motorcycle, rider, helmet)."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained start; yolov8n.pt is faster")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640, help="use 960 if riders are small/far")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--name", default="helmet_gr2")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project="runs", name=args.name, patience=15,
        hsv_v=0.5,   # stronger brightness changes -> better day/night robustness
        scale=0.5,   # stronger size changes -> better for near/far riders
        fliplr=0.5,
    )
    print("Best weights:", Path(model.trainer.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()
