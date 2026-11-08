"""Pre-label images so you only have to CORRECT boxes, not draw them.

* A COCO model adds `motorcycle`, and `rider` (a person overlapping a motorcycle).
* Optionally, your own trained model (--helmet-weights) adds `helmet`.
* Existing labels are kept; new boxes that duplicate an existing one are skipped.

Example:
  python auto_label.py --src datasets/own_footage --helmet-weights runs/helmet_gr2/weights/best.pt
Then open the folder in labelImg / Label Studio / CVAT and fix mistakes.
Classes: 0 motorcycle, 1 rider, 2 helmet
"""
import argparse

import numpy as np
from ultralytics import YOLO

from associate import iou, rider_on_motorcycle
from utils import iter_images, label_path_for

MOTO, RIDER, HELMET = 0, 1, 2
COCO_PERSON, COCO_MOTORCYCLE = 0, 3


def read_labels(path):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c = int(float(p[0]))
                x, y, w, h = map(float, p[1:5])
                out.append((c, np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])))
    return out


def to_line(c, b):
    x1, y1, x2, y2 = np.clip(b, 0, 1)
    return f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with images (recursive)")
    ap.add_argument("--coco-weights", default="yolov8m.pt")
    ap.add_argument("--helmet-weights", default=None,
                    help="your trained model; any class named 'helmet' is added as helmet")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--helmet-conf", type=float, default=0.3)
    args = ap.parse_args()

    coco = YOLO(args.coco_weights)
    hel = YOLO(args.helmet_weights) if args.helmet_weights else None

    n_img = n_added = 0
    for img in iter_images(args.src):
        lab = label_path_for(img)
        existing = read_labels(lab)
        new = []

        r = coco.predict(str(img), conf=args.conf, verbose=False)[0]
        boxes = r.boxes.xyxyn.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        motos = [b for b, c in zip(boxes, classes) if c == COCO_MOTORCYCLE]
        persons = [b for b, c in zip(boxes, classes) if c == COCO_PERSON]
        new += [(MOTO, m) for m in motos]
        new += [(RIDER, p) for p in persons if rider_on_motorcycle(p, motos)]

        if hel is not None:
            rh = hel.predict(str(img), conf=args.helmet_conf, verbose=False)[0]
            for b, c in zip(rh.boxes.xyxyn.cpu().numpy(), rh.boxes.cls.cpu().numpy().astype(int)):
                if str(hel.names[c]).lower() == "helmet":
                    new.append((HELMET, b))

        lines = [to_line(c, b) for c, b in existing]
        for c, b in new:
            if any(ec == c and iou(eb, b) > 0.5 for ec, eb in existing):
                continue
            existing.append((c, b))
            lines.append(to_line(c, b))
            n_added += 1

        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text("\n".join(lines))
        n_img += 1

    print(f"processed {n_img} images, added {n_added} boxes")


if __name__ == "__main__":
    main()
