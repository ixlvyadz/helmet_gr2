"""Evaluate the detector AND the rider-level helmet decision on the test split.

1) Detection metrics (precision, recall, mAP50, mAP50-95) per class via Ultralytics.
2) Rider-level compliance metrics. Ground truth is built from your label files
   with the SAME association rule used at inference, so this measures how well
   the full chain (detect rider + detect helmet + associate) works.
   Positive class = violation (rider without helmet).

Example:
  python evaluate.py --weights runs/helmet_gr2/weights/best.pt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from associate import associate_helmets, iou, rider_on_motorcycle
from utils import IMG_EXT, class_ids

MOTO, RIDER, HELMET = 0, 1, 2


def read_gt(path, W, H):
    boxes = {MOTO: [], RIDER: [], HELMET: []}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            x, y, w, h = map(float, p[1:5])
            if c in boxes:
                boxes[c].append(np.array([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]))
    return boxes


def rider_status(boxes, head_frac):
    """Return (rider_boxes, [True if helmeted, ...])."""
    riders = [r for r in boxes[RIDER] if rider_on_motorcycle(r, boxes[MOTO])]
    assoc = associate_helmets(riders, boxes[HELMET], head_frac)
    return riders, [i in assoc for i in range(len(riders))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--helmet-conf", type=float, default=0.3)
    ap.add_argument("--head-frac", type=float, default=0.4)
    a = ap.parse_args()

    model = YOLO(a.weights)
    global MOTO, RIDER, HELMET
    MOTO, RIDER, HELMET = class_ids(model.names)

    print("== Detection metrics ==")
    m = model.val(data=a.data, split=a.split, imgsz=a.imgsz, verbose=False)
    print(f"all classes: P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        print(f"  {m.names[int(c)]:11s} P={p:.3f} R={r:.3f} mAP50={ap50:.3f} mAP50-95={ap:.3f}")

    print("\n== Rider-level compliance (positive = violation) ==")
    img_dir = Path(a.root) / "images" / a.split
    lab_dir = Path(a.root) / "labels" / a.split
    tp = fp = fn = tn = missed = extra = 0
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_riders, gt_helm = rider_status(read_gt(lab_dir / (img_path.stem + ".txt"), W, H), a.head_frac)

        res = model.predict(img, conf=min(a.conf, a.helmet_conf), imgsz=a.imgsz, verbose=False)[0]
        pb = {MOTO: [], RIDER: [], HELMET: []}
        for b, c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                           res.boxes.conf.cpu().numpy()):
            thr = a.helmet_conf if c == HELMET else a.conf
            if c in pb and p >= thr:
                pb[c].append(b)
        pr_riders, pr_helm = rider_status(pb, a.head_frac)

        # greedy match predicted riders to GT riders by IoU >= 0.5
        pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gt_riders)
                        for pi, p in enumerate(pr_riders)), reverse=True)
        used_g, used_p = set(), set()
        for s, gi, pi in pairs:
            if s < 0.5 or gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            gt_v, pr_v = not gt_helm[gi], not pr_helm[pi]
            tp += gt_v and pr_v
            fp += (not gt_v) and pr_v
            fn += gt_v and (not pr_v)
            tn += (not gt_v) and (not pr_v)
        for gi in range(len(gt_riders)):
            if gi not in used_g:
                missed += 1
                fn += not gt_helm[gi]  # a missed violating rider is a missed violation
        for pi in range(len(pr_riders)):
            if pi not in used_p:
                extra += 1
                fp += not pr_helm[pi]  # an invented rider flagged as violation

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    print(f"TP={tp} FP={fp} FN={fn} TN={tn} | riders missed={missed} extra={extra}")
    print(f"violation precision={prec:.3f} recall={rec:.3f} F1={f1:.3f} accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
