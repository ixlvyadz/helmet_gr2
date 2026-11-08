"""Train YOLO26 on your Roboflow-annotated dataset (folders: train / val(or valid) / test).

Usage (from the helmet_gr2 folder):
  python train_elvie.py --root path/to/Elvie
Optional:
  --names motorcycle rider helmet     # only if there is no data.yaml in the folder
  --model yolo26s.pt --epochs 50 --imgsz 640 --batch 16

What it does
  1. Finds each split's images and labels (Roboflow 'images/'+'labels/' subfolders, or
     labels sitting next to the images).
  2. Reads class names from the dataset's data.yaml (or --names).
  3. SANITY-CHECKS the labels (counts, class histogram, bad ids) and stops if
     nothing is labelled - a wrong path otherwise trains on zero labels without any error.
  4. Writes a fresh data yaml with absolute paths, trains, then evaluates on the test split.
"""
import argparse
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO

from utils import IMG_EXT


def find_split(root, candidates):
    for c in candidates:
        d = root / c
        if d.is_dir():
            return c, d
    return None, None


def split_dirs(d):
    """Return (images_dir, labels_dir) for one split folder."""
    if (d / "images").is_dir():
        return d / "images", d / "labels"
    return d, d  # flat: labels next to images


def load_names(root, cli_names):
    if cli_names:
        return list(cli_names)
    for f in (root / "data.yaml", root / "dataset.yaml"):
        if f.exists():
            names = yaml.safe_load(f.read_text()).get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=int)]
            if names:
                return list(names)
    raise SystemExit("No class names found. Put your Roboflow data.yaml in the dataset folder "
                     "or pass --names motorcycle rider helmet (in the SAME order as your label ids).")


def check_split(name, img_dir, lab_dir, nc):
    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    labelled = empty = bad = 0
    hist = Counter()
    for p in imgs:
        lp = lab_dir / (p.stem + ".txt")
        if not lp.exists():
            continue
        labelled += 1
        rows = [l.split() for l in lp.read_text().splitlines() if l.strip()]
        if not rows:
            empty += 1
        for r in rows:
            try:
                c = int(float(r[0]))
            except ValueError:
                bad += 1
                continue
            if 0 <= c < nc and len(r) == 5:
                hist[c] += 1
            else:
                bad += 1  # id out of range, or polygon/segmentation row
    print(f"  {name:5s} images={len(imgs):5d} with label file={labelled:5d} "
          f"(empty={empty}) boxes={sum(hist.values())} bad rows={bad}")
    return len(imgs), labelled, hist, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset folder containing train/ val/ test/")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--model", default="yolo26s.pt", help="yolo26n.pt = fastest, yolo26s.pt = good balance")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640, help="use 960 if riders are small / far")
    ap.add_argument("--batch", type=int, default=16, help="lower to 8 if you run out of GPU memory")
    ap.add_argument("--device", default=None, help="0 = first GPU, cpu, mps; default: auto")
    ap.add_argument("--name", default="helmet_gr2")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation if checks show warnings")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    names = load_names(root, a.names)
    nc = len(names)
    print(f"classes ({nc}): {dict(enumerate(names))}")

    paths, totals = {}, Counter()
    print("dataset check:")
    for key, cands in (("train", ["train"]), ("val", ["val", "valid", "validation"]), ("test", ["test"])):
        found, d = find_split(root, cands)
        if d is None:
            if key == "test":
                print("  test  (no test folder - skipping final test evaluation)")
                continue
            raise SystemExit(f"Missing '{cands[0]}' folder in {root}")
        img_dir, lab_dir = split_dirs(d)
        n_img, n_lab, hist, bad = check_split(key, img_dir, lab_dir, nc)
        if n_lab == 0:
            raise SystemExit(f"No label .txt files found for '{key}'. Expected {lab_dir}. "
                             "Re-export from Roboflow as YOLO format (images + labels), not images only.")
        paths[key] = img_dir
        totals.update({f"{key}:{names[c]}": n for c, n in hist.items()})
        if n_lab < 0.9 * n_img or bad:
            print(f"  [warn] '{key}': some images have no label file or have invalid rows")
            if not a.yes:
                if input("  continue anyway? [y/N] ").strip().lower() != "y":
                    raise SystemExit("stopped")
    print("boxes per class:", dict(totals))

    data = {"path": str(root), "train": str(paths["train"]), "val": str(paths["val"]), "names": names}
    if "test" in paths:
        data["test"] = str(paths["test"])
    yaml_path = root / "data_train.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    print("wrote", yaml_path)

    device = a.device
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print("device:", device)

    model = YOLO(a.model)
    model.train(
        data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device,
        project="runs", name=a.name, patience=15,
        hsv_v=0.5,   # strong brightness changes -> day/night robustness
        scale=0.5,   # strong size changes -> near/far riders
        fliplr=0.5,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("\nBest weights:", best)

    if "test" in paths:
        print("\n== Final evaluation on the TEST split ==")
        m = YOLO(str(best)).val(data=str(yaml_path), split="test", imgsz=a.imgsz)
        print(f"all: P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            print(f"  {m.names[int(c)]:12s} P={p:.3f} R={r:.3f} mAP50={ap50:.3f} mAP50-95={ap:.3f}")


if __name__ == "__main__":
    main()
