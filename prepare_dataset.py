"""Merge datasets into one YOLO dataset with classes: motorcycle, rider, helmet.

1. Edit SOURCES below: one entry per dataset folder, with a class map
   {their_class_id: our_class_id}. Their classes not listed are DROPPED
   (this is how no_helmet / unwanted classes get removed).
2. Run:  python prepare_dataset.py

Our classes: 0 motorcycle, 1 rider, 2 helmet
The split is done by GROUP (same video / same original image stay together)
so near-duplicate frames never leak between train and test.
"""
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from utils import iter_images, label_path_for

SOURCES = [
    # (folder, {their_id: our_id})
    ("datasets/kaggle_helmet", {0: 2}),  # example: their 'helmet'(0) -> helmet(2); everything else dropped
    ("datasets/own_footage", {0: 0, 1: 1, 2: 2}),  # your own labelled/auto-labelled frames
]
OUT = Path("dataset")
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42
NAMES = ["motorcycle", "rider", "helmet"]


def group_key(stem):
    """Frames of one video or augmented copies of one photo share a group.

    Roboflow copies:  photo_jpg.rf.<hash>  -> photo_jpg
    Video frames:     clip3_frame0042 / clip3_f42 -> clip3
    Adjust the regex if your filenames differ.
    """
    return re.sub(r"(\.rf\..*|_frame.*|_f\d+.*)$", "", stem)


def main():
    random.seed(SEED)
    items, skipped = [], 0
    for si, (src, cmap) in enumerate(SOURCES):
        src = Path(src)
        if not src.exists():
            print(f"[warn] {src} not found, skipping")
            continue
        for img in iter_images(src):
            lab = label_path_for(img)
            if not lab.exists():
                skipped += 1  # no label file at all = unlabeled, do not use
                continue
            lines = []
            for line in lab.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                if c in cmap:
                    lines.append(" ".join([str(cmap[c])] + p[1:5]))
            group = f"s{si}_{group_key(img.stem)}"
            items.append((img, lines, group, f"s{si}_{img.name}"))

    if not items:
        raise SystemExit("No labelled images found. Check SOURCES.")

    groups = sorted({g for _, _, g, _ in items})
    random.shuffle(groups)
    n = len(groups)
    n_tr, n_va = int(n * SPLIT[0]), int(n * SPLIT[1])
    split_of = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, g in enumerate(groups)}

    if OUT.exists():
        shutil.rmtree(OUT)
    per_split, per_class = Counter(), {s: Counter() for s in ("train", "val", "test")}
    for img, lines, group, new_name in items:
        s = split_of[group]
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, OUT / "images" / s / new_name)
        (OUT / "labels" / s / (Path(new_name).stem + ".txt")).write_text("\n".join(lines))
        per_split[s] += 1
        for l in lines:
            per_class[s][NAMES[int(l.split()[0])]] += 1

    yaml = f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    yaml += f"nc: {len(NAMES)}\nnames: {NAMES}\n"
    (OUT / "data.yaml").write_text(yaml)

    print(f"groups: {n} | skipped (no label file): {skipped}")
    for s in ("train", "val", "test"):
        print(f"{s:5s} images={per_split[s]:5d} boxes={dict(per_class[s])}")
    print(f"wrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
