"""Merge the team's Roboflow YOLO exports (nested zips, mixed layouts, mixed class orders)
into ONE clean dataset that train_elvie.py can train on.

Usage:
  python merge_annotations.py --zip Complete_Annotations.zip --out dataset
  python train_elvie.py --root dataset

What it fixes automatically
  * Unzips zips inside the zip (one per team member) and finds every images/labels pair.
  * CLASS ORDER differs between members (most: helmet, motorcycle, rider; Elvie: motorcycle,
    rider, helmet). Each folder is converted to ONE order: 0 motorcycle, 1 rider, 2 helmet.
    The order is read from each data.yaml (folders without one use Roboflow's alphabetical
    order) and double-checked against box shapes (helmets are the smallest boxes, riders the
    tallest); a warning is printed if they disagree.
  * LEAKAGE: all images are frames cut from a few CCTV videos. Random frame-level splits put
    neighbouring frames (1 frame apart) in train and test, which makes test scores look far
    better than they are. --split-mode decides how to split instead:
      video     (default) one whole video is held out as TEST; VALIDATION is made of separate
                blocks of the remaining videos, with a gap so train never touches val.
      block     all videos are cut into blocks of consecutive frames, blocks go to
                train/val/test, with a gap between different splits.
      original  keep the team's own Train/Val/Test folders (leaky; only for comparison).
"""
import argparse
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from utils import IMG_EXT, class_ids, label_path_for

CANON = ["motorcycle", "rider", "helmet"]            # output order: ids 0, 1, 2
DEFAULT_ORDER = ["helmet", "motorcycle", "rider"]    # Roboflow alphabetical order (no data.yaml)
SPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
VID = re.compile(r"(XVR_ch\d+_main_\d+_\d+_\d+-\d+_mp4)-(\d+)_jpg", re.I)


def extract_all(zip_path, work):
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)
    changed = True
    while changed:                                   # zips inside zips
        changed = False
        for z in sorted(work.rglob("*.zip")):
            d = z.with_suffix("")
            if "__MACOSX" in z.parts or d.exists():
                continue
            d.mkdir(parents=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
            changed = True


def find_yaml(img, root):
    for parent in img.parents:
        y = parent / "data.yaml"
        if y.exists():
            return y
        if parent == root:
            break
    return None


def build_map(yaml_path):
    """{their_id: our_id} + their cleaned class names."""
    names = DEFAULT_ORDER
    if yaml_path:
        names = yaml.safe_load(yaml_path.read_text()).get("names", DEFAULT_ORDER)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
    cleaned = [re.sub(r"^\s*\d+[\s:_.-]*", "", str(n)).strip().lower() for n in names]  # '0 motorcycle' -> 'motorcycle'
    moto, rider, helmet = class_ids(cleaned)
    return {moto: 0, rider: 1, helmet: 2}, cleaned


def infer_map(shapes):
    """Guess the class order from box shapes: helmet = smallest, rider = tallest, motorcycle = rest."""
    if len(shapes) != 3:
        return None
    area = {c: sum(w * h for w, h in v) / len(v) for c, v in shapes.items()}
    helmet = min(area, key=area.get)
    rest = [c for c in shapes if c != helmet]
    asp = {c: sum(h / max(w, 1e-6) for w, h in shapes[c]) / len(shapes[c]) for c in rest}
    rider = max(asp, key=asp.get)
    moto = [c for c in rest if c != rider][0]
    return {moto: 0, rider: 1, helmet: 2}


def path_info(img, root):
    parts = img.relative_to(root).parts
    person = "src"
    for i, p in enumerate(parts[:-1]):
        if p.lower() in ("train", "val", "test"):      # outer Train/Val/Test folder
            person = parts[i + 1] if i + 1 < len(parts) - 1 else "src"
            break
    orig = "train"
    for p in reversed(parts[:-1]):                      # innermost split-like folder wins
        if p.lower() in SPLIT_NAMES:
            orig = SPLIT_NAMES[p.lower()]
            break
    return person, orig


def assign_blocks(frames, block, gap, pattern):
    """frames: items of ONE video sorted by frame. Whole blocks go to a split; train images
    within `gap` positions of a val/test image are dropped so splits never touch."""
    n = len(frames)
    for i, it in enumerate(frames):
        it["split"] = pattern(i // block)
    for i, it in enumerate(frames):
        if it["split"] == "train":
            lo, hi = max(0, i - gap), min(n, i + gap + 1)
            if any(frames[j]["split"] != "train" for j in range(lo, hi)):
                it["drop"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="the annotations zip")
    ap.add_argument("--src", help="already-extracted folder (instead of --zip)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--work", default="_extracted")
    ap.add_argument("--split-mode", choices=["video", "block", "original"], default="video")
    ap.add_argument("--test-video", type=int, default=None,
                    help="video index to hold out as test (video mode); default = smallest video")
    ap.add_argument("--block", type=int, default=40, help="images per block")
    ap.add_argument("--gap", type=int, default=8, help="train images this close to val/test are dropped")
    a = ap.parse_args()

    if a.zip:
        work = Path(a.work).resolve()
        if work.exists():
            shutil.rmtree(work)
        extract_all(a.zip, work)
    elif a.src:
        work = Path(a.src).resolve()
        extract_all_nested = [z for z in work.rglob("*.zip")]
        if extract_all_nested:
            print("[note] --src contains zips; use --zip so nested zips are extracted")
    else:
        raise SystemExit("give --zip or --src")

    # ---- read every image + label, convert class ids to the common order ----
    items, shapes_by_person, maps = [], defaultdict(lambda: defaultdict(list)), {}
    skipped = poly = bad = 0
    for img in sorted(p for p in work.rglob("*") if p.suffix.lower() in IMG_EXT and "__MACOSX" not in p.parts):
        lp = label_path_for(img)
        if not lp.exists():
            skipped += 1
            continue
        person, orig = path_info(img, work)
        yml = find_yaml(img, work)
        key = (person, str(yml))
        if key not in maps:
            maps[key] = build_map(yml)
        cmap, cleaned = maps[key]
        rows = []
        for line in lp.read_text().splitlines():
            t = line.split()
            if not t:
                continue
            if len(t) != 5:
                poly += 1 if len(t) > 5 else 0
                continue
            c = int(float(t[0]))
            if c not in cmap:
                bad += 1
                continue
            rows.append((cmap[c], t[1], t[2], t[3], t[4]))
            shapes_by_person[person][c].append((float(t[3]), float(t[4])))
        m = VID.search(img.name)
        video, frame = (m.group(1), int(m.group(2))) if m else (person, len(items))
        items.append({"img": img, "person": person, "orig": orig, "video": video,
                      "frame": frame, "rows": rows, "split": orig, "drop": False})
    if not items:
        raise SystemExit("No labelled images found.")

    # ---- sanity check of the class order per folder ----
    print("class order check (their yaml order -> our ids):")
    for (person, _), (cmap, cleaned) in sorted(maps.items()):
        guess = infer_map(shapes_by_person[person])
        flag = "" if guess is None or guess == cmap else "   <-- [WARN] box shapes suggest a different order, check this folder!"
        print(f"  {person:24s} names={cleaned}{flag}")

    # ---- decide splits ----
    videos = defaultdict(list)
    for it in items:
        videos[it["video"]].append(it)
    for v in videos.values():
        v.sort(key=lambda it: it["frame"])
    vnames = sorted(videos)
    print(f"\nvideos found: {len(vnames)}")
    for i, v in enumerate(vnames):
        print(f"  [{i}] {v[-44:]}  images={len(videos[v])}")

    if a.split_mode == "video":
        if len(vnames) < 2:
            raise SystemExit("video mode needs at least 2 source videos; use --split-mode block")
        ti = a.test_video if a.test_video is not None else min(range(len(vnames)), key=lambda i: len(videos[vnames[i]]))
        print(f"test video = [{ti}] (held out completely)")
        for i, v in enumerate(vnames):
            if i == ti:
                for it in videos[v]:
                    it["split"] = "test"
            else:
                assign_blocks(videos[v], a.block, a.gap, lambda b: "val" if b % 7 == 3 else "train")
    elif a.split_mode == "block":
        for v in vnames:
            assign_blocks(videos[v], a.block, a.gap,
                          lambda b: "val" if b % 7 == 3 else ("test" if b % 7 == 5 else "train"))

    # ---- write the dataset ----
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    kept = [it for it in items if not it["drop"]]
    count, boxes = Counter(), {s: Counter() for s in ("train", "val", "test")}
    for it in kept:
        s = it["split"]
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
        name = f"{it['person']}__{it['img'].name}"
        shutil.copy2(it["img"], out / s / "images" / name)
        (out / s / "labels" / (Path(name).stem + ".txt")).write_text(
            "\n".join(" ".join(map(str, r)) for r in it["rows"]))
        count[s] += 1
        for r in it["rows"]:
            boxes[s][CANON[r[0]]] += 1
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "train/images", "val": "val/images",
         "test": "test/images", "nc": 3, "names": CANON}, sort_keys=False))

    # ---- report ----
    print(f"\nsplit mode: {a.split_mode} | skipped (no label file): {skipped} | "
          f"dropped as buffer: {len(items) - len(kept)} | unusable rows: polygon={poly} bad id={bad}")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} images={count[s]:5d} boxes={dict(boxes[s])}")
    print("\nleakage check (frames between each val/test image and the nearest TRAIN frame of the same video):")
    for s in ("val", "test"):
        gaps = []
        for v in vnames:
            tr = sorted(it["frame"] for it in videos[v] if it["split"] == "train" and not it["drop"])
            for it in videos[v]:
                if it["split"] == s and not it["drop"]:
                    gaps.append(min([abs(it["frame"] - f) for f in tr] or [10 ** 6]))
        if gaps:
            gaps.sort()
            med = gaps[len(gaps) // 2]
            print(f"  {s:5s} median gap = {'n/a (no train frames from that video)' if med >= 10**6 else str(med) + ' frames'};"
                  f" within 5 frames of a train frame: {sum(g <= 5 for g in gaps)}/{len(gaps)}")
    print(f"\nwrote {out / 'data.yaml'}  ->  next: python train_elvie.py --root {out}")


if __name__ == "__main__":
    main()
