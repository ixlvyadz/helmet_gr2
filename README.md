# GR2 – Motorcycle Helmet Detection & Compliance Monitoring (code walkthrough)

Classes (fixed, in this order): `0 motorcycle`, `1 rider`, `2 helmet`.
There is **no `no_helmet` class** (group decision). A violation = a rider on a
motorcycle with no helmet associated over the life of their track.

## Files

| File | What it does |
|---|---|
| `utils.py` | Finds images and their YOLO label files (handles Roboflow/YOLO/labelImg layouts) |
| `associate.py` | The core logic: rider-on-motorcycle check and rider-to-helmet association |
| `prepare_dataset.py` | Merges datasets, remaps/drops classes, splits train/val/test by group |
| `auto_label.py` | Pre-labels motorcycle/rider (COCO model) and helmet (your model) so you only fix errors |
| `train.py` | Fine-tunes YOLO on the dataset built by `prepare_dataset.py` |
| `merge_annotations.py` | **Merges the team's Roboflow zips** (nested zips, different class orders) into one clean dataset with leak-free train/val/test splits |
| `train_elvie.py` | **Trains directly on a Roboflow-annotated dataset** (train / val / test folders): checks labels, writes the data yaml, trains YOLO26, evaluates on test |
| `run_pipeline.py` | Detect → track → associate → decide → log evidence |
| `evaluate.py` | Detection metrics + rider-level violation metrics |

## 0. Setup
```bash
pip install -r requirements.txt
```
Use a GPU if you can (Google Colab / Kaggle notebooks are fine). On CPU use `--model yolov8n.pt --epochs 30`.

## Quick path: the team's annotations are already done (one zip with a zip per member)
```bash
pip install -r requirements.txt
python merge_annotations.py --zip Complete_Annotations.zip --out dataset
python train_elvie.py --root dataset
```
`merge_annotations.py` unzips everything, converts every member's labels to one class order
(0 motorcycle, 1 rider, 2 helmet; Elvie's export used a different order than the rest) and rebuilds
train/val/test **without leakage**. All images are consecutive frames of only 3 CCTV videos, so the team's
own random split put val/test frames 1 frame away from training frames (about 90-97% of them within 5 frames).
Default `--split-mode video` holds one whole video out as test. Other modes: `block`, `original`.

### Single dataset path (train_elvie.py on its own)
```bash
python train_elvie.py --root path/to/dataset
```
The folder must contain `train`, `val` (or `valid`) and `test`, each with `images/` and `labels/`
(Roboflow export, format YOLO). If the folder has no `data.yaml`, add
`--names motorcycle rider helmet` in the same order as your label ids.
Class names must be `motorcycle`, `rider`, `helmet` (order does not matter: `run_pipeline.py`
and `evaluate.py` look the ids up by name). The script prints a label check first and stops if
labels are missing. Then continue at step 6 (run the pipeline) with `runs/helmet_gr2/weights/best.pt`.

## 1. Gather datasets (only needed if you build the dataset yourself)
Put each dataset in `datasets/<name>/` (any layout where labels are YOLO `.txt` files:
`images/x.jpg` + `labels/x.txt`, Roboflow `train/images` + `train/labels`, or image and txt side by side).
Export from Roboflow/CVAT in **YOLO format**; bounding boxes only (no polygons).

Edit `SOURCES` at the top of `prepare_dataset.py`. Each entry maps *their* class id to *ours*;
ids you don't list are dropped. Example: a dataset with `0 = helmet, 1 = no helmet` becomes `{0: 2}`,
so its no-helmet boxes disappear automatically.

## 2. Add motorcycle and rider labels (most public sets only have helmets)
```bash
python auto_label.py --src datasets/kaggle_helmet
```
This adds `motorcycle` and `rider` boxes from a COCO model (a person overlapping a
motorcycle becomes a rider). Then **review** the folder in labelImg / Label Studio / CVAT:
delete wrong boxes, add missed ones. Reviewing is far faster than drawing from scratch.
Do the same for your own footage (extract frames first, e.g. every 10th frame).

## 3. Build the dataset
```bash
python prepare_dataset.py
```
It prints image and box counts per split. Check that `helmet` and `rider` counts look sensible
and that test is not empty. Splitting is by group (same video / same photo stays together)
so train and test never share near-identical frames. If your filenames do not hold a video id,
edit `group_key()`.

## 4. Train
```bash
python train.py --epochs 50 --device 0
```
Best weights: `runs/helmet_gr2/weights/best.pt`. If far riders are missed, retry with `--imgsz 960`.

## 5. Improve with your own footage (optional but worth it)
Run `auto_label.py --src datasets/own_footage --helmet-weights runs/helmet_gr2/weights/best.pt`,
fix the boxes, rebuild with `prepare_dataset.py`, retrain. Collect extra examples of whatever
the model gets wrong (night, passengers, helmets in hand).

## 6. Run the full system on video
```bash
python run_pipeline.py --weights runs/helmet_gr2/weights/best.pt \
    --source clips/street1.mp4 --start-time 2026-11-08T08:00:00 --save-video
```
Outputs in `outputs/`: `violations.csv`, `evidence/` (rider crop + annotated frame), `annotated.mp4`.
Use `--source 0` for a webcam.

### How it works, step by step
1. **Detection + tracking**: `model.track()` runs YOLO with ByteTrack, giving every motorcycle and rider a persistent ID.
2. **Rider filter**: a rider counts only if it overlaps a motorcycle (drops pedestrians).
3. **Association** (`associate.py`): a helmet can belong to a rider only if its centre is inside the rider box,
   in the **top 40%** of it, and mostly contained in it. This rejects helmets held in hand or on the handlebar.
   Matching is one-to-one, so a passenger's helmet is not credited to the driver.
4. **Per-track memory**: for each rider ID, count frames seen and frames with a helmet; keep the clearest (largest) view as evidence.
5. **Decision when the track ends** (rider gone for 30 frames, or video ends):
   helmet ratio ≤ 0.3 → `violation`, ≥ 0.6 → `compliant`, otherwise or too few/small frames → `uncertain`.
6. **Logging**: one CSV row per rider track (no duplicates), with timestamps, frame count, helmet ratio,
   rider/helmet confidences, and the evidence image path.

### Parameters to tune
| Flag | Default | Effect |
|---|---|---|
| `--conf` | 0.4 | Motorcycle/rider confidence |
| `--helmet-conf` | 0.3 | Kept **lower** so missed helmets (false violations) are rarer |
| `--head-frac` | 0.4 | How far down the rider box a helmet may sit |
| `--min-frames` | 10 | Frames needed before judging a rider |
| `--min-height` | 60 | Rider box height in px; smaller = too far to judge |
| `--violation-max` / `--compliant-min` | 0.3 / 0.6 | Helmet-ratio cut-offs |
| `--lost-frames` | 30 | How long a track may vanish before it is closed |

## 7. Evaluate
```bash
python evaluate.py --weights runs/helmet_gr2/weights/best.pt
```
Prints per-class precision, recall, mAP50 and mAP50-95, then rider-level violation
precision, recall, F1 and accuracy (positive = violation).

**Tracking/logging evaluation (do by hand on 3–5 clips):** count the real riders and violators,
then compare with `violations.csv`. Report duplicate records per violator and ID switches you observe.
Use the `annotated.mp4` to check these quickly.

## Common problems
- **Helmeted riders flagged as violations**: lower `--helmet-conf` (0.2–0.25), raise `--head-frac` slightly, add more far/night helmet images.
- **Pedestrians logged as riders**: raise `--conf` or add pedestrians near motorcycles as negatives.
- **Same rider logged twice**: the tracker is losing the ID; raise `--lost-frames`, try `--tracker botsort.yaml`, or increase `--imgsz`.
- **Everything is `uncertain`**: riders are too small/short-lived; lower `--min-height`/`--min-frames` or use a closer camera angle.
- **Test score looks too good**: you probably have near-duplicate frames across splits; check `group_key()`.
