# GR2 – Helmet Detection: Complete Steps (VS Code, Windows)

Goal: train a YOLO26 model on the team's annotations (motorcycle, rider, helmet) and run the
compliance system (detection, rider-helmet association, tracking, logging, evaluation).

Project folder: `helmet_gr2` (from `helmet_gr2.zip`). Commands below are for PowerShell.
If your folder name has spaces or parentheses (for example `D:\helmet_gr2 (2)`), always put it in quotes.

---

## 0. What you need

- Python 3.10 or newer (Python 3.13 works)
- VS Code with the **Python** extension (Microsoft)
- `helmet_gr2.zip` (the project code)
- `Complete_Annotations-...zip` (the team's annotations)
- Internet connection (first run downloads libraries and pretrained weights)
- Recommended: an NVIDIA GPU, or Google Colab with a free GPU (see Section 4)

---

## 1. Open the project

1. Unzip `helmet_gr2.zip`.
2. VS Code: **File → Open Folder** → choose the `helmet_gr2` folder.
3. Copy the annotations zip into the same folder (next to `merge_annotations.py`).
4. Open the terminal: **Ctrl + `** (backtick). Make sure you are in the project folder:
   ```powershell
   cd "D:\helmet_gr2 (2)"
   dir *.zip
   ```
   You should see the annotations zip listed. If not, the next steps will fail with
   `FileNotFoundError` (see Troubleshooting).

---

## 2. Create a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

- If PowerShell blocks the script: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
- Select the interpreter: **Ctrl+Shift+P → Python: Select Interpreter →** the one in `.venv`.
- The terminal line should start with `(.venv)`.
- Use `py` instead of `python` if `python` is not recognized.

---

## 3. Install the libraries

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs `ultralytics`, `opencv-python`, `numpy`, and `pyyaml`.

### 3a. Check whether PyTorch can use your GPU

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- `True` → GPU training will work. Go to Section 5.
- `False` → see Section 4.

---

## 4. If the GPU check prints `False`

**Check 1 – do you have an NVIDIA GPU?**
```powershell
nvidia-smi
```
- Prints a table with a GPU name → you have one. The installed PyTorch is the CPU version. Do **Fix A**.
- "not recognized" / error → no NVIDIA GPU or no driver. Check Task Manager → Performance → GPU. If you only
  see Intel/AMD graphics, use **Option B**.

**Fix A – install the GPU build of PyTorch**
```powershell
python -c "import torch; print(torch.__version__)"      # a version ending in +cpu means the CPU build
pip uninstall -y torch torchvision
```
Then open https://pytorch.org/get-started/locally, choose **Windows → Pip → Python → your CUDA version**, and run
the command it shows. It has the form:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```
(`cuXXX` = CUDA version, for example `cu128`. Copy the exact command from the website.)
`nvidia-smi` shows the highest CUDA version your driver supports (top-right). Update the NVIDIA driver if it is old.
Check again:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Also confirm the terminal shows `(.venv)` and VS Code uses the `.venv` interpreter.

**Option B – no NVIDIA GPU**

1. **Google Colab (recommended).** colab.research.google.com → Runtime → Change runtime type → T4 GPU.
   Upload `helmet_gr2.zip` and the annotations zip, then run in cells:
   ```
   !unzip -q helmet_gr2.zip
   %cd helmet_gr2
   !pip install -r requirements.txt
   !python merge_annotations.py --zip "Complete_Annotations-20261108T122503Z-1-001.zip" --out dataset
   !python train_elvie.py --root dataset
   ```
   Download `runs/helmet_gr2/weights/best.pt` right after training (Colab disconnects after a while).
2. **CPU training in VS Code** (slow, use the small model):
   ```powershell
   python train_elvie.py --root dataset --model yolo26n.pt --epochs 30 --batch 8
   ```

Note: `train_elvie.py` falls back to CPU automatically when no GPU is found, without an error.

---

## 5. Merge the team's annotations

```powershell
python merge_annotations.py --zip "Complete_Annotations-20261108T122503Z-1-001.zip" --out dataset
```
If the zip is somewhere else, use its full path in quotes.

What this does:
- Unzips the zips inside the zip (one per team member) into `_extracted/`.
- **Aligns class order.** Elvie's export uses `0 motorcycle, 1 rider, 2 helmet`; the others use
  `0 helmet, 1 motorcycle, 2 rider`. Every label is converted to one order:
  `0 motorcycle, 1 rider, 2 helmet`. The class order is read from each `data.yaml`; folders without one use
  the others' order. Box shapes are checked too and a `[WARN]` is printed if a folder disagrees.
- **Fixes the leaky split.** All images are frames of only 3 CCTV videos. In the team's original split,
  about 90 percent of val and 97 percent of test images are within 5 frames of a training frame, so test
  scores would look unrealistically good. The default `--split-mode video` holds one whole video out as
  test, builds validation from separate blocks of the other videos, and drops training frames next to them.

Check the printed summary. With the default mode you should see about:

| Split | Images |
|---|---|
| train | 1,259 |
| val | 200 |
| test | 468 |

and no `[WARN]` lines. In the class order check, `Elvie_train`, `Elvie_val`, and `Elvie_test` should show
`names=['motorcycle', 'rider', 'helmet']`. `dataset/data.yaml` should list `names: [motorcycle, rider, helmet]`.

Other split modes:
```powershell
python merge_annotations.py --zip "<zip>" --out dataset --split-mode block      # blocks from all videos
python merge_annotations.py --zip "<zip>" --out dataset --split-mode original   # team's own split (leaky)
```

---

## 6. Train

```powershell
python train_elvie.py --root dataset
```

The script first prints a label check (stops if labels are missing), writes `dataset\data_train.yaml`,
trains `yolo26s.pt` for 50 epochs (stops early if validation stops improving), then evaluates on the test split.

Useful options:

| Option | When to use |
|---|---|
| `--model yolo26n.pt` | faster and lighter (CPU or weak GPU) |
| `--batch 8` (or `4`) | out-of-memory errors |
| `--imgsz 960` | riders and helmets are small / far |
| `--epochs 80` | curves still improving at epoch 50 |
| `--device cpu` | force CPU |

While it trains: keep the laptop plugged in, turn off sleep, keep the terminal open.
If it is interrupted, resume from `runs\helmet_gr2\weights\last.pt`.

---

## 7. Read the results

Everything is saved in `runs\helmet_gr2\`:
- `weights\best.pt` – the model to use
- `results.png` – loss and mAP curves (losses should fall, mAP should rise then flatten)
- `confusion_matrix.png` – which classes get confused (watch for missed helmets: a missed helmet becomes a
  false violation in this system)

The terminal prints precision, recall, mAP50, and mAP50-95 per class on the test split at the end. Copy them
for the paper. Because test is a video the model never saw, these are your honest results.

How to improve weak results:
- Low helmet recall → `--imgsz 960` or `--model yolo26m.pt`
- High training mAP but low validation mAP → overfitting; fewer epochs or more varied images
- Many rider false positives → check for pedestrians labeled as riders

---

## 8. Evaluate the whole system

```powershell
python evaluate.py --weights runs\helmet_gr2\weights\best.pt
```

Reports detection metrics, plus rider-level violation precision, recall, F1, and accuracy
(positive = violation, meaning a rider without a helmet).

---

## 9. Run the system on video

```powershell
python run_pipeline.py --weights runs\helmet_gr2\weights\best.pt --source your_clip.mp4 --save-video
```

Optional: `--start-time 2026-11-08T08:00:00` (timestamps in the log), `--show` (live window, press Q to stop),
`--source 0` (webcam), `--tracker botsort.yaml`.

Outputs in `outputs\`:
- `violations.csv` – one row per rider track: status (violation / compliant / uncertain), times, helmet ratio,
  confidences, evidence path
- `evidence\` – cropped rider and annotated frame for violations and uncertain riders
- `annotated.mp4` – open with VLC or Windows Media Player

How a decision is made: each tracked rider on a motorcycle is checked every frame for a helmet in the top
40 percent of the rider box. When the track ends, helmet in 30 percent or fewer frames = violation, 60 percent
or more = compliant, otherwise (or too few/small frames) = uncertain. One track = one record, so the same rider
is not logged twice.

Tuning flags: `--helmet-conf 0.25` (fewer false violations), `--head-frac`, `--min-frames`, `--min-height`,
`--violation-max`, `--compliant-min`, `--lost-frames`.

**Tracking and logging evaluation (by hand, 3 to 5 clips):** count the real riders and violators, compare
with `violations.csv`, and report duplicate records per violator and ID switches you see in `annotated.mp4`.

---

## 10. Quick command summary

```powershell
cd "D:\helmet_gr2 (2)"
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python merge_annotations.py --zip "Complete_Annotations-20261108T122503Z-1-001.zip" --out dataset
python train_elvie.py --root dataset
python evaluate.py --weights runs\helmet_gr2\weights\best.pt
python run_pipeline.py --weights runs\helmet_gr2\weights\best.pt --source your_clip.mp4 --save-video
```

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'yaml'` (or `ultralytics`, `cv2`) | Run `python -m pip install -r requirements.txt` in the activated `.venv`. |
| `FileNotFoundError: ...zip` | The zip is not in the folder you ran the command from. Run `dir *.zip`, copy the zip into the project folder, or pass its full path in quotes. To find it: `Get-ChildItem -Path $HOME -Recurse -Filter "Complete_Annotations*.zip" -ErrorAction SilentlyContinue \| Select-Object FullName` |
| `torch.cuda.is_available()` is `False` | See Section 4. |
| `python` not recognized | Use `py`, or reinstall Python with "Add to PATH" ticked. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process Bypass`, then activate again. |
| Still `No module named ...` after installing | pip and Python differ. Check `python -c "import sys; print(sys.executable)"` ends in `.venv\Scripts\python.exe`; select the `.venv` interpreter and open a new terminal. |
| Out of memory | Add `--batch 4` or `--batch 8`, or use `--model yolo26n.pt`. |
| "paging file is too small" (WinError 1455) | Close other apps and restart; if it repeats, reduce workers or batch size. |
| Training looks frozen on epoch 1 | It is usually loading data; wait a couple of minutes. |
| `yolo26s.pt` will not download | Use `--model yolo26n.pt`, and check your internet connection. |
| `[WARN]` in the class order check | A folder's yaml order disagrees with its box shapes. Check that folder's labels in Roboflow before training. |
| Helmeted riders flagged as violations | Lower `--helmet-conf` (0.2 to 0.25), raise `--head-frac` slightly, add more far/night helmet images. |
| Same rider logged twice | Raise `--lost-frames`, try `--tracker botsort.yaml` or a larger `--imgsz`. |
| Everything is `uncertain` | Riders are too small or short-lived; lower `--min-height` / `--min-frames`. |

---

## 12. Notes for the paper

- Dataset: 2,007 annotated images, 3 classes, from only 3 CCTV video segments of one camera.
- Mention the leakage problem and the video-held-out test split you used.
- Limitation: a single camera, so results describe new footage from the same camera, not other cameras.
- There is no `no_helmet` class (group decision). A violation is a rider with no helmet associated over the
  life of their track, which is why helmet recall matters so much.

## Project files

| File | Purpose |
|---|---|
| `merge_annotations.py` | Merge the team's zips, align class order, leak-free split |
| `train_elvie.py` | Check labels, train YOLO26, evaluate on test |
| `evaluate.py` | Detection and rider-level violation metrics |
| `run_pipeline.py` | Detect, track, associate, decide, log evidence |
| `associate.py` | Rider-on-motorcycle check and helmet association logic |
| `utils.py` | Helpers (file finding, class lookup by name) |
| `auto_label.py`, `prepare_dataset.py`, `train.py` | Older tools for building a dataset from scratch (not needed with the team's annotations) |
