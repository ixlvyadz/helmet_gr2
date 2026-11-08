"""Detect -> track -> associate helmets -> decide per rider -> log evidence.

Example:
  python run_pipeline.py --weights runs/helmet_gr2/weights/best.pt \
      --source clips/street1.mp4 --start-time 2026-11-08T08:00:00 --save-video

Outputs (in --out):
  violations.csv      one row per rider track (status, times, confidences, evidence path)
  evidence/           cropped rider + annotated full frame for violations/uncertain riders
  annotated.mp4       (with --save-video)

How a decision is made
  Every frame, each tracked rider (riding a motorcycle) either has a helmet
  associated with it or not. When the track ends, the share of frames WITH a
  helmet decides: <= violation-max -> violation, >= compliant-min -> compliant,
  otherwise (or too few frames / too small) -> uncertain. One track = one record,
  so the same rider is never logged twice.
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from associate import area, associate_helmets, rider_on_motorcycle
from utils import class_ids

YELLOW, BLUE, GREEN, RED = (0, 215, 255), (255, 128, 0), (0, 200, 0), (0, 0, 255)


class RiderTrack:
    def __init__(self, t, frame_idx):
        self.first_t = self.last_t = t
        self.last_frame = frame_idx
        self.frames = 0
        self.helmet_frames = 0
        self.rider_conf, self.helmet_conf = [], []
        self.best_area = 0.0
        self.best_h = 0.0
        self.best_box = None
        self.crop = None
        self.snapshot = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video path or camera index (e.g. 0)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="or botsort.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4, help="motorcycle / rider threshold")
    ap.add_argument("--helmet-conf", type=float, default=0.3,
                    help="lower than --conf on purpose: a missed helmet = false violation")
    ap.add_argument("--head-frac", type=float, default=0.4,
                    help="helmet must be in the top this-fraction of the rider box")
    ap.add_argument("--min-frames", type=int, default=10, help="min frames before judging a rider")
    ap.add_argument("--min-height", type=float, default=60, help="min rider box height in px")
    ap.add_argument("--violation-max", type=float, default=0.3, help="helmet ratio <= this -> violation")
    ap.add_argument("--compliant-min", type=float, default=0.6, help="helmet ratio >= this -> compliant")
    ap.add_argument("--lost-frames", type=int, default=30, help="frames unseen before a track is closed")
    ap.add_argument("--start-time", default=None, help="ISO time of video start, e.g. 2026-11-08T08:00:00")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()


def decide(st, a):
    if st.frames < a.min_frames or st.best_h < a.min_height:
        return "uncertain"
    ratio = st.helmet_frames / st.frames
    if ratio <= a.violation_max:
        return "violation"
    if ratio >= a.compliant_min:
        return "compliant"
    return "uncertain"


def finalize(tid, st, a, writer, fh, ev_dir, tag, base_time, counts):
    status = decide(st, a)
    counts[status] += 1
    ratio = st.helmet_frames / max(1, st.frames)
    evidence = ""
    if status in ("violation", "uncertain") and st.crop is not None:
        stem = f"{tag}_rider{tid}_{status}"
        cv2.imwrite(str(ev_dir / f"{stem}_crop.jpg"), st.crop)
        snap = st.snapshot.copy()
        x1, y1, x2, y2 = st.best_box.astype(int)
        cv2.rectangle(snap, (x1, y1), (x2, y2), RED, 3)
        cv2.putText(snap, f"rider {tid}: {status} (helmet ratio {ratio:.2f})",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
        evidence = str(ev_dir / f"{stem}_frame.jpg")
        cv2.imwrite(evidence, snap)
    event_time = (base_time + timedelta(seconds=st.first_t)).isoformat(timespec="seconds")
    writer.writerow([
        tag, tid, status, event_time, f"{st.first_t:.2f}", f"{st.last_t:.2f}", st.frames,
        f"{ratio:.3f}", f"{np.mean(st.rider_conf):.3f}",
        f"{np.mean(st.helmet_conf):.3f}" if st.helmet_conf else "", evidence,
    ])
    fh.flush()


def main():
    a = parse_args()
    out = Path(a.out)
    ev_dir = out / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(a.weights)
    MOTO, RIDER, HELMET = class_ids(model.names)  # looked up by class NAME
    is_cam = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source) if is_cam else a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tag = "camera" if is_cam else Path(a.source).stem
    base_time = datetime.fromisoformat(a.start_time) if a.start_time else datetime.now()

    vw = None
    if a.save_video:
        vw = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    csv_path = out / "violations.csv"
    new_file = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["video", "track_id", "status", "event_time", "first_time_s", "last_time_s",
                         "frames", "helmet_ratio", "mean_rider_conf", "mean_helmet_conf", "evidence"])

    tracks, counts, frame_idx = {}, {"violation": 0, "compliant": 0, "uncertain": 0}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        res = model.track(frame, persist=True, tracker=a.tracker, imgsz=a.imgsz,
                          conf=min(a.conf, a.helmet_conf), verbose=False)[0]
        motos, riders, helmets = [], [], []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            ids = (res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None
                   else [None] * len(cls))
            for b, c, p, i in zip(xyxy, cls, conf, ids):
                if c == MOTO and p >= a.conf:
                    motos.append((b, p))
                elif c == RIDER and p >= a.conf and i is not None:
                    riders.append((b, p, int(i)))
                elif c == HELMET and p >= a.helmet_conf:
                    helmets.append((b, p))

        # keep only riders that are on a motorcycle (drops pedestrians)
        moto_boxes = [m[0] for m in motos]
        riders = [r for r in riders if rider_on_motorcycle(r[0], moto_boxes)]
        assoc = associate_helmets([r[0] for r in riders], [h[0] for h in helmets], a.head_frac)

        for ri, (b, p, tid) in enumerate(riders):
            st = tracks.setdefault(tid, RiderTrack(t, frame_idx))
            has_helmet = ri in assoc
            st.frames += 1
            st.helmet_frames += int(has_helmet)
            st.rider_conf.append(float(p))
            if has_helmet:
                st.helmet_conf.append(float(helmets[assoc[ri][0]][1]))
            st.last_t, st.last_frame = t, frame_idx
            st.best_h = max(st.best_h, float(b[3] - b[1]))
            if area(b) > st.best_area * 1.15:  # keep the clearest (largest) view as evidence
                st.best_area, st.best_box = area(b), b.copy()
                x1, y1, x2, y2 = b.astype(int)
                pad = int(0.1 * max(x2 - x1, y2 - y1))
                st.crop = frame[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)].copy()
                st.snapshot = frame.copy()

        # close tracks that disappeared
        for tid in [k for k, s in tracks.items() if frame_idx - s.last_frame > a.lost_frames]:
            finalize(tid, tracks.pop(tid), a, writer, fh, ev_dir, tag, base_time, counts)

        if vw is not None or a.show:
            vis = frame.copy()
            for b, p in motos:
                cv2.rectangle(vis, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), YELLOW, 2)
            for b, p in helmets:
                cv2.rectangle(vis, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), BLUE, 2)
            for ri, (b, p, tid) in enumerate(riders):
                color = GREEN if ri in assoc else RED
                cv2.rectangle(vis, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), color, 2)
                cv2.putText(vis, f"R{tid} {'helmet' if ri in assoc else 'NO HELMET'}",
                            (int(b[0]), max(15, int(b[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if vw is not None:
                vw.write(vis)
            if a.show:
                cv2.imshow("helmet", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        frame_idx += 1

    for tid, st in tracks.items():  # tracks still open at the end
        finalize(tid, st, a, writer, fh, ev_dir, tag, base_time, counts)
    cap.release()
    if vw is not None:
        vw.release()
    fh.close()
    cv2.destroyAllWindows()
    print(f"frames: {frame_idx} | riders judged: {counts} | log: {csv_path}")


if __name__ == "__main__":
    main()
