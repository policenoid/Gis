#!/usr/bin/env python3
# people_stop_detector.py
# Detect, track, count people, flag STOP when someone stops walking,
# and save the processed result to a new video file.

import cv2
import numpy as np
from collections import deque
import argparse
import time
import math
import os

# ---------- Simple tracker based on nearest-centroid matching ----------

class Track:
    _next_id = 1
    def __init__(self, centroid, bbox, frame_time):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox  # (x, y, w, h)
        self.centroids = deque(maxlen=20)  # recent positions
        self.centroids.append(np.array(centroid, dtype=float))
        self.last_update = frame_time
        self.age = 0.0
        self.visible = True
        self.missed = 0
        self.stopped = False
        self.stop_since = None

    def update(self, centroid, bbox, frame_time):
        self.centroids.append(np.array(centroid, dtype=float))
        self.bbox = bbox
        dt = frame_time - self.last_update
        if dt < 0: dt = 0
        self.age += dt
        self.last_update = frame_time
        self.visible = True
        self.missed = 0

    def mark_missed(self):
        self.visible = False
        self.missed += 1

    def speed_px_per_frame(self):
        if len(self.centroids) < 2:
            return 0.0
        pts = list(self.centroids)
        dists = [np.linalg.norm(pts[i] - pts[i-1]) for i in range(1, len(pts))]
        if not dists:
            return 0.0
        return float(np.mean(dists))  # px per frame


def parse_args():
    ap = argparse.ArgumentParser(description="Detect, count and flag STOP for stationary persons; save to video.")
    ap.add_argument("-i", "--input", type=str, default="",
                    help="Path to video file. Omit for webcam.")
    ap.add_argument("-o", "--output", type=str, default="output_stop.mp4",
                    help="Path to output video file (e.g., output.mp4).")
    ap.add_argument("--detect-every", type=int, default=3,
                    help="Run the HOG detector every N frames (default: 3).")
    ap.add_argument("--match-max-dist", type=float, default=80.0,
                    help="Max centroid distance (px) for matching detections to tracks.")
    ap.add_argument("--miss-tolerance", type=int, default=10,
                    help="Frames to keep a track without match before deleting.")
    ap.add_argument("--stop-speed-thresh", type=float, default=15.0,
                    help="Speed threshold in px/sec below which a person is considered stopped.")
    ap.add_argument("--stop-min-secs", type=float, default=1.5,
                    help="Seconds below threshold to declare STOP.")
    ap.add_argument("--resize-width", type=int, default=960,
                    help="Resize frame width for faster processing (0 to keep original).")
    ap.add_argument("--display", action="store_true", help="Show a live window while processing.")
    return ap.parse_args()


def non_max_suppression_fast(boxes, overlapThresh=0.65):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=float)
    pick = []
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [len(idxs)-1]

        for pos in range(0, len(idxs)-1):
            i = last
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            overlap = (w * h) / area[j]
            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)
    return boxes[pick].astype(int).tolist()


def choose_fourcc(path):
    ext = os.path.splitext(path.lower())[1]
    # Reasonable defaults cross-platform:
    if ext in [".mp4", ".m4v"]:
        # Try H.264 if available; fallback to mp4v
        # Many OpenCV builds accept 'avc1' or 'H264', but 'mp4v' is safer.
        return cv2.VideoWriter_fourcc(*"mp4v")
    if ext in [".avi"]:
        return cv2.VideoWriter_fourcc(*"XVID")
    # Default
    return cv2.VideoWriter_fourcc(*"mp4v")


def main():
    args = parse_args()

    cap = cv2.VideoCapture(0 if args.input == "" else args.input)
    if not cap.isOpened():
        raise SystemExit("ERROR: Cannot open video source.")

    # FPS for writer: use the source fps if available, else 30.
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-2:
        src_fps = 30.0

    # Live FPS estimate for on-screen stats (independent of writer fps)
    smooth_fps = src_fps
    last_time = time.time()

    # HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    tracks = []
    unique_ids_seen = set()
    frame_idx = 0
    writer = None
    out_fourcc = choose_fourcc(args.output)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        # Resize for speed (optional)
        if args.resize_width and frame.shape[1] != args.resize_width:
            scale = args.resize_width / float(frame.shape[1])
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            scale = 1.0

        now = time.time()
        dt = now - last_time
        if dt > 0.001:
            smooth_fps = 0.9*smooth_fps + 0.1*(1.0/dt)
        last_time = now
        frame_idx += 1

        # Run detector every N frames
        detections = []
        if frame_idx % args.detect_every == 0:
            rects, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
            boxes = [[x, y, x+w, y+h] for (x,y,w,h) in rects]
            boxes = non_max_suppression_fast(boxes, 0.6)
            detections = [(x, y, x2-x, y2-y) for (x, y, x2, y2) in boxes]
        else:
            # No fresh detections this frame; keep tracks but mark missed
            pass

        # Match detections to existing tracks
        used_det = set()
        if detections:
            det_centroids = [((x + w/2), (y + h/2)) for (x,y,w,h) in detections]
            for tr in tracks:
                # find nearest unused detection
                best_j = -1
                best_dist = 1e9
                for j, c in enumerate(det_centroids):
                    if j in used_det:
                        continue
                    dist = math.dist(c, tr.centroids[-1])
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                if best_j != -1 and best_dist <= args.match_max_dist:
                    tr.update(det_centroids[best_j], detections[best_j], now)
                    used_det.add(best_j)
                else:
                    tr.mark_missed()
            # new tracks for unmatched detections
            for j, (bbox, c) in enumerate(zip(detections, det_centroids)):
                if j in used_det:
                    continue
                t = Track(c, bbox, now)
                tracks.append(t)
        else:
            # mark all as missed (we keep last bbox)
            for tr in tracks:
                tr.mark_missed()

        # prune old tracks
        tracks = [tr for tr in tracks if tr.missed <= args.miss_tolerance]

        # compute speeds and STOP
        current_count = 0
        any_stop = False
        for tr in tracks:
            if tr.visible:
                current_count += 1
                unique_ids_seen.add(tr.id)
            px_per_frame = tr.speed_px_per_frame()
            speed_px_sec = px_per_frame * smooth_fps
            if speed_px_sec < args.stop_speed_thresh:
                if not tr.stopped:
                    if tr.stop_since is None:
                        tr.stop_since = now
                    elif (now - tr.stop_since) >= args.stop_min_secs:
                        tr.stopped = True
                        any_stop = True
                else:
                    any_stop = True
            else:
                tr.stopped = False
                tr.stop_since = None

        # draw overlays
        overlay = frame.copy()
        for tr in tracks:
            x, y, w, h = map(int, tr.bbox)
            color = (0, 255, 0) if tr.visible else (128, 128, 128)
            if tr.stopped:
                color = (0, 0, 255)

            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
            label = f"ID {tr.id}" + ("  STOP" if tr.stopped else "")
            cv2.putText(overlay, label, (x, max(0, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 38), (0,0,0), -1)
        info = f"On-screen: {current_count}   Unique: {len(unique_ids_seen)}   FPS: {smooth_fps:.1f}"
        cv2.putText(overlay, info, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        if any_stop:
            txt = "STOP"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 6)
            cx = overlay.shape[1]//2 - tw//2
            cy = 80
            cv2.putText(overlay, txt, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 6, cv2.LINE_AA)

        # ---------- lazy-init the writer once we know final frame size ----------
        if writer is None:
            h, w = overlay.shape[:2]
            writer = cv2.VideoWriter(
                args.output,
                out_fourcc,
                src_fps,         # constant fps in the output file
                (w, h)
            )
            if not writer.isOpened():
                raise SystemExit(f"ERROR: Cannot open writer for '{args.output}'. Try a different extension or codec.")

        writer.write(overlay)

        if args.display:
            cv2.imshow("People STOP Detector", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Saved processed video to: {args.output} (note: no audio)")
    

if __name__ == "__main__":
    main()
