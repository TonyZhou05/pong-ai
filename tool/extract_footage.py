"""Extract a pong-ai footage-demo fixture from an OpenTTGames clip.

Reads test_2.mp4 + its ball/event markup, runs YOLO11n-pose (players) and
YOLO11n (ball) over every 4th frame (120fps -> 30fps), fuses the detector's
ball with the dataset's labeled ball positions, and writes:

  out/openttgames_test2.json  -- ClipFixture JSON (the app's benchmark format)
  out/openttgames_test2.mp4   -- 960x540 @ 30fps H.264 re-encode for bundling
"""

import json
import statistics
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent / "openttgames"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SRC_FPS = 120
STRIDE = 4  # 120fps -> 30fps
OUT_W, OUT_H = 960, 540
BALL_BOX_PX = 18  # synthesized ball box size in source pixels
INTERP_MAX_GAP = 24  # max source-frame gap to linearly bridge markup labels

video = ROOT / "test_2.mp4"
ball_markup = {
    int(k): v for k, v in json.loads((ROOT / "test_2_markup/ball_markup.json").read_text()).items()
}
events = {
    int(k): v for k, v in json.loads((ROOT / "test_2_markup/events_markup.json").read_text()).items()
}

cap = cv2.VideoCapture(str(video))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- net line: median normalized x of the labeled ball at 'net' events -------
ball_frames = sorted(ball_markup)


def nearest_label(f, tol):
    best = None
    for g in ball_frames:
        if abs(g - f) <= tol and (best is None or abs(g - f) < abs(best - f)):
            best = g
    return best


net_xs = []
for f, ev in events.items():
    if ev == "net":
        g = nearest_label(f, 3)
        if g is not None:
            net_xs.append(ball_markup[g]["x"] / W)
net_x = statistics.median(net_xs) if net_xs else 0.5
print(f"netX = {net_x:.4f} from {len(net_xs)} net events")

# --- markup gap stats (sanity for interpolation) ------------------------------
gaps = [b - a for a, b in zip(ball_frames, ball_frames[1:])]
small = sum(1 for g in gaps if g <= INTERP_MAX_GAP)
print(f"markup gaps: {small}/{len(gaps)} <= {INTERP_MAX_GAP} frames")


def markup_ball_at(f):
    """Labeled ball at source frame f: exact/near label, or a bridged gap."""
    g = nearest_label(f, 2)
    if g is not None:
        p = ball_markup[g]
        return p["x"] / W, p["y"] / H, 1.0, "markup"
    # linear interpolation across a small labeled gap
    prev = max((g for g in ball_frames if g < f), default=None)
    nxt = min((g for g in ball_frames if g > f), default=None)
    if prev is not None and nxt is not None and nxt - prev <= INTERP_MAX_GAP:
        a, b = ball_markup[prev], ball_markup[nxt]
        t = (f - prev) / (nxt - prev)
        return (
            (a["x"] + (b["x"] - a["x"]) * t) / W,
            (a["y"] + (b["y"] - a["y"]) * t) / H,
            0.8,
            "interp",
        )
    return None


# --- models -------------------------------------------------------------------
pose_model = YOLO("yolo11n-pose.pt")
det_model = YOLO("yolo11n.pt")
try:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    device = "cpu"
print(f"device = {device}")

frames_json = []
gt_frames_json = []
writer = cv2.VideoWriter(
    str(OUT / "openttgames_test2.mp4"),
    cv2.VideoWriter_fourcc(*"avc1"),
    SRC_FPS / STRIDE,
    (OUT_W, OUT_H),
)
assert writer.isOpened(), "H.264 writer failed to open"

stats = {"markup": 0, "interp": 0, "detector": 0, "none": 0, "people2": 0}
bw, bh = BALL_BOX_PX / W, BALL_BOX_PX / H

f = -1
while True:
    ok, img = cap.read()
    if not ok:
        break
    f += 1
    if f % STRIDE:
        continue
    t_ms = round(f * 1000 / SRC_FPS)
    writer.write(cv2.resize(img, (OUT_W, OUT_H)))

    # players: pose model, keep the 2 largest boxes by area
    pres = pose_model.predict(img, imgsz=640, conf=0.4, classes=[0], device=device, verbose=False)[0]
    people = []
    if pres.boxes is not None and len(pres.boxes):
        order = sorted(
            range(len(pres.boxes)),
            key=lambda i: float(pres.boxes.xywh[i][2] * pres.boxes.xywh[i][3]),
            reverse=True,
        )[:2]
        for i in order:
            x1, y1, x2, y2 = (float(v) for v in pres.boxes.xyxyn[i])
            kps = []
            if pres.keypoints is not None and pres.keypoints.xyn is not None:
                xy = pres.keypoints.xyn[i]
                kconf = (
                    pres.keypoints.conf[i]
                    if pres.keypoints.conf is not None
                    else [1.0] * len(xy)
                )
                kps = [
                    [round(float(x), 4), round(float(y), 4), round(float(c), 3)]
                    for (x, y), c in zip(xy.tolist(), kconf)
                ]
            people.append(
                {
                    "box": [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)],
                    # stable id by table side so overlays/analytics stay consistent
                    "trackId": 0 if (x1 + x2) / 2 < net_x else 1,
                    "keypoints": kps,
                }
            )
    if len(people) == 2:
        stats["people2"] += 1

    # ball: dataset markup first, then the detector, then a bridged gap
    ball = None
    m = markup_ball_at(f)
    if m and m[3] == "markup":
        cx, cy, conf, src = m
        ball = (cx, cy, conf)
        stats["markup"] += 1
    else:
        dres = det_model.predict(img, imgsz=1280, conf=0.25, classes=[32], device=device, verbose=False)[0]
        best = None
        if dres.boxes is not None and len(dres.boxes):
            for i in range(len(dres.boxes)):
                x1, y1, x2, y2 = (float(v) for v in dres.boxes.xyxyn[i])
                conf = float(dres.boxes.conf[i])
                if min(x2 - x1, y2 - y1) > 0.05:  # too big to be a ping-pong ball
                    continue
                if best is None or conf > best[2]:
                    best = ((x1 + x2) / 2, (y1 + y2) / 2, conf)
        if best is not None:
            ball = best
            stats["detector"] += 1
        elif m:  # interpolated markup bridge
            ball = (m[0], m[1], m[2])
            stats["interp"] += 1
        else:
            stats["none"] += 1

    frame = {"t": t_ms}
    if ball:
        cx, cy, conf = ball
        frame["ball"] = {
            "label": "sports ball",
            "confidence": round(conf, 3),
            "box": [round(cx - bw / 2, 4), round(cy - bh / 2, 4), round(bw, 4), round(bh, 4)],
        }
    if people:
        frame["people"] = people
    frames_json.append(frame)

    # ground truth: markup-labeled ball only (exact labels, no interpolation)
    gt = {"t": t_ms}
    g = nearest_label(f, 2)
    if g is not None:
        p = ball_markup[g]
        gt["ball"] = {
            "label": "ball",
            "confidence": 1.0,
            "box": [
                round(p["x"] / W - bw / 2, 4),
                round(p["y"] / H - bh / 2, 4),
                round(bw, 4),
                round(bh, 4),
            ],
        }
    gt_frames_json.append(gt)

cap.release()
writer.release()

fixture = {
    "name": "openttgames_test2",
    "source": "OpenTTGames test_2 (lab.osai.ai/datasets/openttgames)",
    "fps": SRC_FPS // STRIDE,
    "netX": round(net_x, 4),
    "leftPlayer": "a",
    "firstServer": "a",
    "pointsPerGame": 11,
    "bestOf": 5,
    # Scoring outcome not annotated in the dataset; totals unknown -> 0/0.
    # This fixture is for the in-app footage demo (frames/overlays), not the
    # scoring-accuracy benchmark corpus.
    "groundTruth": {"pointsA": 0, "pointsB": 0},
    "frames": frames_json,
    "groundTruthEvents": [
        {"type": "bounce", "t": round(f * 1000 / SRC_FPS)}
        for f, ev in sorted(events.items())
        if ev == "bounce"
    ],
    "groundTruthFrames": gt_frames_json,
}
(OUT / "openttgames_test2.json").write_text(json.dumps(fixture))

total = len(frames_json)
print(f"frames: {total}")
print(f"ball coverage: markup {stats['markup']}, detector {stats['detector']}, "
      f"interp {stats['interp']}, none {stats['none']} "
      f"({(total - stats['none']) / total * 100:.1f}% covered)")
print(f"both players detected: {stats['people2']}/{total} ({stats['people2']/total*100:.1f}%)")
import os
print(f"video: {os.path.getsize(OUT / 'openttgames_test2.mp4')/1e6:.1f} MB")
print(f"json:  {os.path.getsize(OUT / 'openttgames_test2.json')/1e6:.1f} MB")
