"""Build the pong-ai real-footage match corpus from OpenTTGames clips.

For every downloaded OpenTTGames test video, segments the labeled play into
per-rally clips, runs YOLO11n-pose (players) + YOLO11n (ball) inference over
each segment, fuses the detector's ball with the dataset's labeled positions,
and writes per-segment assets plus a manifest:

  out_corpus/<seg>.mp4        960x540 @30fps H.264 segment video
  out_corpus/<seg>.json       ClipFixture detection track (t relative to clip)
  out_corpus/manifest.json    list the app's Matches screen renders

Also regenerates the two-rally test_2 main demo (openttgames_test2.*).
"""

import json
import statistics
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent / "openttgames"
OUT = Path(__file__).parent / "out_corpus"
OUT.mkdir(exist_ok=True)

SRC_FPS = 120
STRIDE = 4  # 120fps -> 30fps
OUT_W, OUT_H = 960, 540
BALL_BOX_PX = 18
INTERP_MAX_GAP = 24  # source frames
CLUSTER_GAP_S = 1.5  # label gap that splits rally clusters
MIN_CLUSTER_S = 0.4  # drop clusters with less labeled play than this
SEG_PAD_S = 1.5  # context seconds kept around a cluster in the cut video
PLAY_PAD_BEFORE_S = 1.0  # detector-ball trust window around the labels
PLAY_PAD_AFTER_S = 0.5

VIDEOS = ["test_1", "test_2", "test_3"]

# Rally outcomes verified by watching the clips (user-confirmed for test_2):
# segment id -> (pointsA, pointsB, winners). Others are unannotated.
KNOWN_TRUTH = {
    "test_2_r1": (0, 1, ["b"]),  # left player off-frame, triple bounce left
    "test_2_r2": (1, 0, ["a"]),  # right player's hit flies out of bounds
    "test_3_r1": (0, 1, ["b"]),  # ball dies on left, A never returns
    "test_3_r2": (1, 0, ["a"]),  # A's shot bounces on B's side, B steps away
    "test_3_r3": (0, 1, ["b"]),  # A's shot dives past the right edge, no bounce
    "test_3_r4": (0, 1, ["b"]),  # A nets his return; ball dies on his side
    "test_3_r5": (0, 1, ["b"]),  # B's return bounces the left edge away; A gives up
}

pose_model = YOLO("yolo11n-pose.pt")
det_model = YOLO("yolo11n.pt")
try:
    import torch

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
print(f"device = {DEVICE}")

manifest = []


def process_video(name):
    video = ROOT / f"{name}.mp4"
    ball = {
        int(k): v
        for k, v in json.loads((ROOT / f"{name}_markup/ball_markup.json").read_text()).items()
    }
    events = {
        int(k): v
        for k, v in json.loads((ROOT / f"{name}_markup/events_markup.json").read_text()).items()
    }
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bf = sorted(ball)

    def nearest_label(f, tol):
        best = None
        for g in bf:
            if abs(g - f) <= tol and (best is None or abs(g - f) < abs(best - f)):
                best = g
        return best

    # net line: median labeled ball x at 'net' events
    net_xs = [
        ball[g]["x"] / W
        for f, ev in events.items()
        if ev == "net" and (g := nearest_label(f, 3)) is not None
    ]
    net_x = statistics.median(net_xs) if net_xs else 0.5

    # table surface band: labeled ball position at bounce events, padded
    bxs, bys = [], []
    for f, ev in events.items():
        if ev == "bounce" and (g := nearest_label(f, 3)) is not None:
            bxs.append(ball[g]["x"] / W)
            bys.append(ball[g]["y"] / H)
    table = None
    if len(bxs) >= 4:
        table = {
            "tableLeft": round(max(0.0, min(bxs) - 0.05), 4),
            "tableRight": round(min(1.0, max(bxs) + 0.05), 4),
            "tableTop": round(max(0.0, min(bys) - 0.05), 4),
            "tableBottom": round(min(1.0, max(bys) + 0.05), 4),
        }

    # rally clusters from label gaps
    clusters, cur = [], [bf[0]]
    for g in bf[1:]:
        if g - cur[-1] > CLUSTER_GAP_S * SRC_FPS:
            clusters.append(cur)
            cur = []
        cur.append(g)
    clusters.append(cur)
    clusters = [c for c in clusters if (c[-1] - c[0]) / SRC_FPS >= MIN_CLUSTER_S]
    print(f"{name}: netX={net_x:.4f} table={table} -> {len(clusters)} segments")

    for i, cluster in enumerate(clusters, start=1):
        seg_id = f"{name}_r{i}"
        c0, c1 = cluster[0], cluster[-1]
        seg_start = max(0, int(c0 - SEG_PAD_S * SRC_FPS))
        seg_end = min(frames_total - 1, int(c1 + SEG_PAD_S * SRC_FPS))
        play0 = c0 - PLAY_PAD_BEFORE_S * SRC_FPS
        play1 = c1 + PLAY_PAD_AFTER_S * SRC_FPS
        bw, bh = BALL_BOX_PX / W, BALL_BOX_PX / H

        def markup_ball_at(f):
            g = nearest_label(f, 2)
            if g is not None:
                p = ball[g]
                return p["x"] / W, p["y"] / H, 1.0, "markup"
            prev = max((g for g in bf if g < f), default=None)
            nxt = min((g for g in bf if g > f), default=None)
            if prev is not None and nxt is not None and nxt - prev <= INTERP_MAX_GAP:
                a, b = ball[prev], ball[nxt]
                t = (f - prev) / (nxt - prev)
                return (
                    (a["x"] + (b["x"] - a["x"]) * t) / W,
                    (a["y"] + (b["y"] - a["y"]) * t) / H,
                    0.8,
                    "interp",
                )
            return None

        writer = cv2.VideoWriter(
            str(OUT / f"{seg_id}.mp4"),
            cv2.VideoWriter_fourcc(*"avc1"),
            SRC_FPS / STRIDE,
            (OUT_W, OUT_H),
        )
        assert writer.isOpened()
        frames_json, gt_frames_json = [], []
        ball_covered = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        f = seg_start - 1
        while f < seg_end:
            ok, img = cap.read()
            if not ok:
                break
            f += 1
            if (f - seg_start) % STRIDE:
                continue
            t_ms = round((f - seg_start) * 1000 / SRC_FPS)
            writer.write(cv2.resize(img, (OUT_W, OUT_H)))

            pres = pose_model.predict(
                img, imgsz=640, conf=0.4, classes=[0], device=DEVICE, verbose=False
            )[0]
            people = []
            if pres.boxes is not None and len(pres.boxes):
                order = sorted(
                    range(len(pres.boxes)),
                    key=lambda j: float(pres.boxes.xywh[j][2] * pres.boxes.xywh[j][3]),
                    reverse=True,
                )[:2]
                for j in order:
                    x1, y1, x2, y2 = (float(v) for v in pres.boxes.xyxyn[j])
                    kps = []
                    if pres.keypoints is not None and pres.keypoints.xyn is not None:
                        xy = pres.keypoints.xyn[j]
                        kconf = (
                            pres.keypoints.conf[j]
                            if pres.keypoints.conf is not None
                            else [1.0] * len(xy)
                        )
                        kps = [
                            [round(float(x), 4), round(float(y), 4), round(float(c), 3)]
                            for (x, y), c in zip(xy.tolist(), kconf)
                        ]
                    people.append(
                        {
                            "box": [
                                round(x1, 4),
                                round(y1, 4),
                                round(x2 - x1, 4),
                                round(y2 - y1, 4),
                            ],
                            "trackId": 0 if (x1 + x2) / 2 < net_x else 1,
                            "keypoints": kps,
                        }
                    )

            b = None
            m = markup_ball_at(f)
            if m and m[3] == "markup":
                b = (m[0], m[1], m[2])
            elif play0 <= f <= play1:
                dres = det_model.predict(
                    img, imgsz=1280, conf=0.25, classes=[32], device=DEVICE, verbose=False
                )[0]
                best = None
                if dres.boxes is not None and len(dres.boxes):
                    for j in range(len(dres.boxes)):
                        x1, y1, x2, y2 = (float(v) for v in dres.boxes.xyxyn[j])
                        conf = float(dres.boxes.conf[j])
                        if min(x2 - x1, y2 - y1) > 0.05:
                            continue
                        if best is None or conf > best[2]:
                            best = ((x1 + x2) / 2, (y1 + y2) / 2, conf)
                if best is not None:
                    b = best
                elif m:
                    b = (m[0], m[1], m[2])

            frame = {"t": t_ms}
            if b:
                cx, cy, conf = b
                ball_covered += 1
                frame["ball"] = {
                    "label": "sports ball",
                    "confidence": round(conf, 3),
                    "box": [
                        round(cx - bw / 2, 4),
                        round(cy - bh / 2, 4),
                        round(bw, 4),
                        round(bh, 4),
                    ],
                }
            if people:
                frame["people"] = people
            frames_json.append(frame)

            gt = {"t": t_ms}
            g = nearest_label(f, 2)
            if g is not None:
                p = ball[g]
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
        writer.release()

        pa, pb, winners = KNOWN_TRUTH.get(seg_id, (0, 0, None))
        truth = {"pointsA": pa, "pointsB": pb}
        if winners:
            truth["pointWinners"] = winners
        fixture = {
            "name": seg_id,
            "source": f"OpenTTGames {name} (lab.osai.ai/datasets/openttgames)",
            "fps": SRC_FPS // STRIDE,
            "netX": round(net_x, 4),
            **(table or {}),
            "leftPlayer": "a",
            "firstServer": "a",
            "pointsPerGame": 11,
            "bestOf": 5,
            "groundTruth": truth,
            "frames": frames_json,
            "groundTruthEvents": [
                {"type": "bounce", "t": round((f - seg_start) * 1000 / SRC_FPS)}
                for f, ev in sorted(events.items())
                if ev == "bounce" and seg_start <= f <= seg_end
            ],
            "groundTruthFrames": gt_frames_json,
        }
        (OUT / f"{seg_id}.json").write_text(json.dumps(fixture))
        dur_ms = frames_json[-1]["t"] if frames_json else 0
        bounces = len(fixture["groundTruthEvents"])
        manifest.append(
            {
                "id": seg_id,
                "title": f"{name.replace('_', ' ').title()} — Rally {i}",
                "video": f"assets/footage/{seg_id}.mp4",
                "fixture": f"assets/footage/{seg_id}.json",
                "durationMs": dur_ms,
                "bounces": bounces,
                "source": "OpenTTGames",
            }
        )
        print(
            f"  {seg_id}: {dur_ms/1000:.1f}s, ball {ball_covered}/{len(frames_json)} frames, "
            f"{bounces} labeled bounces"
        )
    cap.release()


for v in VIDEOS:
    process_video(v)

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
total_mb = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
print(f"---\n{len(manifest)} segments, {total_mb:.1f} MB total in {OUT}")
