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

# Rally-ness filter: a labeled cluster only becomes a corpus clip if it is a
# real rally rather than a between-point handover (a player knocking the ball
# to the other). A rally shows the ball crossing the net at least twice (the
# serve over and a return back) — or, since sparse labels can miss crossings,
# bounces on BOTH sides of the table over a non-trivial span. A handover is a
# short one-way trip.
MIN_RALLY_CROSSINGS = 2
MIN_RALLY_SPAN_S = 1.0

# Segment-end extension: the dataset's labels sometimes stop while the rally
# is still being played (e.g. a player retreats off-frame and the annotators
# stop marking). Probe past the last label with the ball detector and keep
# extending the segment while the ball is still seen in play, so a clip never
# cuts (and force-scores) a rally that is still live.
EXTEND_CAP_S = 15.0  # max seconds to extend past the last label
EXTEND_QUIET_S = 2.0  # stop once the ball has been gone this long
SEG_PAD_S = 1.5  # context seconds kept around a cluster in the cut video
PLAY_PAD_BEFORE_S = 1.0  # detector-ball trust window around the labels
PLAY_PAD_AFTER_S = 0.5

VIDEOS = ["test_1", "test_2", "test_3", "test_5", "test_6", "test_7",
          "test_4", "game_3", "game_4"]

# Full-game videos hold dozens of labeled rallies; cap how many become
# corpus clips so the asset bundle and the labeling workload stay sane.
SEGMENT_CAP = {"game_3": 8, "game_4": 8, "test_4": 10}

# Segments excluded on human review (benchmark/labels/rally_labels.json):
# warm-up exchanges before the match starts, or rallies whose resolution the
# source video never shows. Keyed by segment id under the current numbering.
EXCLUDED_SEGMENTS = {"test_5_r7", "test_5_r8"}

# Human-labeled rally end times (seconds, segment-relative). The segment
# extension probe is capped at label_end + END_LABEL_ROOM_S — generous room
# so a rally's real ending (dying bounces, ball settling) is never cut,
# while dead-time tails are.
END_LABEL_ROOM_S = 1.5
LABELED_ENDS = {
    "game_3_r2": 4.5,
    "game_3_r3": 5.5,
    "game_3_r4": 4.6,
    "game_3_r5": 4.4,
    "game_3_r6": 4.4,
    "game_3_r7": 2.9,
    "game_3_r8": 5.4,
    "game_4_r1": 4.6,
    "game_4_r2": 3.7,
    "game_4_r3": 4,
    "game_4_r4": 3.8,
    "game_4_r5": 7,
    "game_4_r6": 7,
    "game_4_r7": 10.4,
    "game_4_r8": 3.5,
    "test_1_r1": 6.4,
    "test_1_r2": 14.8,
    "test_2_r1": 10.7,
    "test_2_r2": 6.8,
    "test_3_r1": 3.7,
    "test_3_r2": 2.6,
    "test_3_r3": 3.8,
    "test_3_r4": 9.8,
    "test_3_r5": 4.5,
    "test_4_r1": 5.4,
    "test_4_r2": 3.8,
    "test_4_r3": 3.9,
    "test_4_r4": 7.2,
    "test_4_r6": 3,
    "test_4_r7": 7.4,
    "test_4_r8": 5.2,
    "test_5_r1": 3.9,
    "test_5_r2": 3.9,
    "test_5_r3": 4,
    "test_5_r4": 3.6,
    "test_5_r5": 3.5,
    "test_5_r6": 3.9,
    "test_6_r1": 4.2,
    "test_6_r2": 4.5,
    "test_6_r3": 9.5,
    "test_6_r4": 3.2,
    "test_6_r5": 3.2,
    "test_6_r6": 6.4,
    "test_6_r7": 7.5,
    "test_6_r8": 4,
    "test_7_r1": 3.6,
    "test_7_r2": 6,
    "test_7_r3": 4.1,
    "test_7_r4": 12,
    "test_7_r5": 3,
}

# Rally outcomes verified by watching the clips (user-confirmed for test_2):
# segment id -> (pointsA, pointsB, winners). Others are unannotated.
KNOWN_TRUTH = {
    "game_3_r2": (1, 0, ["a"]),  # user-verified: outOfBounds
    "game_3_r3": (0, 1, ["b"]),  # user-verified: outOfBounds
    "game_3_r4": (0, 1, ["b"]),  # user-verified: intoNet
    "game_3_r5": (0, 1, ["b"]),  # user-verified: outOfBounds
    "game_3_r6": (1, 0, ["a"]),  # user-verified: outOfBounds
    "game_3_r7": (1, 0, ["a"]),  # user-verified: outOfBounds
    "game_3_r8": (1, 0, ["a"]),  # user-verified: outOfBounds
    "game_4_r1": (0, 1, ["b"]),  # user-verified: outOfBounds
    "game_4_r2": (1, 0, ["a"]),  # user-verified: intoNet
    "game_4_r3": (0, 1, ["b"]),  # user-verified: intoNet
    "game_4_r4": (0, 1, ["b"]),  # user-verified: outOfBounds
    "game_4_r5": (1, 0, ["a"]),  # user-verified: outOfBounds
    "game_4_r6": (0, 1, ["b"]),  # user-verified: notReturned
    "game_4_r7": (1, 0, ["a"]),  # user-verified: intoNet
    "game_4_r8": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_1_r1": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_1_r2": (1, 0, ["a"]),  # user-verified: outOfBounds
    "test_2_r1": (0, 1, ["b"]),  # user-verified: intoNet
    "test_2_r2": (1, 0, ["a"]),  # user-verified: outOfBounds
    "test_3_r1": (0, 1, ["b"]),  # user-verified: intoNet
    "test_3_r2": (1, 0, ["a"]),  # user-verified: intoNet
    "test_3_r3": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_3_r4": (0, 1, ["b"]),  # user-verified: intoNet
    "test_3_r5": (0, 1, ["b"]),  # user-verified: notReturned
    "test_4_r1": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_4_r2": (1, 0, ["a"]),  # user-verified: intoNet
    "test_4_r3": (1, 0, ["a"]),  # user-verified: outOfBounds
    "test_4_r4": (1, 0, ["a"]),  # user-verified: intoNet
    "test_4_r6": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_4_r7": (0, 1, ["b"]),  # user-verified: intoNet
    "test_4_r8": (0, 1, ["b"]),  # user-verified: intoNet
    "test_5_r1": (1, 0, ["a"]),  # user-verified: outOfBounds
    "test_5_r2": (0, 1, ["b"]),  # user-verified: notReturned
    "test_5_r3": (1, 0, ["a"]),  # user-verified: notReturned
    "test_5_r4": (1, 0, ["a"]),  # user-verified: outOfBounds
    "test_5_r5": (1, 0, ["a"]),  # user-verified: serveFault
    "test_5_r6": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_6_r1": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_6_r2": (1, 0, ["a"]),  # user-verified: outOfBounds
    "test_6_r3": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_6_r4": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_6_r5": (0, 1, ["b"]),  # user-verified: intoNet
    "test_6_r6": (0, 1, ["b"]),  # user-verified: notReturned
    "test_6_r7": (0, 1, ["b"]),  # user-verified: outOfBounds
    "test_6_r8": (0, 1, ["b"]),  # user-verified: intoNet
    "test_7_r1": (0, 1, ["b"]),  # user-verified: notReturned
    "test_7_r2": (1, 0, ["a"]),  # user-verified: intoNet
    "test_7_r3": (1, 0, ["a"]),  # user-verified: outOfBounds
    "test_7_r4": (0, 1, ["b"]),  # user-verified: intoNet
    "test_7_r5": (1, 0, ["a"]),  # user-verified: intoNet
}

pose_model = YOLO("yolo11n-pose.pt")
# Prefer the fine-tuned single-class ball detector (tool/train_ball_detector.py,
# held-out val P=0.98 R=0.94 mAP50=0.95 on test_7) over the stock COCO
# `sports ball` class, whose recall on a motion-blurred 40 mm ball is poor.
_FINETUNED = next(
    (
        p
        for p in [
            Path(__file__).resolve().parent.parent / "models/pingpong_ball_yolo11n.pt",
            Path.cwd() / "models/pingpong_ball_yolo11n.pt",
            Path.cwd() / "runs/detect/pingpong_ball/weights/best.pt",
        ]
        if p.exists()
    ),
    None,
)
if _FINETUNED is not None:
    det_model = YOLO(str(_FINETUNED))
    BALL_CLASSES = [0]
    print(f"ball detector: fine-tuned ({_FINETUNED})")
else:
    det_model = YOLO("yolo11n.pt")
    BALL_CLASSES = [32]
    print("ball detector: stock COCO sports-ball")
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
            bx, by = ball[g]["x"] / W, ball[g]["y"] / H
            # Markup occasionally carries junk coordinates (e.g. -0.001);
            # a single outlier would blow the bounds open to the frame edge.
            if 0.02 < bx < 0.98 and 0.02 < by < 0.98:
                bxs.append(bx)
                bys.append(by)
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

    def is_rally(c):
        span = (c[-1] - c[0]) / SRC_FPS
        crossings = 0
        sides = set()
        prev = None
        for g in c:
            x = ball[g]["x"] / W
            if not (0.02 < x < 0.98):
                continue
            side = x < net_x
            if prev is not None and g - prev[0] <= 12 and side != prev[1]:
                crossings += 1
            prev = (g, side)
        for f, ev in events.items():
            if ev == "bounce" and c[0] - 60 <= f <= c[-1] + 120:
                g = nearest_label(f, 3)
                if g is not None and 0.02 < ball[g]["x"] / W < 0.98:
                    sides.add(ball[g]["x"] / W < net_x)
        return crossings >= MIN_RALLY_CROSSINGS or (
            len(sides) >= 2 and span >= MIN_RALLY_SPAN_S
        )

    raw_clusters = list(clusters)
    kept = [c for c in clusters if is_rally(c)]
    dropped = len(clusters) - len(kept)
    seg_cap = SEGMENT_CAP.get(name)
    if seg_cap is not None and len(kept) > seg_cap:
        print(f"  ({len(kept) - seg_cap} rallies beyond the cap of {seg_cap} skipped)")
        kept = kept[:seg_cap]
    clusters = kept
    print(
        f"{name}: netX={net_x:.4f} table={table} -> "
        f"{len(clusters)} rallies ({dropped} handovers dropped)"
    )

    def probe_extension(c1, next_c0):
        """Last frame (<= c1 + cap) where the detector still sees the ball in
        play past the final label — the rally may outlive the annotations.
        Hard-capped before the *next* cluster starts: between rallies the
        detector often tracks the ball being carried/held, which would
        otherwise extend one segment into the following rally."""
        last_active = c1
        f = c1
        cap_frames = min(
            c1 + EXTEND_CAP_S * SRC_FPS,
            (next_c0 - 2.0 * SRC_FPS) if next_c0 is not None else float("inf"),
        )
        x_lo = (table["tableLeft"] if table else 0.0) - 0.15
        x_hi = (table["tableRight"] if table else 1.0) + 0.15
        cap_reader = cv2.VideoCapture(str(video))
        cap_reader.set(cv2.CAP_PROP_POS_FRAMES, c1)
        while f < cap_frames:
            ok, img = cap_reader.read()
            if not ok:
                break
            f += 1
            if (f - c1) % STRIDE:
                continue
            if f - last_active > EXTEND_QUIET_S * SRC_FPS:
                break
            dres = det_model.predict(
                img, imgsz=1280, conf=0.3, classes=BALL_CLASSES, device=DEVICE, verbose=False
            )[0]
            if dres.boxes is not None:
                for j in range(len(dres.boxes)):
                    x1, y1, x2, y2 = (float(v) for v in dres.boxes.xyxyn[j])
                    if min(x2 - x1, y2 - y1) > 0.05:
                        continue
                    cx = (x1 + x2) / 2
                    if x_lo <= cx <= x_hi:
                        last_active = f
                        break
        cap_reader.release()
        return last_active

    # Extension caps come from the *raw* cluster boundaries (handovers count:
    # they still mark where new ball activity begins).
    raw_starts = sorted(c[0] for c in raw_clusters)

    for i, cluster in enumerate(clusters, start=1):
        seg_id = f"{name}_r{i}"
        if seg_id in EXCLUDED_SEGMENTS:
            print(f"  {seg_id}: excluded (human review)")
            continue
        c0, c1 = cluster[0], cluster[-1]
        next_c0 = next((s0 for s0 in raw_starts if s0 > c1), None)
        if seg_id in LABELED_ENDS:
            # Segment start is c0 - SEG_PAD_S; the labeled end is relative to
            # that. Cap the extension probe with room to spare.
            # probe_extension subtracts a 2 s margin from next_c0, so add
            # it back: the effective cap lands at label_end + room.
            label_cap = int(
                c0
                + (-SEG_PAD_S + LABELED_ENDS[seg_id] + END_LABEL_ROOM_S + 2.0)
                * SRC_FPS
            )
            next_c0 = min(next_c0, label_cap) if next_c0 is not None else label_cap
            next_c0 = max(next_c0, int(c1 + 2.0 * SRC_FPS))  # never cap before the labels end
        c1_ext = probe_extension(c1, next_c0)
        if c1_ext > c1 + SRC_FPS:  # extended by more than a second
            print(f"  {seg_id}: play continues {(c1_ext - c1) / SRC_FPS:.1f}s "
                  "past the labels — segment extended")
        c1 = c1_ext
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
                    img, imgsz=1280, conf=0.25, classes=BALL_CLASSES, device=DEVICE, verbose=False
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
