"""Combine each source video's per-rally segments into one multi-rally clip.

For every `<video>_rN` fixture/video pair in out_corpus, concatenates the
rally segments (in order) into a single `<video>_full.mp4` + `.json` whose
frame timestamps run continuously, so the app's pipeline scores the rallies
cumulatively in one sitting. A 1.5 s freeze-frame gap (last frame
held, no detections) is inserted between segments: combined with each
segment's own no-play padding this guarantees the tracker's ball-lost
timeout elapses at every boundary, so each rally scores exactly as it does
standalone — without it, a trajectory can merge across the cut into the
next rally's serve and silently swallow the pending rally's point.

The combined ground truth is the sum of the segment truths when *every*
segment is labeled (pointWinners concatenated); otherwise it is left at 0-0
provisional (hand-verification pending).
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2

OUT = Path(__file__).parent / "out_corpus"
FPS = 30
# Freeze-frame gap between segments: must out-wait the pipeline's most
# patient ball-loss budget (extendedGapFrames 60 ≈ 2 s) so trajectories can
# never merge across a rally boundary.
GAP_FRAMES = 75  # 2.5 s

groups = defaultdict(list)
for p in sorted(OUT.glob("*_r*.json")):
    video = p.stem.rsplit("_r", 1)[0]
    groups[video].append(p)

for video, seg_paths in sorted(groups.items()):
    seg_paths.sort(key=lambda p: int(p.stem.rsplit("_r", 1)[1]))
    combined_frames = []
    combined_gt_frames = []
    combined_events = []
    winners = []
    points = {"a": 0, "b": 0}
    all_labeled = True
    first = json.loads(seg_paths[0].read_text())

    writer = cv2.VideoWriter(
        str(OUT / f"{video}_full.mp4"),
        cv2.VideoWriter_fourcc(*"avc1"),
        FPS,
        (960, 540),
    )
    assert writer.isOpened()

    offset_frames = 0
    for seg_i, p in enumerate(seg_paths):
        d = json.loads(p.read_text())
        # video frames
        cap = cv2.VideoCapture(str(OUT / f"{p.stem}.mp4"))
        n = 0
        last_img = None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            writer.write(img)
            last_img = img
            n += 1
        cap.release()

        def shift(t):
            # Recover the segment-local frame index, then restamp globally so
            # the combined clock is exactly the video's 30 fps frame clock.
            idx = round(t * FPS / 1000)
            return round((offset_frames + idx) * 1000 / FPS)

        for fr in d["frames"]:
            fr = dict(fr)
            fr["t"] = shift(fr["t"])
            combined_frames.append(fr)
        for fr in d.get("groundTruthFrames") or []:
            fr = dict(fr)
            fr["t"] = shift(fr["t"])
            combined_gt_frames.append(fr)
        for ev in d.get("groundTruthEvents") or []:
            ev = dict(ev)
            ev["t"] = shift(ev["t"])
            combined_events.append(ev)

        gt = d.get("groundTruth") or {}
        seg_winners = gt.get("pointWinners")
        if seg_winners:
            winners.extend(seg_winners)
            points["a"] += gt.get("pointsA", 0)
            points["b"] += gt.get("pointsB", 0)
        elif gt.get("pointsA", 0) == 0 and gt.get("pointsB", 0) == 0:
            all_labeled = False
        offset_frames += n

        # Inter-segment gap: hold the last frame, ball-less, but KEEP the
        # players (copied from the segment's last frame that saw them) — a
        # people-less gap would trigger the tracker's off-frame patience and
        # re-merge the boundary the gap exists to separate.
        if seg_i < len(seg_paths) - 1 and last_img is not None:
            gap_people = next(
                (fr["people"] for fr in reversed(d["frames"]) if fr.get("people")),
                None,
            )
            for _ in range(GAP_FRAMES):
                writer.write(last_img)
                gap_frame = {"t": round(offset_frames * 1000 / FPS)}
                if gap_people:
                    gap_frame["people"] = gap_people
                combined_frames.append(gap_frame)
                combined_gt_frames.append(
                    {"t": round(offset_frames * 1000 / FPS)})
                offset_frames += 1
    writer.release()

    truth = (
        {"pointsA": points["a"], "pointsB": points["b"], "pointWinners": winners}
        if all_labeled and winners
        else {"pointsA": 0, "pointsB": 0}
    )
    fixture = {
        "name": f"{video}_full",
        "source": first["source"],
        "fps": FPS,
        "netX": first["netX"],
        **{
            k: first[k]
            for k in ("tableLeft", "tableRight", "tableTop", "tableBottom")
            if k in first
        },
        "leftPlayer": first["leftPlayer"],
        "firstServer": first["firstServer"],
        "pointsPerGame": first["pointsPerGame"],
        "bestOf": first["bestOf"],
        "groundTruth": truth,
        "frames": combined_frames,
        "groundTruthEvents": combined_events,
        "groundTruthFrames": combined_gt_frames,
    }
    (OUT / f"{video}_full.json").write_text(json.dumps(fixture))
    dur = combined_frames[-1]["t"] / 1000 if combined_frames else 0
    print(
        f"{video}_full: {len(seg_paths)} rallies, {dur:.1f}s, "
        f"truth {truth['pointsA']}-{truth['pointsB']}"
        f"{' (provisional)' if not (all_labeled and winners) else ''}"
    )
