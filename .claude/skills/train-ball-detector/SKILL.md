---
name: train-ball-detector
description: Fine-tune / retrain the YOLO ball detector on labeled videos and integrate it safely (the detector is the only learned component; everything downstream is rules).
---

# Training the ball detector

Only the detection layer learns. `tool/train_ball_detector.py` fine-tunes
`yolo11n.pt` into a single-class ball detector from OpenTTGames-style labels
(per-frame ball centres → synthesized ~22px boxes).

## Run

From the dir holding `openttgames/` (needs anaconda python w/ ultralytics):

    nohup python3 tool/train_ball_detector.py --epochs 30 --imgsz 960 \
        > train_ball.log 2>&1 &

- Detach with nohup — harness task cleanup kills long background tasks.
- Dataset: train = five videos, val = a held-out video (test_7). ~2,400/385
  images. ~2h on MPS; checkpoints per epoch in
  `runs/detect/pingpong_ball/weights/`.
- Metrics that matter: P / R / mAP50. Achieved (2026-07-13): **P 0.98,
  R 0.94, mAP50 0.95** vs stock COCO's ~10–40% recall. Ignore mAP50-95 —
  labels are synthesized fixed-size boxes, tight-IoU is meaningless (repo
  benchmark uses IoU 0.3 for the same reason).
- New user-provided videos with ball labels (any format → convert to centre
  per frame) extend the dataset; new venues/angles are the main win.

## Integrate (CAUTION — verified regression risk)

Weights live at `models/pingpong_ball_yolo11n.pt` (committed). The corpus
extractor auto-prefers them (class id becomes 0, not COCO 32) — verify the
log prints "ball detector: fine-tuned" (path resolution must find the
weights from the working dir).

**A better detector changes the referee's operating point.** Dense tracks
surface dead-ball table bounces, held-ball trajectories (segment
over-extension — tighten the probe x-gate), and remove the gaps some rules
key on. The first fine-tuned re-extraction regressed 4/11 verified truths.
NEVER swap the detector into the shipping corpus without the full
verification sweep (see the footage-corpus skill); retune rules against the
verified suite first (candidates: eager double-cross evaluation, exact
table-bounds probe gate, revisit maxGapFrames 30).

App-side (live camera): export weights (`model.export(format='tflite')` /
coreml) → bundle → point `pingPongDetectProfile`
(`lib/core/vision/vision_model_profile.dart`) at them.
