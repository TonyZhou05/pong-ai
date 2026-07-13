# pong-ai — Architecture & Model Selection

pong-ai is a Flutter app that watches a table-tennis match from a phone placed on
the side of the table, tracks the ball and the players, keeps score
automatically, and produces a post-match performance summary. It also has a
**training mode** for practising against a robot/return net where it grades the
quality of each shot.

This document records the technical decisions for the vision pipeline and the
overall app architecture. It is the source of truth for *why* we picked the
model we picked — the objective explicitly prioritises "finding a good model
that can track the players well."

---

## 1. Vision model selection

### Requirements

| Need | Why |
| --- | --- |
| **Player tracking (pose)** | Score/rally logic needs to know who is at which end and when a player swings. Performance analytics (footwork, stance, reaction) needs joint keypoints. |
| **Ball detection (small, fast)** | A 40 mm ball can move >100 km/h and blurs to a few faint pixels. This is the hardest part of the pipeline. |
| **On-device, real-time** | The phone sits table-side with no guaranteed network. We target ≥25 FPS on a mid-range 2022+ phone. |
| **Runs from Flutter** | Single codebase, iOS + Android. |

### Decision

- **Primary runtime: [`ultralytics_yolo`](https://pub.dev/packages/ultralytics_yolo) (official Ultralytics Flutter plugin, v0.6.x).**
  It is the official plugin, runs on-device via LiteRT/TFLite (Android) and Core
  ML (iOS), supports **both detection and pose** tasks, reports native FPS, and
  lets us drop in our own exported `.tflite` / Core ML models. This removes the
  need to hand-write platform channels for camera + inference.

- **Players: YOLO *pose* model (`yolo11n-pose` / `yolo26n-pose` class).**
  Nano pose model gives 17 COCO keypoints per person at real-time speed. We only
  ever expect ≤2–4 people in frame, so the nano tier is plenty and keeps latency
  low. Pose (not just a box) is required for the training-mode shot-quality
  analysis (swing arc, stance width, contact timing).

- **Ball: a dedicated single-class detector, fine-tuned.**
  Off-the-shelf COCO models have `sports ball` but perform poorly on a small,
  motion-blurred ping-pong ball. Plan:
  1. Start with **YOLO nano detection** fine-tuned on a ping-pong-ball dataset.
  2. Because the ball is defined more by **motion than texture**, fuse the
     detector with a **motion cue** (frame differencing / background subtraction)
     and a **Kalman-filter tracker** to bridge frames where the detector misses
     the blurred ball. This mirrors the published state-of-the-art
     (YOLOv4-Tiny + GRU/Kalman trajectory models reach ~150 Hz ball tracking).

### Why not alternatives

- **MediaPipe / ML Kit pose only** — good for people, but no path to a custom
  fine-tuned *ball* detector in the same runtime.
- **Cloud inference** — violates the "place phone table-side, just works"
  requirement and adds latency the ball can't tolerate.
- **Full YOLO (s/m/l)** — too heavy for sustained real-time on mid-range phones.

### Model export path

Train/fine-tune with Ultralytics in Python → export to `.tflite` (Android,
INT8-quantized) and Core ML (iOS) → bundle in `assets/models/`. Quantization is
expected to be required to hit the FPS target.

---

## 2. Benchmarking plan

The objective asks us to "use various resources to find videos to benchmark on."
We will assemble an evaluation set and track metrics per iteration.

### Candidate public data / video sources

- **SPIN** — high-speed, high-resolution ping-pong dataset for tracking + action
  recognition (arXiv 1912.06640). Primary source for ball-tracking accuracy.
- **OpenTTGames** — table-tennis dataset with ball position + event annotations
  (bounce, net, empty-event) suitable for scoring-logic evaluation.
- **Broadcast / amateur match footage** (YouTube) — side-angle clips that match
  our real deployment geometry; used for qualitative end-to-end checks.

### Metrics we track

| Stage | Metric |
| --- | --- |
| Ball detection | Precision / recall @ IoU 0.3 (small object → loose IoU), % frames tracked through blur |
| Player pose | Keypoint mAP (OKS), ID stability across a rally |
| Scoring | Point-level accuracy vs. ground-truth scoreboard, rally-boundary F1 |
| Latency | End-to-end FPS on reference devices |

A `benchmark/` harness runs the pipeline over labeled clips and emits these
numbers so we can compare model swaps objectively. It is implemented (pure Dart,
runs in `flutter test`): `lib/core/benchmark/` defines the JSON `ClipFixture`
format and `BenchmarkRunner`; `benchmark/` holds the clip corpus and the format
docs. Two evaluation stages exist:

- **Scoring** (`BenchmarkRunner`) — point-total and ordered point accuracy of
  the tracker→referee→scoring pipeline vs. ground truth.
- **Perception** (`DetectionBenchmark`, iteration 15) — per-frame **ball
  detection** precision / recall / F1 at a loose IoU 0.3 (mean IoU + centre
  error on matches) and **player pose** detection rate + PCK / mean keypoint
  error, computed by comparing the pipeline's predicted `FrameResult`s to a
  clip's optional per-frame `groundTruthFrames`. This directly scores "how well
  does the model track the players/ball," the objective's stated priority. It
  runs the moment an annotated clip carries ground-truth frames — no camera
  needed.

See [`../benchmark/README.md`](../benchmark/README.md).

---

## 3. App architecture

```
lib/
  main.dart              App entry
  app.dart               MaterialApp + routing + theme
  core/
    scoring/             Pure-Dart table-tennis rules engine (no Flutter deps → unit-testable)
      match.dart         Match/game/point state
      scoring_engine.dart Serve rotation, deuce, game/match win logic
    vision/              Camera + model abstraction
      detection.dart     Detection / keypoint data models
      vision_service.dart Interface over the YOLO runtime (swappable, mockable)
      yolo_frame_adapter.dart Pure mapping: ultralytics_yolo streaming output → FrameResult
      yolo_vision_service.dart Camera-backed VisionService: routes plugin callbacks → monotonic FrameResult stream
    tracking/            Ball Kalman tracker, rally/point event detection
                         (TableGeometry calibrates the net line + table surface
                         region so off-table/floor bounces aren't scored;
                         TableCalibrator auto-estimates that geometry from a
                         warm-up of observed ball/player positions)
    analysis/            Performance + shot-quality analytics
  features/
    home/                Landing screen (Match vs. Training)
    match/               Live match tracking screen
    training/            Training-mode screen + shot grading
    summary/             Post-session summary + charts
test/
  scoring_engine_test.dart  Pure-Dart tests (runnable without a device)
```

**Key architectural rule:** all game logic (scoring, rally detection, shot
grading) lives in **pure Dart** under `core/`, decoupled from the vision runtime
behind a `VisionService` interface. This lets us:

- unit-test the rules without a camera or GPU,
- swap the underlying model (or feed recorded detections from the benchmark
  harness) without touching UI or rules,
- develop scoring logic in parallel with model work.

---

## 4. Roadmap (incremental)

1. **[done — iteration 1]** Project scaffold, model decision, pure-Dart scoring
   engine + tests.
2. Wire `ultralytics_yolo` `YOLOView`, render live pose + ball overlays.
   - **[done — iteration 9]** `YoloFrameAdapter`: pure, unit-tested mapping from
     the plugin's `onStreamingData` payload (or parsed `List<YOLOResult>`) into
     the runtime-agnostic `FrameResult` — the seam a camera-backed
     `VisionService` funnels live detections through.
   - **[done — iteration 10]** `YoloVisionService`: the camera-backed
     `VisionService` implementation — routes the `YOLOView.onStreamingData`
     callback through `YoloFrameAdapter` onto a `FrameResult` broadcast stream,
     enforcing a strictly-monotonic clock (the `BallTracker` drops
     non-increasing timestamps) and gating emits to the start/stop lifecycle.
     Pure-Dart / unit-tested (no platform channel).
   - **[done — iteration 19]** `CameraMatchScreen`
     (`features/match/camera_match_screen.dart`): the live-camera counterpart to
     the demo `MatchScreen`. It instantiates the real `ultralytics_yolo`
     `YOLOView` platform view (default `YOLOTask.detect` + `yolo11n`, which
     labels both `person` and `sports ball` in one pass), routes its
     `onStreamingData` into a `YoloVisionService`, and drives a
     self-calibrating `MatchController` from that stream — rendering the
     scoreboard, calibration hint, referee-call feed, and manual-resolution
     prompt overlaid on the camera preview. The camera preview, vision service,
     and controller are injectable so the whole wiring is widget-tested
     headlessly (no platform view). Reachable from the home screen's new "Live
     Match" card. Remaining: bundle/point at a fine-tuned ping-pong `.tflite` /
     `.mlpackage` for better ball recall (the COCO `sports ball` class is the
     default fallback).
3. Ball Kalman tracker + rally/point event detection from detections.
   - **[done — iteration 14]** `BallTrajectoryFilter`: a pure-Dart
     constant-velocity Kalman smoother/predictor (two independent 1-D filters,
     one per image axis) that fuses noisy detections and — the point of it —
     *extrapolates* the ball's position through detector dropouts. Wired into
     `BallTracker` (`estimateBallAt`/`estimatedVelocity`), kept in lock-step with
     the accepted samples and reset on ball-lost/reset, so the live overlay draws
     a predicted "ghost" ball through motion-blur gaps instead of freezing. It
     runs alongside the raw-detection event logic — scoring still fires only on
     real detections — so it is purely additive.
   - **[done — iteration 11]** `TableGeometry` table-surface calibration gating
     off-table bounces.
   - **[done — iteration 12]** `TableCalibrator`: pure-Dart auto-calibration
     that infers `TableGeometry` (surface band from a trimmed percentile range
     of ball positions + net line from the two players' x, or the ball-travel
     midpoint) over a warm-up window, so the user just places the phone
     table-side instead of hand-marking the table corners.
   - **[done — iteration 13]** `MatchController` now accepts an optional
     `TableCalibrator` and runs a warm-up phase: it feeds frames to the
     calibrator (scoring nothing, `isCalibrating == true`) until a trustworthy
     `TableGeometry` is inferred, then rebuilds its `BallTracker` on that
     geometry and starts scoring — so the self-calibration is actually in force
     on the live pipeline, not just an isolated component.
4. Connect events → scoring engine → live scoreboard UI.
5. **[done — iteration 8]** Benchmark harness: JSON `ClipFixture` format +
   `BenchmarkRunner` scoring the pipeline's point accuracy against labeled
   clips, ready for SPIN/OpenTTGames conversion.
6. Post-match summary + analytics charts.
   - **[done — iteration 16]** `PlayerMovementAnalyzer`
     (`core/analysis/player_movement.dart`): the first layer to consume the
     *pose* model for performance analytics. It folds each frame's
     `PersonPose`s into per-player footwork metrics — distance travelled,
     lateral/depth court coverage, mobility (distance / tracked second), and
     average stance width — locating each player by the midpoint of their
     visible ankle keypoints (falling back to the box bottom-centre) and
     attributing them to `Player.a`/`Player.b` by the net-split side, matching
     `RallyReferee`'s mapping. It is wired live into `MatchController`
     (`movementFor(player)`), rebuilt on the calibrated geometry so
     side-assignment uses the inferred net line, and surfaced in the Match
     screen's post-match summary panel.
   - **[done — iteration 17]** `RallyAnalyzer`
     (`core/analysis/rally_analyzer.dart`): folds each rally's `BallTracker`
     events and the `RallyReferee`'s ending `PointDecision` into per-rally
     records (stroke count = net crossings, duration, winner/reason) and an
     aggregate `RallyStats` — rally count, average/longest strokes, average
     duration, and short (≤2) / medium (3–5) / long (≥6) buckets, the
     rally-length breakdown match apps headline. Wired live into
     `MatchController` (`rallyStats`, live-only like the movement analytics — not
     rewound by `undo`) and surfaced in the Match screen's post-match summary.
   - **[done — iteration 18]** `BouncePlacementAnalyzer`
     (`core/analysis/bounce_placement.dart`): the shot-map / placement layer.
     `BallTracker` already emits a `BounceEvent` (table-relative x/y + side) on
     every surface touch, but nothing mined it. This folds each bounce into a
     per-side placement distribution — depth-from-net (0 net … 1 baseline)
     bucketed short/middle/deep, lateral spread across the table's near/far
     depth, depth consistency (stddev), and a `depthBins × lateralBins` landing
     heatmap — measured against the (calibrated) `TableGeometry` net/edges. Wired
     live into `MatchController` (`placementFor(side)`, live-only like the
     movement/rally analytics, rebuilt on the calibrated geometry) and surfaced
     as per-side short/mid/deep counts in the Match screen's post-match summary.
7. Training mode: shot segmentation + quality grading.
