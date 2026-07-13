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
docs. Three evaluation stages exist:

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
- **Event detection** (`EventDetectionBenchmark`, iteration 40) — the middle
  layer the other two brackets skip: does `BallTracker` fire a **bounce** (or
  net-crossing) at the right *instant*? It replays a clip's frames through a real
  tracker and matches its emitted `BounceEvent`/`NetCrossEvent`s to a list of
  ground-truth `GroundTruthEvent`s (greedy nearest-in-time within a temporal
  tolerance), yielding per-type precision / recall / F1 and the mean timing
  error. Perception scores whether the model *saw* the ball; scoring scores the
  *final* number; this isolates whether the analysis layer that turns detections
  into rally events — the core of every awarded point — is correct.

- **OpenTTGames converter** (`lib/core/benchmark/openttgames_converter.dart`,
  iteration 31) — the concrete conversion path the plan named. OpenTTGames ships
  a per-game `ball_markup.json` (frame index → ball centre in pixels);
  `clipFixtureFromOpenTtGames(...)` normalizes it to `[0,1]`, synthesizes a small
  ball box, and emits the labeled positions as a fixture's `groundTruthFrames`,
  so a real dataset folder feeds the perception benchmark with no hand-authored
  JSON. Model predictions plug in as `predictedFrames`. OpenTTGames also ships an
  `events_markup.json` (frame index → `bounce`/`net`/`empty`);
  `openTtGamesBounceEvents(...)` (iteration 40) converts its `bounce` labels into
  `GroundTruthEvent`s on the same ms clock, feeding the event-detection
  benchmark. (`net` = ball *hitting* the net, a different event from the
  tracker's over-the-net crossing, so it is intentionally not mapped.) Passing
  that map as `clipFixtureFromOpenTtGames(..., eventsMarkup:)` (iteration 42)
  attaches the events as the fixture's `groundTruthEvents` so one converted clip
  feeds all three benchmark stages.
- **Runnable entrypoint** (`bin/benchmark.dart` + `benchmark_corpus.dart`,
  iteration 41) — `dart run bin/benchmark.dart` discovers `benchmark/clips/*.json`
  (via `loadClipDirectory`), scores them through the scoring + perception +
  event-detection stages (`buildCorpusReport`; Stage 3 added iteration 42, scoring
  clips that carry `groundTruthEvents`), and prints one consolidated report, so
  the corpus can be evaluated outside `flutter test` and gate CI.

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
   - **[done — iteration 43]** `VisionModelProfile`
     (`core/vision/vision_model_profile.dart`): the model-selection seam that
     makes that "point at a fine-tuned model" a *single coherent choice*. Picking
     a model is not just picking a file — the model fixes the inference `task`,
     the class *labels* it emits (stock COCO calls the ball `sports ball`; a
     fine-tuned model may call it `ball`), and the confidence bar those
     detections deserve. Before this those lived in two places: the model path in
     the camera screen and the label/threshold decode config (`YoloFrameConfig`)
     in the adapter — and the camera screens built a *default* `YoloVisionService`,
     so a custom decode config could not even reach the live pipeline. A
     `VisionModelProfile` bundles `modelPath` + `task` + `frameConfig` and its
     `createVisionService()` builds a `YoloVisionService` whose adapter decodes
     *that* model's output. Two profiles ship: `cocoDetectProfile` (the
     zero-setup `yolo11n` default) and `pingPongDetectProfile` (the drop-in slot
     for a fine-tuned multi-class person+ball detector at
     `assets/models/pingpong.tflite`, with a lower ball-confidence bar for higher
     recall). Both `CameraMatchScreen` and `CameraTrainingScreen` now take a
     single `model` param (default `defaultVisionModel`), so enabling the
     fine-tuned model once bundled is a one-line change.
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
   - **[done — iteration 29]** Kalman-prediction **outlier gate** in
     `BallTracker` (`maxJump`): the complement to iteration 14's filter. The
     filter *bridged* frames where the detector loses the ball; this rejects
     frames where the detector finds the *wrong* ball. Once a trajectory is
     established (≥2 accepted samples), a detection landing more than `maxJump`
     (normalized distance) from the constant-velocity prediction is treated as a
     spurious detection — the detector latching onto a round object or bright
     logo elsewhere in the frame — and routed through the missing-ball path
     instead of accepted, so it can't teleport the trajectory and manufacture a
     bogus net-cross/bounce → mis-scored point; persistent spurious detections
     end the rally via the normal `BallLostEvent` after `maxGapFrames`. The
     residual is gated *after* the CV prediction, so it is time-scale-invariant
     (a true ball's residual is measurement noise + gentle bounce reversal, far
     below a conservative `0.4`). Disabled by default (`maxJump == null`) so the
     raw-detection scoring path and every synthetic-clip test are unchanged; the
     live-camera `CameraMatchScreen` enables it (`0.4`) where real detector
     false-positives occur, and `MatchController` preserves it across the
     post-calibration tracker rebuild.
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
     **Iteration 34** mined the previously-unused `Rally.winner` field into a
     rally-length *win* breakdown: `RallyStats.ralliesWonBy(player)` and
     `ralliesWonByLength(player, RallyLength)` (with the `rallyLengthOf(strokes)`
     short/medium/long classifier and a `hasWinData` guard) answer "who thrives
     in short first-strike vs long grinding exchanges" — long-rally dominance
     being a headline endurance/consistency signal. `report()` gains a "Rally
     wins" and "long (≥6) wins" line (flowing into `buildMatchReport`), and the
     Match screen surfaces the long-rally win split.
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
   - **[done — iteration 21]** `ShotMapView`
     (`features/summary/shot_map.dart`): the *visual* shot-map — the headline
     placement view of match apps. It renders every recorded `BouncePlacement`
     from both sides' `SidePlacementStats` as translucent dots on a schematic
     top-down table (net a vertical centre line; each side's depth fans outward
     toward its baseline; lateral maps to across-table y), so overlapping
     landings read as a density/heat cloud. The placement→pixel geometry is a
     pure, unit-tested `shotMapPosition(BouncePlacement)` seam so the map math is
     verifiable without pixels; the `CustomPaint` painter just draws those
     points. Surfaced in the Match screen's post-match summary panel below the
     per-side short/mid/deep counts.
   - **[done — iteration 24]** `PlayerPositionMapView`
     (`features/summary/player_map.dart`): the *player-positioning* /
     court-coverage heatmap — the headline "where did each player stand" view of
     match apps. `PlayerMovementAnalyzer` mined the pose model into aggregate
     footwork metrics (distance, coverage span, average position) since iteration
     16 but discarded the individual foot-position samples; the analyzer now
     retains them (`positionsFor(player)`) and this view renders both players'
     samples as per-player translucent heat clouds on a schematic top-down table,
     each half re-centred around the calibrated net line so the net always sits
     at the map centre. The frame-foot → pixel geometry is a pure, unit-tested
     `playerMapPosition(foot, netX:)` seam; the `CustomPaint` painter just draws
     the clouds. Surfaced in the Match screen's post-match summary panel below the
     shot map, shown once either player was tracked.
   - **[done — iteration 23]** `MomentumChartView`
     (`features/summary/momentum_chart.dart`): the score-progression / momentum
     timeline. `MatchSummary` already keeps the ordered `ScoredPoint` log, but it
     was only reduced to aggregate counts. This plots the *running lead* — the
     cumulative point differential (Player A − Player B) after each rally — as a
     filled-area timeline that rides above the centre line while A leads and dips
     below while B leads, so runs and comebacks are visible at a glance. The
     point-log → differential math is a pure, unit-tested `momentumSeries(points)`
     seam (leading `0` start, `points.length + 1` entries); the `CustomPaint`
     painter just draws the curve. Surfaced in the Match screen's post-match
     summary panel above the rally/placement stats.

   - **[done — iteration 26]** Serve / receive point analytics
     (`core/analysis/match_summary.dart`). The `ScoringEngine` tracks who serves
     each point, but the durable `ScoredPoint` log never captured it, so the
     server was lost to analytics. Each `ScoredPoint` now records the `server`
     (captured in `MatchController` *before* `awardPoint`, since awarding
     advances the serve rotation), and `MatchSummary` derives the headline
     serve-effectiveness stats: `servePointsPlayedBy` / `servePointsWonBy` (the
     serve-hold count), `receivePointsWonBy` (return-of-serve breaks), and
     `serveWinRateFor` (own-serve win fraction, null when a player served no
     recorded points). The `server` field is nullable/back-compatible, so
     serve analytics count only points where it is known (`hasServeData`). The
     per-player serve win rate is surfaced in the Match screen's post-match
     summary and, via `summary.report()`, flows into the exported
     `buildMatchReport`.
   - **[done — iteration 28]** Per-game score breakdown
     (`core/analysis/match_summary.dart`). The `ScoringEngine` tracks games won,
     but the durable `ScoredPoint` log never recorded *which* game each point
     belonged to, so the summary could report the final games total (e.g. 3–1)
     yet not the game-by-game score line (`11–7, 9–11, 11–8`) that every
     scoreboard shows. Each `ScoredPoint` now records its `gameIndex` (games
     completed *before* the point, captured in `MatchController` before
     `awardPoint`, since awarding may complete the game), and `MatchSummary`
     exposes `gameScores` — a `GameScore` (pointsA/pointsB, plus a `winnerAt`
     helper) per game reconstructed by counting each game's points per player,
     including a trailing in-progress game. The `gameIndex` field is
     nullable/back-compatible (`hasGameData`), a "Games:" line is added to
     `report()` (so it flows into the exported `buildMatchReport`), and the
     per-game line is surfaced in the Match screen's post-match summary.
   - **[done — iteration 36]** Game-point / pressure analytics
     (`core/analysis/match_summary.dart`). Iteration 28 recorded each point's
     `gameIndex`, letting `MatchSummary` reconstruct the game-by-game score, but
     the *within-game running score* it walks was never mined for the headline
     clutch stat every match app shows: game-point conversion. `MatchSummary`
     now replays the point log per game and tags each point with which player
     (if any) held a **game point** going into it — would win the game by
     winning that rally, detected with the same ITTF `_scoringWinsGame`
     (target reached with a 2-point lead) rule the engine uses, so it handles
     deuce correctly and at most one player is ever at game point. From that it
     derives `gamePointsHeldBy` / `gamePointsConvertedBy` (chances to close the
     game and how many were taken) and the mirror `gamePointsFacedBy` /
     `gamePointsSavedBy` (game points survived on the receiving end), plus a
     `gamePointConversionRateFor` and a `hasPressureData` guard (needs game
     indexing). `report()` gains a per-player "game points: converted X/Y, saved
     Z/W" line (flowing into the exported `buildMatchReport`), and the Match
     screen surfaces the same conversion/save split.
   - **[done — iteration 37]** Match-tension analytics
     (`core/analysis/match_summary.dart`). Iteration 23's `MomentumChartView`
     *plotted* the running point differential (A−B after each rally) but no
     numeric stat mined that same series — how close the match actually was was
     only ever a picture. `MatchSummary` now derives the tension trio from the
     point log: `leadChanges` (how many times the player who is ahead switched,
     counting through ties), `largestLeadBy` (biggest lead each player ever
     held), and `largestDeficitOvercomeBy` (the worst deficit a player trailed by
     and then erased to at least level — for the winner, their comeback factor).
     `report()` gains a "Lead changes: N" line (marked "wire-to-wire" at 0) plus
     per-player "biggest lead" and "overcame an N-point deficit" lines (flowing
     into `buildMatchReport`), and the Match screen surfaces lead changes under
     the momentum chart and the biggest-lead/comeback split per player.
   - **[done — iteration 32]** Real-world ball-speed analytics
     (`core/analysis/ball_speed.dart`). Every prior analytics layer worked in the
     vision pipeline's *normalized* `[0,1]` coordinates, which carry no physical
     meaning — so the headline "Ball AI"-style **km/h** ball speed was un-derived.
     `BallSpeedEstimator` turns the ball's along-table (frame-x) displacement
     between consecutive detections into metres-per-second / km/h, using the
     calibrated `TableGeometry` as the ruler: the phone sits on the *side* of the
     table so the regulation **2.74 m** length spans the frame horizontally, and
     one x-unit therefore maps to `2.74 / (right - left)` metres. Only the
     horizontal component is scaled (a side camera can't recover the foreshortened
     near/far depth or the ball's height to scale), it skips intervals that bridge
     a ball-loss gap (`maxGapMs`) or imply a physically-impossible speed
     (`maxPlausibleKmh`, spurious detector teleports), and it accumulates
     `maxKmh` / `averageKmh`. Wired live into `MatchController`
     (`maxBallSpeedKmh` / `averageBallSpeedKmh` / `hasBallSpeedData`, live-only
     and rebuilt on the calibrated table span like the movement/rally/placement
     analytics), surfaced as a "Top ball speed" line in the Match screen's
     post-match summary, and added as a "Ball speed" section to the exported
     `buildMatchReport`.
   - **[done — iteration 35]** Tracking-quality / detection-health analytics
     (`core/analysis/tracking_quality.dart`). The objective's deployment story —
     "place the phone table-side and let the app keep score" — only holds if the
     phone is positioned so the model can actually see the ball and both
     players, yet every prior analytics layer *assumed* the detections were good
     and none measured whether they were. The pipeline has carried a
     per-detection `Detection.confidence` (and per-keypoint confidences) since
     iteration 1, but they were only used to *select* the best detection, never
     mined into a health signal. `TrackingQualityAnalyzer` folds every frame
     (including calibration warm-up — detection health is scoring-independent)
     into `ballDetectionRate`, `averageBallConfidence`, and `twoPlayerRate` (the
     fraction of frames where *both* ends of the table are in view), rolls them
     into a weighted `qualityScore` / A–F `grade`, and derives a plain-language
     placement `hint` targeting the weakest signal ("Both players are often out
     of frame — move the phone back…"). Wired live into `MatchController`
     (`trackingQuality`, observed on every frame and never rebuilt on
     calibration so it spans the whole session), surfaced as a "Tracking
     quality: grade …" line in the Match screen's post-match summary, and added
     as a "Tracking quality" section to the exported `buildMatchReport`.
   - **[done — iteration 25]** `buildMatchReport`
     (`core/analysis/match_report.dart`): the unified, exportable text report.
     Every prior analytics layer — the `MatchSummary` scoring breakdown,
     `RallyStats` rally-length distribution, per-player `PlayerMovementStats`
     footwork, and per-side `SidePlacementStats` bounce placement — was only ever
     rendered piecemeal into its own summary-panel widget; there was no single
     shareable artifact answering "how did this match go?", which the objective's
     "produce summary" goal asks for. `buildMatchReport(controller)` folds all of
     those live analytics into one deterministic, Flutter-free string (scoring →
     rally → both players' movement → both sides' placement, with "not
     tracked" / "no bounces" fallbacks). The Match screen's post-match summary
     panel now has a **Copy report** action that writes it to the clipboard
     (`Clipboard.setData`, no new dependency) with a confirmation snackbar.
   - **[done — iteration 38]** `buildMatchReportJson` / `matchReportJsonString`
     (`core/analysis/match_report_json.dart`): the *structured* (JSON) companion
     to `buildMatchReport`. The text report is human-readable but can't be
     re-parsed — it can't be stored as match history, diffed across sessions, or
     fed to another tool/backend — so the same analytics are also emitted as a
     versioned (`matchReportSchemaVersion`), JSON-encodable `Map`: score, the
     `MatchSummary` breakdown (per-player points/serve/game-points/lead-tension),
     rally-length + win distribution, ball speed, tracking quality, per-player
     movement, and per-side placement. Optional sections are explicit `null`s
     (not missing keys) so a reader can rely on the schema shape, and doubles are
     rounded for compact, deterministic output that round-trips through
     `dart:convert`. The Match screen's post-match panel gained an **Export JSON**
     action next to Copy report that writes the pretty-printed string to the
     clipboard.
   - **[done — iteration 45]** `SessionHistoryStore`
     (`core/history/session_history_store.dart`): the across-session *memory* the
     structured exports were built for. Iterations 38/39 made the match and
     training reports JSON-encodable specifically so they could be "stored as
     match history, diffed across sessions" — but nothing actually kept them;
     copy-to-clipboard was the only sink, so closing a screen lost the session
     and the objective's "keep track of the scores … and produce summary" goal
     had no persistence. `SessionHistoryStore` takes a target `Directory` (the
     app passes the platform documents dir; a test passes a temp dir) and
     `save(kind:, report:)` / `list()` / `load(id)` / `delete(id)` persist each
     `buildMatchReportJson` / `buildTrainingReportJson` map as a small wrapped
     (`{kind, savedAt, report}`) `<kind>-<millis>.json` file. Saves never
     overwrite an earlier same-millisecond session (a `-2`/`-3` suffix is
     appended), `list()` returns newest-first and silently skips corrupt/foreign
     `*.json` files, and — like the rest of `core/` — it is pure Dart (`dart:io`
     + `dart:convert`, no Flutter/plugin) so it is unit-tested end-to-end against
     a temp dir with no device.

7. Training mode: shot segmentation + quality grading.
   - **[done — iteration 44]** Training-mode tracking-quality / detection-health.
     Iteration 35 added `TrackingQualityAnalyzer` for the match path — the
     phone-placement health signal behind the objective's "place the phone
     table-side" story — but training mode had none, though the story applies
     equally to practice. The analyzer's player component hard-coded
     `twoPlayerRate` (both ends of the table), which is match-specific: a
     training drill has a *single* player against a rebound net. The analyzer now
     takes a `requireBothPlayers` flag (default `true`, unchanged for the match)
     and, when `false`, scores `qualityScore` / `hint` / `report` on
     `playerVisibilityRate` = `anyPlayerRate` with single-player wording ("You
     are often out of frame…"). Both `TrainingScreen` (demo) and
     `CameraTrainingScreen` (live) now own a `TrackingQualityAnalyzer(
     requireBothPlayers: false)`, observe every frame, reset it on Restart, and
     surface a "Tracking quality: grade — hint" line in the end-of-session report
     (also appended to the Copy-report clipboard export).
   - **[done — iteration 39]** Training structured (JSON) export
     (`core/training/training_report_json.dart`): the practice-mode companion to
     iteration 38's `buildMatchReportJson`. Training only ever produced the
     human-readable `TrainingSummary.report()` text blob, so a drill couldn't be
     stored as history, diffed across sessions, or fed to another tool.
     `buildTrainingReportJson(summary, config:)` / `trainingReportJsonString`
     emit the full session analytics — config target side, session grade,
     depth/lateral placement + consistency, pace (km/h), tempo/rhythm, grade
     buckets, and the per-shot list — as a versioned
     (`trainingReportSchemaVersion`), JSON-encodable `Map` that round-trips
     through `dart:convert`. Optional sections (pace km/h, tempo) are explicit
     `null`s so the schema shape is stable. Both `TrainingScreen` (demo) and
     `CameraTrainingScreen` (live) session reports gained an **Export JSON**
     action next to Copy report that writes the pretty-printed string to the
     clipboard.
   - **[done — iteration 33]** Training real-world shot-speed (km/h)
     (`core/training/shot_analyzer.dart`). Each `Shot` recorded only a
     *normalized* `speed` (units/s) which — as iterations 6/7 noted — saturates
     against an arbitrary `referenceSpeed`, so a training drill had no legible
     pace metric, the practice-mode gap left by iteration 32's match ball-speed
     radar. `Shot` now also carries `speedKmh`: its peak along-table (frame-x)
     approach speed scaled to real-world km/h via the same table ruler as
     `BallSpeedEstimator` (`TrainingConfig.metersPerUnitX = tableLengthMeters /
     (right - left)`, the ITTF 2.74 m length spanning the frame x-axis from a
     side camera). `TrainingSummary` exposes `maxSpeedKmh` / `averageSpeedKmh`,
     and `report()` gains a "Ball speed: N km/h top, M km/h avg" line (omitted
     when no shot carries a scale), so the physical pace flows into both training
     screens' on-screen summary and Copy-report export with zero widget changes.
   - **[done — iteration 30]** Training tempo / rhythm analytics
     (`core/training/shot_analyzer.dart`). Each `Shot` has carried a
     `timestampMs` since iteration 6, but `TrainingSummary` only ever reduced it
     to the total `durationMs` — the *regularity* of a drill's tempo (a headline
     coaching metric: is the player feeding at a steady, repeatable rhythm?) was
     un-mined. `TrainingSummary` now exposes `shotIntervalsMs` (the gaps between
     consecutive shots), `averageIntervalMs`, `shotsPerMinute` (drill cadence),
     and `rhythmConsistency` — a `[0, 1]` metronome score from the coefficient of
     variation (stddev / mean) of the inter-shot intervals, the tempo companion
     to the depth/lateral placement `consistency` metrics. When a session has ≥2
     shots, `report()` (and thus both training screens' Copy-report export and
     on-screen summary) gains a "Tempo: N shots/min" and "Rhythm consistency: M%"
     line; single-shot sessions omit them since no interval exists.
   - **[done — iteration 27]** Training placement analytics + shareable report.
     `Shot` has carried a `lateral` (across-table) landing coordinate since
     iteration 22, but it fed only the visual `TrainingShotMapView` — no
     aggregate stat mined it, so a drill's *across-table* grouping was invisible
     to the textual summary. `TrainingSummary` now exposes `averageLateral` and
     `lateralConsistency` (the companion to the existing depth-only
     `consistency`, from the population stddev of `Shot.lateral`), and
     `report()` reports both a "Depth consistency" and a "Lateral consistency"
     line so a player can see whether they are grouping shots into a spot on
     both axes. Both training screens' end-of-session reports gained a **Copy
     report** action (`Clipboard.setData`, no new dependency, confirmation
     snackbar) — the training-mode counterpart to the match `buildMatchReport`
     export, closing the "produce summary" gap for practice sessions. The demo
     `TrainingScreen`'s finished-state panel is now `Flexible` so the scrollable
     report (with the placement map) fits instead of overflowing the column.
   - **[done — iteration 20]** `CameraTrainingScreen`
     (`features/training/camera_training_screen.dart`): the live-camera
     counterpart to the demo `TrainingScreen`. It instantiates the real
     `ultralytics_yolo` `YOLOView` platform view (default `YOLOTask.detect` +
     `yolo11n`, which labels `sports ball`), routes its `onStreamingData` into a
     `YoloVisionService`, and drives a `ShotAnalyzer` from that stream —
     overlaying the session grade/shot-count header, the net line + target
     landing band, the tracked ball, and a rolling graded-shot feed on the
     camera preview. A "Finish" action freezes the session and shows the
     end-of-session report; "Restart" resets the analyzer and resumes. The
     camera preview and vision service are injectable so the whole wiring is
     widget-tested headlessly (no platform view). Reachable from the home
     screen's new "Live Training" card. This closes the objective's second
     priority (analyse training-shot quality) for the live-camera path, leaving
     only the shared fine-tuned ping-pong model bundling (roadmap item 2).
   - **[done — iteration 22]** `TrainingShotMapView`
     (`features/training/training_shot_map.dart`): the *visual* training
     placement map, the practice-mode counterpart to the match `ShotMapView`.
     Each graded `Shot` now carries not just its landing `depth` (net → baseline)
     but also a `lateral` (across-table) position — computed in `ShotAnalyzer`
     from the bounce's `y` against the (calibrated) `TableGeometry`, exactly like
     `BouncePlacementAnalyzer`. The view draws every shot as a grade-coloured dot
     on a schematic target half (depth → x, lateral → y) with the target landing
     band highlighted, so a player can *see* how tightly their drill shots
     cluster. The placement→pixel geometry is a pure, unit-tested
     `trainingShotMapPosition(Shot)` seam; the `CustomPaint` painter just draws
     the points and band. Surfaced in both the demo `TrainingScreen` and the
     live `CameraTrainingScreen` end-of-session reports.
