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
     `VisionService` funnels live detections through. When more than `maxPeople`
     persons are detected it keeps the two largest by box **area** (iteration 64
     fix; was box width). With the phone at the side of the table the players are
     seen side-on — narrow but tall boxes — so a width-only cap would drop the
     real players in favor of a wide, short spectator facing the camera.
     A ball candidate whose *smaller* box dimension exceeds
     `YoloFrameConfig.maxBallRelativeSize` (iteration 65; both live model
     profiles set 0.25, off by default in the bare config) is rejected as too
     large to be a ping-pong ball — the generic COCO "sports ball" class fires
     on heads, logos, and gym basketballs, and such a gross false positive would
     otherwise seed a bad trajectory *before* the `BallTracker` `maxJump` gate
     (which only engages once a trajectory exists) can catch it. Gating on the
     smaller dimension keeps a motion-blurred ball (elongated along one axis).
     Symmetrically, a person detection shorter than
     `YoloFrameConfig.minPersonRelativeHeight` (iteration 67; both live profiles
     set 0.25, off by default) is rejected as a distant background bystander: the
     iteration-64 area cap only chooses *among* more than `maxPeople` boxes, so
     when only one real player and one far-away bystander are detected the
     bystander would still be accepted and corrupt the net-split side assignment
     and movement/coverage analytics. It gates on box **height** (not width/area)
     because a legitimate side-on player is narrow but always tall.
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
     - **[done — iteration 69]** Live tracking overlay. The demo `MatchScreen`
       rendered a top-down `_TableView` of what the pipeline is following (player
       boxes, tracked/predicted "ghost" ball, net line) since iterations 4/14,
       but the *production* live-camera screen drew nothing of its own
       interpretation on the preview — only the scoreboard and call feed — so the
       user couldn't see whether the app was actually tracking the right players
       and ball. `_LiveTrackingOverlay` now draws each frame's player bounding
       boxes, the ball (or the dimmed Kalman-predicted ghost when the detector
       loses it, via `tracker.estimateBallAt`), and the calibrated net line
       (suppressed during calibration) coordinate-aligned over the camera preview,
       in the same normalized `[0,1]` space the detections use. Wrapped in
       `IgnorePointer` so the undetermined-point prompt and match-over panel below
       it still receive taps — the live "Ball AI" view of what's being followed.
     - **[done — iteration 71]** Live-training tracking parity. The live
       `CameraTrainingScreen` overlay had drawn only the raw detected ball (plus
       target band + net line) — no player box and no ghost ball through
       dropouts — so it lagged the iteration-69 match overlay. `_TargetOverlay`
       now also draws each frame's player bounding box(es) and, when the detector
       loses the ball, the dimmed Kalman-predicted "ghost" ball via a new
       `ShotAnalyzer.tracker` getter (`tracker.estimateBallAt`), reaching
       tracking-overlay parity with the match path.
     - **[done — iteration 73]** Player-side selection. `CameraTrainingScreen`
       hardcoded `const TrainingConfig()` (`playerSide == left`, so target =
       right), so a player who set the phone on the *right* side of the table
       had their shots cross the net to the left and never segment/grade — a real
       usability gap parallel to the match screen's iteration-70/72 first-server
       and format pickers. A new `_PlayerSidePicker` ("I hit from: Left / Right")
       is offered until the first shot is graded and rebuilds the `ShotAnalyzer`
       on a `TrainingConfig.copyWith(playerSide:)`, flipping both the target band
       and the shot-segmentation target half; it locks away once scoring begins.
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
   - **[done — iteration 66]** Prediction-aware ball-candidate recovery. The
     `maxJump` gate above rejects a spurious *primary* ball, but the real ball is
     often *also* detected the same frame at lower confidence (a round object
     out-scoring the true ball). `YoloFrameAdapter` now keeps those alternatives
     on `FrameResult.ballCandidates` (best-first) instead of discarding them, and
     when the gate rejects the primary, `BallTracker` falls back to the candidate
     closest to the Kalman prediction that itself lands within `maxJump` — so the
     detector latching onto the wrong object no longer forces a ball-lost when
     the real ball was in view. Purely additive: `ballCandidates` is empty on the
     synthetic path and recovery only runs when gating is active.
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
   - **[done — iteration 61]** End changes between games. In table tennis the
     players swap ends after every game while the phone stays put, so the
     camera's physical left/right half maps to the *opposite* scoring `Player`
     from the next game on. `RallyReferee.switchEnds()` flips its side→player
     mapping (`leftPlayer`), and `MatchController` (opt-in
     `switchEndsBetweenGames`, symmetric on `undo`) calls it whenever an awarded
     point completes a game — so a real multi-game match keeps attributing
     bounces to the correct player instead of mis-scoring every even game. Off
     by default (scripted synthetic clips never physically switch ends); the
     live-camera `CameraMatchScreen` enables it.
   - **[done — iteration 62]** Deciding-game mid-game end change (ITTF 2.13.4).
     In the last possible game the players *also* swap ends the first time
     someone reaches half the game points (5 in an 11-point game), so the second
     half of a decider is played from the swapped ends. `MatchController` fires a
     once-per-decider mid-game `switchEnds()` (latched, reversed symmetrically on
     `undo`) when an awarded point puts a player at/over `pointsPerGame ~/ 2` in
     the 1–1-games (best-of-N) deciding game — gated behind the same
     `switchEndsBetweenGames` flag so scripted clips stay unchanged.
   - **[done — iteration 63]** Movement analytics follow the end change too.
     `PlayerMovementAnalyzer.switchEnds()` flips its side→player mapping (and
     breaks distance continuity so the one-off cross-court walk isn't logged as a
     teleport jump), and `MatchController._switchEnds()` calls it in lock step
     with `RallyReferee.switchEnds()` on every between-games and mid-decider end
     change (and their `undo` reversals). Footwork/coverage samples now stay
     attributed to the correct player after the players change ends, closing the
     iteration-61/62 follow-up.
   - **[done — iteration 70]** First-server selection. `ScoringEngine` defaulted
     to `Player.a` serving with no way to record who actually serves first, so
     the "who's serving" indicator and the iteration-26 serve/receive analytics
     were systematically wrong whenever B served first. `ScoringEngine.setFirstServer()`
     (valid only before the first point) sets `server`/`initialServer`,
     `MatchController.setFirstServer()`/`matchNotStarted` expose it, and
     `CameraMatchScreen`'s scoreboard shows a "First server: A/B" chip picker
     while the match hasn't started (throughout calibration too).
   - **[done — iteration 72]** Match-format selection. The live match was
     hardcoded to best-of-5 11-point games with no way to pick a shorter/longer
     match. `ScoringEngine.setMatchFormat({pointsPerGame, bestOf})` (valid only
     before the first point, rejecting invalid formats) reconfigures the format
     preserving the chosen server, `MatchController.setMatchFormat()` exposes it,
     and `CameraMatchScreen`'s scoreboard shows a "Best of: 3/5/7" chip picker
     alongside the first-server picker while the match hasn't started.
   - **[done — iteration 74]** Manual point correction. The live match could
     only auto-score from the ball path, resolve an *undetermined* in-flight
     loss, or `undo` — but if the vision missed a rally entirely (occlusion,
     ball out of frame, a serve the tracker never picked up) the score silently
     drifted with no way to correct it upward. `MatchController.awardManualPoint(winner)`
     directly awards a point (no-op once the match is over), recording it in the
     point log with a new `PointReason.manual` so it undoes, switches ends, and
     feeds the summary exactly like an auto-scored point, stamped at the last
     frame's timestamp. `CameraMatchScreen`'s call-feed panel shows "Missed a
     point? +A / +B" buttons once scoring is live (hidden during calibration).
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
     - **[done — iteration 68]** Jitter deadband. The `distanceTravelled` /
       `mobilityPerSecond` footwork metric summed the raw frame-to-frame foot
       displacement, so a stationary player's few-pixel pose/box jitter
       accumulated across a match into a systematically inflated distance.
       `PlayerMovementAnalyzer.minStep` now accumulates distance from the last
       *counted* position only once the feet drift at least that far, so
       in-deadband noise is ignored while genuine (even slow, steady) movement
       still crosses the threshold — the footwork analog of the ball tracker's
       `minBounceSpeed` jitter rejection. `MatchController.movementJitterThreshold`
       plumbs it through (preserved across the calibration rebuild); `0` (the
       default) keeps scripted synthetic clips' exact distances, and
       `CameraMatchScreen` enables `0.01` on the live on-device path.
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
   - **[done — iteration 52]** Match coaching insights
     (`core/analysis/match_insights.dart`). The match path measured serve holds,
     return breaks, and game-point conversion but — exactly like the training
     side before iteration 51's `TrainingFeedback` — never *prioritized* one into
     "what to work on". `MatchInsights(summary)` folds the per-player metrics into
     scored coachable `InsightDimension`s (serve effectiveness, return of serve,
     closing games), each derived from `MatchSummary` with the denominator-guard
     so a dimension only appears when there is data for it. `insightsFor(player)`
     names the weakest as that player's focus (or encouragement when even the
     weakest clears a 0.6 bar) and the strongest as a confirmed strength; a
     `report()` "Coaching insights" section (flowing into `buildMatchReport`) and
     a "Coaching" block in the Match summary panel surface each player's focus
     cue. Pure Dart, unit-tested from a synthetic `MatchSummary`.
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
     - **[done — iteration 75]** Live km/h readout. The estimator reduced every
       reading into the whole-match `maxKmh` / `averageKmh` aggregates but never
       exposed the *latest* one, so the "Ball AI"-style live speed number that
       flashes beside the ball during play was un-surfaced. `lastKmh`
       (→ `MatchController.currentBallSpeedKmh`) exposes the most recent accepted
       reading, and `CameraMatchScreen`'s `_LiveTrackingOverlay` renders it as a
       "N km/h" label next to the tracked ball — drawn only while the ball is in
       view, since a detector dropout leaves the reading stale.
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
     clipboard. *(iteration 53)* The structured export now also carries a
     `coaching` section (`MatchInsights` per-player focus/strength/dimension
     scores) that the text `buildMatchReport` gained in iteration 52 but the JSON
     companion had lacked — so a persisted history session, not just the on-screen
     panel, can surface each player's "focus next" cue. The training JSON export
     (`buildTrainingReportJson`) gained the same `coaching` section from
     `TrainingFeedback` (iteration 51), closing the parallel gap.
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
   - **[done — iteration 46]** Session-history UI: producer + consumer wiring for
     the iteration-45 store, which until now had neither. `SessionHistoryScreen`
     (`features/history/session_history_screen.dart`) lists stored sessions
     newest-first (kind icon, a pure/unit-tested `sessionHeadline` derived from the
     report — match score or training grade/shot-count — and save time), opens any
     one to a read-only detail view of its pretty-printed report, and can delete;
     it takes an injectable `store` (tests) and otherwise resolves the on-device
     store lazily via `defaultSessionHistoryStore()`
     (`core/history/history_store_provider.dart`, the `path_provider`
     documents-dir seam). The training report gained a "Save to history" action
     (an injectable `historyStoreLoader` on `TrainingScreen`) that persists
     `buildTrainingReportJson` via the store, and the home screen gained a
     "History" card. Screen tests run against an in-memory fake store so they can
     pump/settle normally (a `testWidgets` body's real `dart:io` only advances
     under `runAsync`); the on-disk store keeps its own temp-dir unit tests.
   - **[done — iteration 47]** Match-side "Save to history" producer, closing the
     parallel gap iteration 46 left (only training could save). `MatchScreen`'s
     post-match summary panel gained a "Save to history" action (an injectable
     `historyStoreLoader`, default `defaultSessionHistoryStore`) that persists
     `buildMatchReportJson` as a `SessionKind.match` record, so both modes now
     feed the history screen. Reaching that panel also required making it
     scrollable (`SingleChildScrollView` in a `Flexible`) — previously it was an
     unbounded `Column` that overflowed once a match actually completed — and
     `MatchScreen` gained a `matchControllerBuilder` seam so tests can inject a
     short-match controller (`ScoringEngine(pointsPerGame: 3, bestOf: 1)`) that
     the demo rallies finish, exercising the summary panel end-to-end for the
     first time.
   - **[done — iteration 57]** Live-camera "Save to history" parity. Iterations
     46/47 added the producer only to the scripted *demo* screens
     (`TrainingScreen`/`MatchScreen`), so the actual production live-camera path
     could not persist a session — `CameraTrainingScreen`'s end-of-session report
     had Copy report / Export JSON but no Save to history. It now carries the same
     injectable `historyStoreLoader` (default `defaultSessionHistoryStore`) and a
     "Save to history" action that persists `buildTrainingReportJson` as a
     `SessionKind.training` record, closing the demo-vs-live gap for training.
     (`CameraMatchScreen` still has no end-of-match summary panel at all — a
     larger follow-up than a single producer button.)
   - **[done — iteration 58]** Live-camera match summary panel. Iteration 57 left
     `CameraMatchScreen` (the production camera match path) with only a "Match
     over" call-feed line — unlike the demo `MatchScreen`'s full `_SummaryPanel`,
     it could not Copy/Export/Save any analytics. It now shows a `_MatchOverPanel`
     once the match ends: winner, points/games, top ball speed, tracking-quality
     grade, and per-player points/forced-errors, plus the same Save to history /
     Export JSON / Copy report actions (injectable `historyStoreLoader`, default
     `defaultSessionHistoryStore`; reports composed via `buildMatchReport` /
     `matchReportJsonString` / `buildMatchReportJson`), reaching demo-vs-live
     parity for the match path.
   - **[done — iteration 59]** Live-camera visual analytics. Iteration 58's
     `_MatchOverPanel` surfaced only *text* stats — the demo `MatchScreen`'s
     `_SummaryPanel` also renders the headline "Ball AI"-style visual charts
     (`MomentumChartView`, `ShotMapView`, `PlayerPositionMapView`) which the live
     production match path still lacked. The `_MatchOverPanel` now renders the
     momentum lead-timeline, the bounce-placement shot map, and the player
     coverage heatmap (each guarded by its own data-present check, derived from
     `controller.summary.points` / `placementFor` / `positionsFor` +
     `geometry.netX`), completing visual demo-vs-live parity for the match path.
   - **[done — iteration 48]** Across-session progression / trends. Iterations
     45–47 persisted each session and listed them one-by-one, but nothing mined
     the *collection* — yet both JSON exporters name "diff pace/placement/rhythm
     across sessions" as their whole reason for existing. `SessionTrends`
     (`core/history/session_trends.dart`) folds the stored records into
     training-progression metrics (first→latest average-score improvement, best
     session, personal-best km/h, a match tally) by parsing the stored report
     maps only — pure Dart, no Flutter/plugin, unit-tested end-to-end. The
     history screen surfaces it as a compact "Training progress" card shown once
     ≥ 2 drills are saved.
   - **[done — iteration 49]** Consistency progression. `TrainingTrendPoint`
     already parsed each session's `depthConsistency` (placement stddev, lower =
     tighter) and `rhythmConsistency` (metronome score, higher = steadier), but
     `SessionTrends` only trended average-score and km/h — those two signals were
     captured-but-unconsumed. Added `depthConsistencyImprovement` (first − latest,
     positive = tighter placement) and `rhythmConsistencyImprovement` (latest −
     first, positive = steadier tempo), each computed over the sessions that
     actually recorded the metric, with matching `report()` lines and a compact
     "Placement tighter · Rhythm up N%" line on the Training-progress card.
   - **[done — iteration 50]** Visual progress chart. `SessionTrends` was only
     ever surfaced as a first→latest text delta, never as the *shape* of the
     progression. `ProgressChartView`
     (`lib/features/history/progress_chart.dart`) plots one point per saved
     training drill (oldest → latest) as a line chart on the Training-progress
     card, the across-session analog of the match-side `MomentumChartView`.
     Rendering splits from math via the pure, unit-tested `progressChartPoints`
     seam (per-session score → normalized `[0,1]×[0,1]` plot coordinates, rising
     line = improving) and a thin `_ProgressPainter`.
   - **[done — iteration 54]** Shot-speed progression. `TrainingTrendPoint`
     parsed each session's peak `maxSpeedKmh` but `SessionTrends` only reduced it
     to a personal-best `bestMaxSpeedKmh` — the first→latest *trend* was the
     captured-but-unconsumed signal (the km/h analog of iteration 49's
     consistency deltas). Added `speedImprovement` (latest − first km/h over the
     sessions that scaled pace, positive = hitting harder), a "Shot speed: up N
     km/h" `report()` line, and a "Speed up N km/h" segment on the
     Training-progress card's trend line.
   - **[done — iteration 79]** On-table accuracy progression. Iteration 77 added
     `TrainingSummary.onTableRate` (fraction of strokes kept on the table) and
     persisted it to the training export as `session.onTableRate`, but
     `SessionTrends`/`TrainingTrendPoint` never mined it — the captured-but-
     unconsumed signal (the accuracy analog of iteration 54's `speedImprovement`).
     Added `onTableRate` parsing to `TrainingTrendPoint` plus `accuracyImprovement`
     (latest − first over the sessions that tracked it, positive = missing the
     table less) and `bestOnTableRate` (personal-best consistency), surfaced as an
     "On-table accuracy: up N%" `report()` line and an "Accuracy up N%" segment on
     the Training-progress card's trend line.
   - **[done — iteration 80]** On-target streak progression. Iteration 78 added
     `TrainingSummary.longestOnTargetStreak` (best run of consecutive good-or-
     better shots) and persisted it to the training export as
     `session.longestOnTargetStreak`, but `SessionTrends`/`TrainingTrendPoint`
     never parsed it — the streak analog of iteration 79's on-table-accuracy gap.
     Added `longestOnTargetStreak` parsing to `TrainingTrendPoint` plus
     `bestOnTargetStreak` (personal-best "in a row" across every session that
     tracked it), surfaced as a "Best on-target streak: N in a row" `report()`
     line and a "· streak N" segment on the Training-progress card's Best line.
   - **[done — iteration 82]** Match career play-time total. `MatchTrendPoint`
     had parsed each match's `summary.durationMs` since iteration 56 but no
     `SessionTrends` getter or report line ever referenced it — a
     captured-but-unconsumed match-side signal. Added
     `SessionTrends.totalMatchDurationMs` (sum of `durationMs` over every saved
     match that recorded one, null if none — the match-side twin of iteration
     81's `totalShotsPracticed`), surfaced as a "Total play time: Mm SSs"
     `report()` line and a "· N m played" chip on the history screen's
     Match-record card.
   - **[done — iteration 81]** Training career practice-volume total. Iteration 56
     added `totalMatchPoints` as the match-side cumulative "career" volume stat,
     but training trends had no equivalent — the per-session `shotCount` was
     parsed into `TrainingTrendPoint` yet only ever used to label individual
     sessions, never summed across the history. Added
     `SessionTrends.totalShotsPracticed` (sum of `shotCount` over every saved
     drill, the training twin of `totalMatchPoints`), surfaced as a "Total shots
     practiced: N" `report()` line and a "· N shots" segment on the
     Training-progress card's Best line.
   - **[done — iteration 55]** Recurring coaching focus. Iteration 53 persisted
     each training session's coaching `focus` (its weakest dimension) into the
     structured export, but `SessionTrends` only ever mined numeric metrics —
     the persisted focus was captured-but-unconsumed. Added `focusArea` parsing
     to `TrainingTrendPoint` plus `focusCounts`, `recurringFocus` (most common
     focus, ties broken toward the more recent session), `recurringFocusCount`,
     and a `hasRecurringFocus` (>=2 drills) guard, surfaced as a "Recurring
     focus: X (N of M drills)" `report()` line and a "Keep working on x" cue on
     the Training-progress card — turning per-session coaching into a
     cross-session *persistent weak point* callout.
   - **[done — iteration 56]** Match career totals. Every prior `SessionTrends`
     addition mined *training* history; saved matches were only ever counted
     (`matchCount++`) because a match pits Player A vs B (no single tracked user
     to trend). But the persisted match reports still carry cumulative "career"
     data. Added a `MatchTrendPoint` parser (total points, longest rally, peak
     ball km/h, winner — all nullable so a sparse/old report still counts) plus
     `totalMatchPoints`, `fastestMatchBallSpeedKmh`, `longestMatchRallyStrokes`,
     and a `hasMatchData` guard, surfaced as a "Matches: N played …" `report()`
     section and a "Match record" card on the history screen — the match-side
     analog of the training-progress card, turning the previously count-only
     match history into cumulative bests.
   - **[done — iteration 60]** Match head-to-head win record. Iteration 56's
     `MatchTrendPoint` parsed the match `winner` (`A`/`B`/null) but nothing ever
     read it — the career section surfaced points/rally/speed bests only. Added
     `matchWinsBy('A'|'B')`, `decidedMatchCount`, and a `hasMatchWinRecord`
     guard that tally the finished matches into an A-vs-B seat-vs-seat record,
     surfaced as a "Head-to-head: A N–M B" `report()` line and a line on the
     "Match record" history card. For a recurring two-player pairing this is the
     running series score they'd otherwise keep by hand; unfinished / older
     (pre-winner-field) matches simply don't contribute.

7. Training mode: shot segmentation + quality grading.
   - **[done — iteration 51]** Coaching feedback (`core/training/
     training_feedback.dart`). Every prior training layer *measured* the drill
     (depth, placement/lateral consistency, tempo/rhythm, pace) and surfaced each
     number on its own, but nothing turned that wall of percentages into "what to
     work on next" — the prioritization a coach provides. `TrackingQualityAnalyzer`
     only advises on phone placement, not stroke technique.
     `TrainingFeedback(summary, config:)` scores the coachable dimensions —
     placement accuracy (average depth vs `targetDepth`, within `depthTolerance`),
     depth consistency, lateral consistency, and rhythm (the last three only with
     ≥2 shots) — into `[0,1]` `FeedbackDimension`s, then names the *weakest* as the
     focus (`focusTip`, with a directional placement cue when the player is short
     vs overshooting) and the *strongest* as a confirmed strength. When even the
     weakest dimension clears the `_goodEnough` bar it returns encouragement
     instead of a fix-it cue. Pure Dart, derived only from the `TrainingSummary`,
     unit-tested. Both `TrainingScreen` (demo) and `CameraTrainingScreen` (live)
     surface a "Focus next: …" line in the end-of-session report and append the
     full coaching section to the Copy-report clipboard export.
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
     - **[done — iteration 76]** Live training km/h readout. `ShotAnalyzer`
       already computed a per-frame horizontal ball speed to track each flight's
       peak, but only surfaced it at shot *completion* as `Shot.speedKmh` — so
       during a rally there was no live "radar gun" number like the iteration-75
       match overlay flashes. `ShotAnalyzer.currentSpeedKmh` now exposes the most
       recent per-frame reading scaled through the same `metersPerUnitX` ruler
       (null before two frames establish a velocity and cleared on `BallLost` /
       `reset`), and `CameraTrainingScreen`'s `_TargetOverlay` renders it as a
       "N km/h" label beside the tracked ball — drawn only while the ball is in
       view so a detector dropout doesn't freeze a stale number, reaching live
       speed-readout parity with the match screen.
   - **[done — iteration 77]** Training on-table accuracy (miss rate)
     (`core/training/shot_analyzer.dart`). A `Shot` is recorded only when the
     ball lands on the *target* half (a target-side `BounceEvent`); an outgoing
     stroke that crossed the net but then went off the table — lost in flight
     with no target bounce — was silently discarded (`_outgoing` cleared on
     `BallLostEvent`), so a player who kept missing the table got no shots *and*
     no penalty, and the headline "in %" every training app shows was
     unmeasurable. `ShotAnalyzer` now counts those as `missCount`, feeding
     `TrainingSummary.missedShots`; the summary exposes `attemptedShots`
     (`shotCount + missedShots`) and `onTableRate` (the `[0, 1]` fraction that
     landed in), `report()` gains an "On-table accuracy: N% (X of Y on the
     table)" line when anything was missed (so it flows into both training
     screens' report/Copy-report), and `buildTrainingReportJson`'s `session`
     block carries `missedShots`/`attemptedShots`/`onTableRate` so accuracy
     persists to history. Backward-compatible: `missedShots` defaults to `0`, so
     directly-built summaries and every existing test read 100% accuracy with no
     new report line.
   - **[done — iteration 78]** Training on-target streak
     (`core/training/shot_analyzer.dart`). The match summary has counted a
     player's `longestStreakFor` (consecutive points won) since iteration 5, but
     training mode graded each stroke in isolation and never tracked how many the
     player placed *well in a row* — the headline "N in a row" gamification/
     coaching stat. A stroke counts as on target when it earns at least
     `ShotGrade.good` (`score >= 0.6`), and `TrainingSummary` now exposes
     `longestOnTargetStreak` (the best run in the session, a single off-target
     stroke resets it — the training analog of `MatchSummary.longestStreakFor`)
     and `currentOnTargetStreak` (the trailing run still alive, for a live "in a
     row" readout). `report()` gains a "Best on-target streak: N in a row" line
     when the streak reaches ≥2 (so it flows into both training screens'
     report/Copy-report), and `buildTrainingReportJson`'s `session` block carries
     `longestOnTargetStreak` so the streak persists to history. Derived purely
     from the recorded shots, so it needs no new tracking state.
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
