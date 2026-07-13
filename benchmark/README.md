# pong-ai benchmark harness

This directory holds the offline evaluation promised in
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) §2, in three stages:

- **Scoring accuracy** — replays labeled clips through the exact same
  `BallTracker → RallyReferee → ScoringEngine` pipeline the live camera drives
  (via `MatchController`) and compares the auto-detected outcome to ground truth.
- **Perception accuracy** — compares the pipeline's per-frame predicted
  detections to per-frame ground truth: **ball** precision/recall/F1 (loose IoU
  0.3) and **player pose** detection rate + PCK. This is the metric that answers
  the objective's priority — "how well does the model track the players/ball."
- **Event-detection accuracy** — replays a clip's frames through a real
  `BallTracker` and matches its emitted `BounceEvent`/`NetCrossEvent`s to
  ground-truth event timings (per-type precision/recall/F1 + mean timing error,
  greedy nearest-in-time within a temporal tolerance). Scoring measures the final
  number and perception measures the raw detections; this isolates whether the
  bounce/net-cross analysis that *awards* each point fires at the right instant.

It is pure Dart — no camera, no plugin — so it all runs in `flutter test`.

## Pieces

- `lib/core/benchmark/clip_fixture.dart` — `ClipFixture`, the JSON representation
  of one labeled clip (per-frame predicted detections + ground-truth score, plus
  optional per-frame ground-truth detections in `groundTruthFrames` and
  ground-truth event timings in `groundTruthEvents`).
- `lib/core/benchmark/benchmark_runner.dart` — `BenchmarkRunner` /
  `BenchmarkResult` / `BenchmarkSuiteResult`: scoring-accuracy metrics.
- `lib/core/benchmark/detection_metrics.dart` — `DetectionBenchmark` /
  `BallDetectionMetrics` / `PoseDetectionMetrics`: per-frame perception metrics.
- `lib/core/benchmark/event_metrics.dart` — `EventDetectionBenchmark` /
  `EventTypeMetrics` / `GroundTruthEvent`: tracker event-detection metrics.
- `lib/core/benchmark/benchmark_corpus.dart` — `loadClipDirectory` /
  `loadClipFixtures` / `buildCorpusReport`: load the on-disk corpus and compose
  the scoring + perception + event-detection stages into one report.
- `bin/benchmark.dart` — the runnable entrypoint (see **Running** below).
- `clips/` — the fixture corpus (start with `synthetic_demo.json`).
- `test/benchmark_test.dart`, `test/detection_metrics_test.dart`,
  `test/event_metrics_test.dart`, `test/benchmark_corpus_test.dart` —
  regression tests over the harness.

## Running

The harness runs both inside `flutter test` and as a standalone command:

```sh
dart run bin/benchmark.dart                 # score every clip in clips/
dart run bin/benchmark.dart path/a.json ... # score the given fixtures
```

It prints Stage 1 (scoring accuracy over all clips), Stage 2 (perception
accuracy for clips carrying `groundTruthFrames`), and Stage 3 (event-detection
accuracy for clips carrying `groundTruthEvents`), and exits non-zero when no
clips are found so it can gate CI.

## Fixture format (`clips/*.json`)

```jsonc
{
  "name": "openttgames_game1_p1",
  "source": "OpenTTGames game_1",
  "fps": 120,
  "netX": 0.5,            // normalized x of the net line for this camera placement
  "leftPlayer": "a",      // which player occupies the left half of the frame
  "firstServer": "a",
  "pointsPerGame": 11,
  "bestOf": 5,
  "groundTruth": {
    "pointsA": 11,
    "pointsB": 7,
    "pointWinners": ["a", "b", "a", ...]   // optional, enables ordered accuracy
  },
  "frames": [
    // one entry per frame; omit "ball" on frames where it wasn't detected.
    { "t": 0, "ball": { "label": "ball", "confidence": 0.9, "box": [l, t, w, h] } },
    { "t": 8 },
    { "t": 16, "ball": { "label": "ball", "confidence": 0.8, "box": [l, t, w, h] },
      "people": [ { "box": [l, t, w, h], "trackId": 1,
                    "keypoints": [[x, y, conf], ...] } ] }
  ]
}
```

- All coordinates are **normalized `[0,1]`**, `box` is `[left, top, width, height]`,
  and `y` increases downward (the vision-model convention).
- `t` is milliseconds; the tracker uses it for velocity, so keep it monotonic.
- `frames` are the pipeline's **predictions**. To also score perception, add a
  parallel `groundTruthFrames` array (same per-frame shape, index-aligned with
  `frames`) holding the *true* ball/people boxes and keypoints. Ground-truth
  keypoints with `conf == 0` are treated as unlabeled/occluded and skipped by
  PCK, the standard convention. Omit `groundTruthFrames` for scoring-only clips.
- To also score the tracker's **event-detection** timing (Stage 3), add a
  `groundTruthEvents` array of `{ "t": <ms>, "type": "bounce" | "netCross" }`
  entries — the true table-bounce / net-crossing instants. The
  `EventDetectionBenchmark` replays `frames` through a `BallTracker` and scores
  the emitted bounce/net-cross events against these. Omit for clips without
  event labels.

## Metrics

Per clip, scoring (`BenchmarkResult`):

- **Final score correct** — detected per-player point totals equal ground truth.
- **Point-total error** — L1 distance between detected and true `(A, B)`.
- **Ordered point accuracy** — fraction of rallies whose winner matched, in
  order (only when `pointWinners` is present).
- **Undetermined count** — rallies the referee couldn't attribute (in-flight
  ball loss); surfaced for a manual call rather than guessed.

Aggregated (`BenchmarkSuiteResult`): clips scored exactly, mean point recall,
and total undetermined rallies.

Per clip, perception (`DetectionBenchmarkResult`, when `groundTruthFrames` is
present):

- **Ball** — precision, recall, F1 at IoU ≥ 0.3, plus mean IoU and mean centre
  error over matched frames. A detection that overlaps the true ball below the
  IoU threshold is counted as both a false positive and a false negative.
- **Pose** — person detection rate (matched / ground-truth people, greedily
  matched by box IoU ≥ 0.5), PCK (fraction of visible keypoints within a
  normalized distance of 0.05), and mean keypoint error.

Per clip, event detection (`EventBenchmarkResult`, when `groundTruthEvents` is
present): per-type (bounce / net-cross) precision, recall, F1, and mean temporal
error of matched events (emitted events greedily matched to the nearest same-type
ground-truth event within a temporal tolerance).

## Adding real clips

**OpenTTGames converter (implemented).** OpenTTGames ships a per-game
`ball_markup.json` mapping a frame index to the ball centre in pixels
(`{"144": {"x": 921, "y": 507}, ...}`).
`lib/core/benchmark/openttgames_converter.dart` turns that map directly into a
fixture — `clipFixtureFromOpenTtGames(name:, ballMarkup:, frameWidth:,
frameHeight:, fps:, ...)` normalizes every ball centre to `[0,1]`, synthesizes a
small ball box, and emits the labeled positions as `groundTruthFrames`. Pass your
model's per-frame output as `predictedFrames` to score detection
precision/recall against that ground truth (or omit it for a perfect-detector
baseline). It is pure Dart, so it runs in `flutter test`.

OpenTTGames also ships a per-game `events_markup.json` (frame index →
`bounce`/`net`/`empty`). `openTtGamesBounceEvents(eventsMarkup:, fps:)` converts
its `bounce` frames into `GroundTruthEvent`s on the same ms clock, which
`EventDetectionBenchmark.evaluate(...)` scores against the `BounceEvent`s a
`BallTracker` emits over the clip's frames. (The `net` label — ball *hitting* the
net — is a different event from the tracker's over-the-net crossing, so it is not
mapped.) Pass that same map as `clipFixtureFromOpenTtGames(..., eventsMarkup:)`
and the converter attaches the events as the fixture's `groundTruthEvents`, so a
single converted clip feeds Stage 2 (perception) and Stage 3 (event detection) in
the corpus report as well as Stage 1 (scoring).

To convert any other public dataset or a side-angle match video into a fixture
by hand:

1. **Run detection** (the fine-tuned ball detector + pose model, or the dataset's
   own labels) per frame; normalize every box/keypoint to `[0,1]`.
2. **Emit one `frames[]` entry per frame** — leave `ball` out where the ball was
   not detected so the tracker's gap/loss handling is exercised realistically.
3. **Label the ground truth** from the clip's scoreboard: fill `pointsA/pointsB`
   (and `pointWinners` if you scored rally-by-rally).
4. **(Optional) For perception scoring**, add `groundTruthFrames` from the
   dataset's own per-frame annotations (OpenTTGames ships true ball positions;
   SPIN ships ball + pose) so `DetectionBenchmark` can score how well the model
   detector matched them — that is the direct model-quality comparison.
5. **(Optional) For event-detection scoring**, add `groundTruthEvents` (the true
   bounce / net-crossing instants) so `EventDetectionBenchmark` can score the
   tracker's event timing.
6. Drop the file in `clips/` and it is picked up by the harness.

Recommended sources (see ARCHITECTURE §2): **OpenTTGames** (ball position +
bounce/net/empty event labels → straightforward `frames` + ground truth),
**SPIN** (high-speed ball tracking), and side-angle **broadcast/amateur YouTube**
clips for real deployment geometry.
