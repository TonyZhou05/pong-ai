# pong-ai benchmark harness

This directory holds the offline **scoring-accuracy** evaluation promised in
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) §2. It replays labeled clips
through the exact same `BallTracker → RallyReferee → ScoringEngine` pipeline the
live camera drives (via `MatchController`) and compares the auto-detected
outcome to ground truth. It is pure Dart — no camera, no plugin — so it runs in
`flutter test`.

## Pieces

- `lib/core/benchmark/clip_fixture.dart` — `ClipFixture`, the JSON representation
  of one labeled clip (per-frame detections + ground-truth score).
- `lib/core/benchmark/benchmark_runner.dart` — `BenchmarkRunner` /
  `BenchmarkResult` / `BenchmarkSuiteResult`: runs fixtures and emits metrics.
- `clips/` — the fixture corpus (start with `synthetic_demo.json`).
- `test/benchmark_test.dart` — regression tests over the harness.

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

## Metrics

Per clip (`BenchmarkResult`):

- **Final score correct** — detected per-player point totals equal ground truth.
- **Point-total error** — L1 distance between detected and true `(A, B)`.
- **Ordered point accuracy** — fraction of rallies whose winner matched, in
  order (only when `pointWinners` is present).
- **Undetermined count** — rallies the referee couldn't attribute (in-flight
  ball loss); surfaced for a manual call rather than guessed.

Aggregated (`BenchmarkSuiteResult`): clips scored exactly, mean point recall,
and total undetermined rallies.

## Adding real clips

The objective calls for benchmarking against real footage. To convert a public
dataset or a side-angle match video into a fixture:

1. **Run detection** (the fine-tuned ball detector + pose model, or the dataset's
   own labels) per frame; normalize every box/keypoint to `[0,1]`.
2. **Emit one `frames[]` entry per frame** — leave `ball` out where the ball was
   not detected so the tracker's gap/loss handling is exercised realistically.
3. **Label the ground truth** from the clip's scoreboard: fill `pointsA/pointsB`
   (and `pointWinners` if you scored rally-by-rally).
4. Drop the file in `clips/` and it is picked up by the harness.

Recommended sources (see ARCHITECTURE §2): **OpenTTGames** (ball position +
bounce/net/empty event labels → straightforward `frames` + ground truth),
**SPIN** (high-speed ball tracking), and side-angle **broadcast/amateur YouTube**
clips for real deployment geometry.
