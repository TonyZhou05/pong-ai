/// Synthetic [FrameResult] sequences for driving the pipeline without a camera.
///
/// These are hand-built ball trajectories that deterministically exercise the
/// real [BallTracker] → [RallyReferee] → [ScoringEngine] path: each rally is a
/// downward-then-upward arc (a bounce) on one side of the table followed by the
/// ball disappearing (a missed return), which the referee scores as a
/// `notReturned` fault against that side. The side of the bounce therefore
/// picks the point winner, letting us script a match whose score climbs the way
/// a real one would.
///
/// Used by the in-app Match demo and by tests that verify the end-to-end score.
library;

import 'detection.dart';

/// Frame cadence used by the synthetic clips (~30 fps).
const int kSyntheticFrameStepMs = 33;

FrameResult _ballFrame(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(
        label: 'ball',
        confidence: 0.9,
        box: BBox(x - 0.01, y - 0.01, 0.02, 0.02),
      ),
    );

FrameResult _emptyFrame(int t) => FrameResult(timestampMs: t);

/// Builds a scripted rally sequence and appends it to [frames].
///
/// [bounceX] is the normalized x the ball bounces at; with the net at 0.5 an
/// x below 0.5 is the left side and above is the right. A vertical arc there
/// produces exactly one [BounceEvent], and the trailing empty frames produce a
/// [BallLostEvent] once the detection gap exceeds the tracker's tolerance —
/// together a `notReturned` point against whoever is on [bounceX]'s side.
int _appendNotReturnedRally(
  List<FrameResult> frames,
  int startMs,
  double bounceX,
) {
  var t = startMs;
  // A clean down-up arc: y rises to an apex then falls, per-frame steps well
  // above the tracker's bounce threshold so the direction flip is unambiguous.
  for (final y in const [0.45, 0.55, 0.65, 0.55, 0.45]) {
    frames.add(_ballFrame(t, bounceX, y));
    t += kSyntheticFrameStepMs;
  }
  // Ball lost: enough consecutive empty frames to exceed BallTracker.maxGapFrames.
  for (var i = 0; i < 8; i++) {
    frames.add(_emptyFrame(t));
    t += kSyntheticFrameStepMs;
  }
  return t;
}

/// A scripted demo match: 7 rallies whose bounce sides award, in order,
/// A, B, A, A, B, A, A — ending 5–2 to player A with no undetermined points.
///
/// (Net at x=0.5: a right-side bounce means the right player failed to return,
/// so the left player — A by [RallyReferee]'s default — wins, and vice-versa.)
List<FrameResult> demoMatchFrames() {
  const left = 0.30; // bounce here -> B scores
  const right = 0.70; // bounce here -> A scores
  const rallyBounceSides = <double>[
    right,
    left,
    right,
    right,
    left,
    right,
    right,
  ];

  final frames = <FrameResult>[];
  var t = 0;
  for (final side in rallyBounceSides) {
    t = _appendNotReturnedRally(frames, t, side);
  }
  return frames;
}

/// Appends one scripted **training stroke** to [frames].
///
/// The player stands on the left half and drives the ball across the net to the
/// right (target) half, where it bounces at [bounceX]. This produces exactly the
/// events the [ShotAnalyzer] segments a stroke from: a left→right
/// [NetCrossEvent] that arms the outgoing flight, then a right-side
/// [BounceEvent] at the apex of the arc that completes and grades the shot. The
/// trailing empty frames drop the ball (a [BallLostEvent]) so the next stroke
/// starts from a clean trajectory.
int _appendTrainingStroke(
  List<FrameResult> frames,
  int startMs,
  double bounceX,
) {
  var t = startMs;
  // x crosses the net between the first two frames (0.48 -> 0.52), then jumps to
  // the target-side landing spot; y traces a down-up arc whose apex (0.65) is
  // the reported bounce. See BallTracker for the apex-detection convention.
  const xs = <double>[0.48, 0.52];
  const ys = <double>[0.45, 0.48, 0.55, 0.65, 0.55, 0.45];
  for (var i = 0; i < ys.length; i++) {
    final x = i < xs.length ? xs[i] : bounceX;
    frames.add(_ballFrame(t, x, ys[i]));
    t += kSyntheticFrameStepMs;
  }
  // Ball lost between strokes so the analyzer's outgoing flight resets.
  for (var i = 0; i < 8; i++) {
    frames.add(_emptyFrame(t));
    t += kSyntheticFrameStepMs;
  }
  return t;
}

/// A scripted training session: six drives landing at varying depths on the
/// far (right) half, exercising the real [ShotAnalyzer] path to a deterministic
/// mix of excellent/good/fair shots (pace saturates, so placement depth grades
/// them — see the iteration-6 notes).
///
/// Depth of a landing at x is `2·(x − 0.5)` with the net at 0.5; against the
/// default [TrainingConfig] (target depth 0.75) these land, in order:
/// excellent, excellent, good, good, fair, excellent.
List<FrameResult> trainingSessionFrames() {
  const bounceXs = <double>[
    0.875, // depth 0.75 — on target
    0.850, // depth 0.70
    0.775, // depth 0.55 — short
    0.800, // depth 0.60
    0.710, // depth 0.42 — well short
    0.900, // depth 0.80 — deep
  ];

  final frames = <FrameResult>[];
  var t = 0;
  for (final x in bounceXs) {
    t = _appendTrainingStroke(frames, t, x);
  }
  return frames;
}
