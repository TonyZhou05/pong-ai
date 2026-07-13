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
