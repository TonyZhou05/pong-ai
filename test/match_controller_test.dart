import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/analysis/table_calibrator.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';

FrameResult _frame(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, y, 0, 0)),
    );
FrameResult _empty(int t) => FrameResult(timestampMs: t);

/// A double bounce on side `x` whose apex sits at [highY] (the lowest on-screen
/// point of the arc), so the calibrated table surface can gate it in or out.
List<FrameResult> _doubleBounceAt(
  double x,
  double lowY,
  double highY, {
  int startT = 0,
}) {
  final mid = (lowY + highY) / 2;
  return [
    _frame(startT + 0, x, lowY),
    _frame(startT + 33, x, mid),
    _frame(startT + 66, x, highY),
    _frame(startT + 99, x, mid), // bounce 1 reported (apex was highY)
    _frame(startT + 132, x, highY),
    _frame(startT + 165, x, mid), // bounce 2 -> double bounce
  ];
}

/// A warm-up sweep of the ball across the table with both players visible, so
/// [TableCalibrator] can infer a surface band (x≈0.23–0.73, y≈0.42–0.54) and a
/// net line at the players' midpoint (≈0.5).
List<FrameResult> _warmup({int count = 8, int startT = 0}) => [
      for (var i = 0; i < count; i++)
        FrameResult(
          timestampMs: startT + i * 33,
          ball: Detection(
            label: 'ball',
            confidence: 0.9,
            box: BBox(0.20 + 0.08 * i, i.isEven ? 0.42 : 0.54, 0, 0),
          ),
          people: const [
            PersonPose(box: BBox(0.22, 0.30, 0.05, 0.40), keypoints: []),
            PersonPose(box: BBox(0.73, 0.30, 0.05, 0.40), keypoints: []),
          ],
        ),
    ];

/// A ball bouncing twice on one side (`x`) with no net crossing: down-up-down-up
/// in y. The tracker reports each low point (y=0.70) as a bounce.
List<FrameResult> _doubleBounceOn(double x, {int startT = 0}) => [
      _frame(startT + 0, x, 0.30),
      _frame(startT + 33, x, 0.50),
      _frame(startT + 66, x, 0.70),
      _frame(startT + 99, x, 0.50), // bounce 1 reported here (apex was 0.70)
      _frame(startT + 132, x, 0.70),
      _frame(startT + 165, x, 0.50), // bounce 2 -> double bounce
    ];

void main() {
  group('MatchController — auto-scoring', () {
    test('a double bounce on the right awards the left player a point', () {
      final mc = MatchController();
      List<PointDecision> last = const [];
      for (final f in _doubleBounceOn(0.75)) {
        last = mc.onFrame(f);
      }
      expect(last.single.reason, PointReason.doubleBounce);
      expect(last.single.winner, Player.a); // left player
      expect(mc.score.pointsA, 1);
      expect(mc.score.pointsB, 0);
    });

    test('two rallies won by opposite sides yield 1-1', () {
      final mc = MatchController();
      // Rally 1: double bounce on the right -> A scores.
      for (final f in _doubleBounceOn(0.75, startT: 0)) {
        mc.onFrame(f);
      }
      // Rally 2: double bounce on the left -> B scores.
      for (final f in _doubleBounceOn(0.25, startT: 1000)) {
        mc.onFrame(f);
      }
      expect(mc.score.pointsA, 1);
      expect(mc.score.pointsB, 1);
      expect(mc.undetermined, isEmpty);
    });

    test('a ball lost in flight is collected as undetermined, not scored', () {
      final mc = MatchController();
      // Establish a moving ball (no bounce), then lose it past maxGapFrames.
      mc.onFrame(_frame(0, 0.40, 0.50));
      mc.onFrame(_frame(33, 0.45, 0.50));
      final decisions = <PointDecision>[];
      for (var i = 1; i <= 8; i++) {
        decisions.addAll(mc.onFrame(_empty(33 + i * 33)));
      }
      expect(decisions.single.reason, PointReason.outOfPlay);
      expect(mc.score.pointsA, 0);
      expect(mc.score.pointsB, 0);
      expect(mc.undetermined, hasLength(1));
    });

    test('without a calibrator the controller is never in calibration', () {
      final mc = MatchController();
      expect(mc.isCalibrating, isFalse);
      expect(mc.geometry.left, 0.0);
      expect(mc.geometry.right, 1.0);
    });
  });

  group('MatchController — auto-calibration', () {
    test('defers scoring during warm-up, then infers the table geometry', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );

      // First 7 warm-up frames: still calibrating, geometry is the full frame.
      final warm = _warmup();
      for (final f in warm.take(7)) {
        expect(mc.onFrame(f), isEmpty);
      }
      expect(mc.isCalibrating, isTrue);
      expect(mc.geometry.right, 1.0);

      // The 8th sample completes calibration.
      mc.onFrame(warm[7]);
      expect(mc.isCalibrating, isFalse);
      expect(mc.geometry.left, closeTo(0.23, 0.03));
      expect(mc.geometry.right, closeTo(0.73, 0.03));
      expect(mc.geometry.bottom, closeTo(0.54, 0.03));
      expect(mc.geometry.netX, closeTo(0.5, 0.03));

      // No points were scored during the warm-up phase.
      expect(mc.score.pointsA, 0);
      expect(mc.score.pointsB, 0);
    });

    test('after calibration, an on-surface double bounce scores normally', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );
      for (final f in _warmup()) {
        mc.onFrame(f);
      }
      // Apex y=0.52 sits inside the calibrated surface (bottom≈0.54); the
      // right-side double bounce awards the left player (A).
      List<PointDecision> last = const [];
      for (final f in _doubleBounceAt(0.65, 0.44, 0.52, startT: 1000)) {
        last = mc.onFrame(f);
      }
      expect(last.single.reason, PointReason.doubleBounce);
      expect(mc.score.pointsA, 1);
    });

    test('after calibration, an off-surface (floor) bounce is not scored', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );
      for (final f in _warmup()) {
        mc.onFrame(f);
      }
      // Apex y=0.90 is well below the calibrated table bottom (≈0.54): the
      // gated tracker drops it, so no point is awarded.
      final decisions = <PointDecision>[];
      for (final f in _doubleBounceAt(0.65, 0.80, 0.90, startT: 1000)) {
        decisions.addAll(mc.onFrame(f));
      }
      expect(decisions, isEmpty);
      expect(mc.score.pointsA, 0);
      expect(mc.score.pointsB, 0);
    });
  });

  group('MatchController — player movement analytics', () {
    /// A frame with a left-side and right-side player whose box bottom-centre
    /// (their inferred foot) sits at the given x's.
    FrameResult playersFrame(int t, double leftX, double rightX) => FrameResult(
          timestampMs: t,
          people: [
            PersonPose(box: BBox(leftX - 0.025, 0.4, 0.05, 0.4), keypoints: const []),
            PersonPose(box: BBox(rightX - 0.025, 0.4, 0.05, 0.4), keypoints: const []),
          ],
        );

    test('accumulates per-player footwork through the live pipeline', () {
      final mc = MatchController();
      mc.onFrame(playersFrame(0, 0.20, 0.80));
      mc.onFrame(playersFrame(33, 0.30, 0.75));
      mc.onFrame(playersFrame(66, 0.30, 0.70));

      final a = mc.movementFor(Player.a); // left side
      final b = mc.movementFor(Player.b); // right side
      expect(a.framesTracked, 3);
      expect(b.framesTracked, 3);
      // Player A's foot moved 0.20→0.30→0.30 = 0.10 total.
      expect(a.distanceTravelled, closeTo(0.10, 1e-9));
      // Player B's foot moved 0.80→0.75→0.70 = 0.10 total.
      expect(b.distanceTravelled, closeTo(0.10, 1e-9));
      expect(a.wasTracked, isTrue);
    });

    test('movement side assignment follows the calibrated net line', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );
      // The warm-up frames that only defer scoring record no movement; the one
      // that completes calibration is a live frame, so it counts (1 each).
      for (final f in _warmup()) {
        mc.onFrame(f);
      }
      expect(mc.movementFor(Player.a).framesTracked, 1);
      expect(mc.movementFor(Player.b).framesTracked, 1);

      // Post-calibration frames feed the (net≈0.5) analyzer: 0.25→left (A),
      // 0.75→right (B).
      mc.onFrame(playersFrame(2000, 0.25, 0.75));
      expect(mc.movementFor(Player.a).framesTracked, 2);
      expect(mc.movementFor(Player.b).framesTracked, 2);
      expect(mc.movementFor(Player.a).averageX, closeTo((0.245 + 0.25) / 2, 1e-9));
    });
  });
}
