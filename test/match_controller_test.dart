import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';

FrameResult _frame(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, y, 0, 0)),
    );
FrameResult _empty(int t) => FrameResult(timestampMs: t);

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
  });
}
