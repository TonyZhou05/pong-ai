import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';

void main() {
  group('demoMatchFrames end-to-end pipeline', () {
    test('drives the real pipeline to a deterministic 5-2 score', () {
      final controller = MatchController();
      for (final frame in demoMatchFrames()) {
        controller.onFrame(frame);
      }

      final state = controller.score;
      expect(state.pointsA, 5);
      expect(state.pointsB, 2);
      expect(state.gamesA, 0);
      expect(state.gamesB, 0);
      expect(state.isMatchOver, isFalse);
    });

    test('every scripted rally is auto-attributed (nothing undetermined)', () {
      final controller = MatchController();
      for (final frame in demoMatchFrames()) {
        controller.onFrame(frame);
      }
      expect(controller.undetermined, isEmpty);
    });

    test('all seven rallies resolve as notReturned faults', () {
      final controller = MatchController();
      final reasons = <PointReason>[];
      for (final frame in demoMatchFrames()) {
        for (final d in controller.onFrame(frame)) {
          reasons.add(d.reason);
        }
      }
      expect(reasons, hasLength(7));
      expect(reasons.every((r) => r == PointReason.notReturned), isTrue);
    });
  });
}
