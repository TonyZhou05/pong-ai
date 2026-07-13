import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/rally_analyzer.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';

FrameResult _ball(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, y, 0, 0)),
    );

/// One net crossing (direction is irrelevant to rally-length counting).
NetCrossEvent _cross(int t) => NetCrossEvent(t, TableSide.left, TableSide.right);

PointDecision _decision(int t, {Player? winner = Player.a}) => PointDecision(
      winner: winner,
      reason: winner == null ? PointReason.outOfPlay : PointReason.notReturned,
      timestampMs: t,
    );

/// Play one rally of [crossings] net crossings starting at [startT] (33ms apart)
/// and end it with a decision at [endT].
void _playRally(
  RallyAnalyzer a, {
  required int crossings,
  required int startT,
  required int endT,
  Player? winner = Player.a,
}) {
  for (var i = 0; i < crossings; i++) {
    a.observe(_cross(startT + i * 33));
  }
  a.endRally(_decision(endT, winner: winner));
}

void main() {
  group('RallyAnalyzer single rally', () {
    test('counts net crossings as the stroke count', () {
      final a = RallyAnalyzer();
      _playRally(a, crossings: 4, startT: 100, endT: 400);
      expect(a.rallies, hasLength(1));
      expect(a.rallies.single.strokeCount, 4);
    });

    test('duration spans the first observed event to the decision', () {
      final a = RallyAnalyzer();
      _playRally(a, crossings: 3, startT: 100, endT: 700);
      // first crossing at t=100, decision at t=700.
      expect(a.rallies.single.durationMs, 600);
    });

    test('bounces do not count as strokes but do bound the duration', () {
      final a = RallyAnalyzer();
      a.observe(const BounceEvent(50, 0.7, 0.5, TableSide.right));
      a.observe(_cross(100));
      a.endRally(_decision(300));
      final rally = a.rallies.single;
      expect(rally.strokeCount, 1); // only the crossing
      expect(rally.durationMs, 250); // 300 - 50 (first event was the bounce)
    });

    test('a decision with no prior events is a zero-duration, zero-stroke rally',
        () {
      final a = RallyAnalyzer();
      a.endRally(_decision(500));
      expect(a.rallies.single.strokeCount, 0);
      expect(a.rallies.single.durationMs, 0);
    });

    test('carries through the referee verdict', () {
      final a = RallyAnalyzer();
      _playRally(a, crossings: 2, startT: 0, endT: 100, winner: null);
      expect(a.rallies.single.winner, isNull);
      expect(a.rallies.single.reason, PointReason.outOfPlay);
    });
  });

  group('RallyStats aggregate', () {
    RallyAnalyzer analyzerWith(List<int> strokeCounts) {
      final a = RallyAnalyzer();
      var t = 0;
      for (final c in strokeCounts) {
        _playRally(a, crossings: c, startT: t, endT: t + 1000);
        t += 2000;
      }
      return a;
    }

    test('empty analyzer reports zeros', () {
      const stats = RallyStats([]);
      expect(stats.rallyCount, 0);
      expect(stats.totalStrokes, 0);
      expect(stats.averageStrokes, 0);
      expect(stats.longestStrokes, 0);
      expect(stats.averageDurationMs, 0);
      expect(stats.report(), contains('No rallies'));
    });

    test('averages and longest are computed over all rallies', () {
      final stats = analyzerWith([2, 4, 6]).stats;
      expect(stats.rallyCount, 3);
      expect(stats.totalStrokes, 12);
      expect(stats.averageStrokes, closeTo(4.0, 1e-9));
      expect(stats.longestStrokes, 6);
    });

    test('buckets rallies into short / medium / long by stroke count', () {
      // 1,2 -> short; 3,5 -> medium; 6,9 -> long
      final stats = analyzerWith([1, 2, 3, 5, 6, 9]).stats;
      expect(stats.shortRallies, 2);
      expect(stats.mediumRallies, 2);
      expect(stats.longRallies, 2);
    });

    test('average duration is the mean rally span in ms', () {
      // every scripted rally spans exactly 1000ms.
      final stats = analyzerWith([1, 3, 5]).stats;
      expect(stats.averageDurationMs, closeTo(1000, 1e-9));
    });

    test('report is deterministic and mentions the headline numbers', () {
      final report = analyzerWith([2, 4, 6]).stats.report();
      expect(report, contains('3 rallies'));
      expect(report, contains('avg 4.0 strokes'));
      expect(report, contains('longest 6'));
    });
  });

  group('reset', () {
    test('forgets all rallies', () {
      final a = RallyAnalyzer();
      _playRally(a, crossings: 3, startT: 0, endT: 100);
      a.reset();
      expect(a.rallies, isEmpty);
      expect(a.stats.rallyCount, 0);
    });
  });

  group('MatchController integration', () {
    test('accumulates one rally per scored point over the demo match', () {
      final controller = MatchController();
      for (final frame in demoMatchFrames()) {
        controller.onFrame(frame);
      }
      // demoMatchFrames resolves 7 points (see demo_match_test); each scripted
      // rally is a serve that bounces and is not returned — the ball never
      // crosses the net — so every rally is a zero-stroke short rally.
      final stats = controller.rallyStats;
      expect(stats.rallyCount, 7);
      expect(stats.totalStrokes, 0);
      expect(stats.shortRallies, 7);
    });

    test('counts a real net crossing as a stroke through the live tracker', () {
      final controller = MatchController();
      // Ball sweeps left -> right across the net at 0.5 (one crossing), bounces
      // on the right side, then is lost -> a notReturned point for Player.a.
      final frames = <FrameResult>[
        _ball(0, 0.30, 0.50),
        _ball(33, 0.45, 0.50),
        _ball(66, 0.60, 0.50), // crosses the net here
        _ball(99, 0.70, 0.45),
        _ball(132, 0.70, 0.60), // descending toward the table
        _ball(165, 0.70, 0.55), // ascending -> bounce at t=132 (right side)
        for (var i = 0; i < 8; i++) FrameResult(timestampMs: 198 + i * 33),
      ];
      for (final f in frames) {
        controller.onFrame(f);
      }
      final stats = controller.rallyStats;
      expect(stats.rallyCount, 1);
      expect(stats.rallies.single.strokeCount, 1);
      expect(stats.rallies.single.winner, Player.a);
    });
  });
}
