import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/match_summary.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';

MatchState _state({
  int gamesA = 0,
  int gamesB = 0,
  int pointsA = 0,
  int pointsB = 0,
  bool over = false,
}) {
  return MatchState(
    pointsA: pointsA,
    pointsB: pointsB,
    gamesA: gamesA,
    gamesB: gamesB,
    server: Player.a,
    initialServer: Player.a,
    pointsPerGame: 11,
    bestOf: 5,
    isMatchOver: over,
  );
}

ScoredPoint _pt(Player winner, PointReason reason, int t) =>
    ScoredPoint(winner: winner, reason: reason, timestampMs: t);

ScoredPoint _ptS(Player winner, Player server, int t) => ScoredPoint(
      winner: winner,
      reason: PointReason.notReturned,
      timestampMs: t,
      server: server,
    );

void main() {
  group('MatchSummary', () {
    test('counts points won per player', () {
      final summary = MatchSummary(
        points: [
          _pt(Player.a, PointReason.doubleBounce, 0),
          _pt(Player.b, PointReason.notReturned, 1000),
          _pt(Player.a, PointReason.notReturned, 2000),
        ],
        finalState: _state(pointsA: 2, pointsB: 1),
      );

      expect(summary.totalPoints, 3);
      expect(summary.pointsWonBy(Player.a), 2);
      expect(summary.pointsWonBy(Player.b), 1);
    });

    test('splits forced-error and open-play points', () {
      final summary = MatchSummary(
        points: [
          _pt(Player.a, PointReason.doubleBounce, 0),
          _pt(Player.a, PointReason.notReturned, 1000),
          _pt(Player.a, PointReason.outOfPlay, 2000),
        ],
        finalState: _state(pointsA: 3),
      );

      expect(summary.forcedErrorsWonBy(Player.a), 2);
      expect(summary.openPlayPointsWonBy(Player.a), 1);
      final breakdown = summary.reasonBreakdownFor(Player.a);
      expect(breakdown[PointReason.doubleBounce], 1);
      expect(breakdown[PointReason.notReturned], 1);
      expect(breakdown[PointReason.outOfPlay], 1);
    });

    test('finds the longest consecutive run per player', () {
      final summary = MatchSummary(
        points: [
          _pt(Player.a, PointReason.notReturned, 0),
          _pt(Player.a, PointReason.notReturned, 1),
          _pt(Player.a, PointReason.notReturned, 2),
          _pt(Player.b, PointReason.notReturned, 3),
          _pt(Player.b, PointReason.notReturned, 4),
          _pt(Player.a, PointReason.notReturned, 5),
        ],
        finalState: _state(),
      );

      expect(summary.longestStreakFor(Player.a), 3);
      expect(summary.longestStreakFor(Player.b), 2);
    });

    test('duration spans first to last point', () {
      final summary = MatchSummary(
        points: [
          _pt(Player.a, PointReason.notReturned, 5000),
          _pt(Player.b, PointReason.notReturned, 95000),
        ],
        finalState: _state(),
      );
      expect(summary.durationMs, 90000);
      expect(summary.report(), contains('1:30'));
    });

    test('empty log has zero duration and no winner', () {
      final summary = MatchSummary(points: const [], finalState: _state());
      expect(summary.durationMs, 0);
      expect(summary.matchWinner, isNull);
    });

    test('match winner is the player with more games once over', () {
      final summary = MatchSummary(
        points: const [],
        finalState: _state(gamesA: 3, gamesB: 1, over: true),
      );
      expect(summary.matchWinner, Player.a);
      expect(summary.report(), contains('Player A wins'));
    });
  });

  group('MatchSummary serve analytics', () {
    test('splits serve-won, receive-won, and serve win rate', () {
      // A serves points 0 and 1; B serves points 2 and 3.
      final summary = MatchSummary(
        points: [
          _ptS(Player.a, Player.a, 0), // A holds serve
          _ptS(Player.b, Player.a, 1), // A serves, B breaks
          _ptS(Player.b, Player.b, 2), // B holds serve
          _ptS(Player.a, Player.b, 3), // B serves, A breaks
        ],
        finalState: _state(pointsA: 2, pointsB: 2),
      );

      expect(summary.servePointsPlayedBy(Player.a), 2);
      expect(summary.servePointsWonBy(Player.a), 1);
      expect(summary.receivePointsWonBy(Player.a), 1); // won on B's serve
      expect(summary.serveWinRateFor(Player.a), closeTo(0.5, 1e-9));

      expect(summary.servePointsPlayedBy(Player.b), 2);
      expect(summary.servePointsWonBy(Player.b), 1);
      expect(summary.receivePointsWonBy(Player.b), 1);
    });

    test('serve win rate is null and no serve data when server unrecorded', () {
      final summary = MatchSummary(
        points: [
          _pt(Player.a, PointReason.notReturned, 0),
          _pt(Player.b, PointReason.notReturned, 1),
        ],
        finalState: _state(pointsA: 1, pointsB: 1),
      );
      expect(summary.hasServeData, isFalse);
      expect(summary.serveWinRateFor(Player.a), isNull);
      expect(summary.servePointsPlayedBy(Player.a), 0);
      // No serve line in the report when the server was never recorded.
      expect(summary.report(), isNot(contains('serve points won')));
    });

    test('report includes a serve line when serve data is present', () {
      final summary = MatchSummary(
        points: [_ptS(Player.a, Player.a, 0), _ptS(Player.a, Player.a, 1)],
        finalState: _state(pointsA: 2),
      );
      expect(summary.report(), contains('serve points won: 2/2 (100%)'));
    });
  });

  group('MatchController point log', () {
    MatchController drivenController() {
      final controller = MatchController();
      for (final frame in demoMatchFrames()) {
        controller.onFrame(frame);
      }
      return controller;
    }

    test('records one point per scored rally, in order', () {
      final controller = drivenController();
      // The scripted demo yields a deterministic 5-2, all notReturned.
      expect(controller.points, hasLength(7));
      expect(
        controller.points.every((p) => p.reason == PointReason.notReturned),
        isTrue,
      );
      expect(controller.summary.pointsWonBy(Player.a), 5);
      expect(controller.summary.pointsWonBy(Player.b), 2);
    });

    test('undo pops the log in lock-step with the score', () {
      final controller = drivenController();
      final beforeLen = controller.points.length;
      final beforeTotal =
          controller.score.pointsA + controller.score.pointsB;

      expect(controller.undo(), isTrue);

      expect(controller.points.length, beforeLen - 1);
      expect(
        controller.score.pointsA + controller.score.pointsB,
        beforeTotal - 1,
      );
    });

    test('captures the serving player on each awarded point', () {
      final controller = drivenController();
      // Every awarded point records who served it (not the post-award server).
      expect(controller.points.every((p) => p.server != null), isTrue);
      final summary = controller.summary;
      expect(summary.hasServeData, isTrue);
      // Serve attributions partition the played points across both players.
      expect(
        summary.servePointsPlayedBy(Player.a) +
            summary.servePointsPlayedBy(Player.b),
        controller.points.length,
      );
      // First demo point is served by the default first server, Player A.
      expect(controller.points.first.server, Player.a);
    });

    test('resolveUndetermined is a no-op for a non-pending decision', () {
      final controller = MatchController();
      const stray = PointDecision(
        winner: null,
        reason: PointReason.outOfPlay,
        timestampMs: 4242,
      );
      controller.resolveUndetermined(stray, Player.b);
      expect(controller.points, isEmpty);
      expect(controller.score.pointsA, 0);
    });
  });
}
