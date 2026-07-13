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

ScoredPoint _ptG(Player winner, int game, int t) => ScoredPoint(
      winner: winner,
      reason: PointReason.notReturned,
      timestampMs: t,
      gameIndex: game,
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

  group('MatchSummary per-game breakdown', () {
    test('reconstructs each game score from the point log', () {
      // Game 0: A wins 11-4. Game 1: B wins 11-9 (partial sample).
      final points = <ScoredPoint>[
        for (var i = 0; i < 11; i++) _ptG(Player.a, 0, i),
        for (var i = 0; i < 4; i++) _ptG(Player.b, 0, 100 + i),
        for (var i = 0; i < 9; i++) _ptG(Player.a, 1, 200 + i),
        for (var i = 0; i < 11; i++) _ptG(Player.b, 1, 300 + i),
      ];
      final summary = MatchSummary(
        points: points,
        finalState: _state(gamesA: 1, gamesB: 1),
      );

      expect(summary.hasGameData, isTrue);
      final games = summary.gameScores;
      expect(games, hasLength(2));
      expect(games[0].pointsA, 11);
      expect(games[0].pointsB, 4);
      expect(games[0].winnerAt(11), Player.a);
      expect(games[1].pointsA, 9);
      expect(games[1].pointsB, 11);
      expect(games[1].winnerAt(11), Player.b);
    });

    test('flags an in-progress game as having no winner yet', () {
      final summary = MatchSummary(
        points: [
          for (var i = 0; i < 7; i++) _ptG(Player.a, 0, i),
          for (var i = 0; i < 5; i++) _ptG(Player.b, 0, 100 + i),
        ],
        finalState: _state(pointsA: 7, pointsB: 5),
      );
      final games = summary.gameScores;
      expect(games, hasLength(1));
      expect(games.single.winnerAt(11), isNull);
      expect(summary.report(), contains('Games: 7–5.'));
    });

    test('no game data when gameIndex unrecorded', () {
      final summary = MatchSummary(
        points: [_pt(Player.a, PointReason.notReturned, 0)],
        finalState: _state(pointsA: 1),
      );
      expect(summary.hasGameData, isFalse);
      expect(summary.gameScores, isEmpty);
      expect(summary.report(), isNot(contains('Games:')));
    });
  });

  group('MatchSummary game-point analytics', () {
    // Game 0 played to A 12-10: alternate to 9-9, then A 10-9, B saves to
    // 10-10, A 11-10, A closes 12-10. A holds game point on the 10-9→11 and
    // 11-10→12 points (converting the last); B faces both and saves the first.
    final pressureGame = <ScoredPoint>[
      for (var i = 0; i < 9; i++) ...[
        _ptG(Player.a, 0, i * 2),
        _ptG(Player.b, 0, i * 2 + 1),
      ],
      _ptG(Player.a, 0, 100), // 10-9  (no game point yet)
      _ptG(Player.b, 0, 101), // A's game point saved -> 10-10
      _ptG(Player.a, 0, 102), // 11-10 (deuce, no game point)
      _ptG(Player.a, 0, 103), // A's game point converted -> 12-10
    ];

    test('tallies game points held / converted / faced / saved', () {
      final summary = MatchSummary(
        points: pressureGame,
        finalState: _state(gamesA: 1),
      );

      expect(summary.hasPressureData, isTrue);

      expect(summary.gamePointsHeldBy(Player.a), 2);
      expect(summary.gamePointsConvertedBy(Player.a), 1);
      expect(summary.gamePointConversionRateFor(Player.a), 0.5);
      expect(summary.gamePointsFacedBy(Player.a), 0);
      expect(summary.gamePointsSavedBy(Player.a), 0);

      expect(summary.gamePointsHeldBy(Player.b), 0);
      expect(summary.gamePointsConvertedBy(Player.b), 0);
      expect(summary.gamePointConversionRateFor(Player.b), isNull);
      expect(summary.gamePointsFacedBy(Player.b), 2);
      expect(summary.gamePointsSavedBy(Player.b), 1);

      // Defensive-clutch rate: A faced none (null), B saved 1 of 2 faced.
      expect(summary.gamePointSaveRateFor(Player.a), isNull);
      expect(summary.gamePointSaveRateFor(Player.b), 0.5);
    });

    test('reports converted/saved game-point lines for both players', () {
      final summary = MatchSummary(
        points: pressureGame,
        finalState: _state(gamesA: 1),
      );
      final report = summary.report();
      expect(report, contains('game points: converted 1/2, saved 0/0'));
      expect(report, contains('game points: converted 0/0, saved 1/2'));
    });

    test('no pressure data when no game point was reached', () {
      // A short in-progress game (5-2) never reaches a game point.
      final summary = MatchSummary(
        points: [
          for (var i = 0; i < 5; i++) _ptG(Player.a, 0, i),
          for (var i = 0; i < 2; i++) _ptG(Player.b, 0, 100 + i),
        ],
        finalState: _state(pointsA: 5, pointsB: 2),
      );
      expect(summary.hasPressureData, isFalse);
      expect(summary.report(), isNot(contains('game points:')));
    });

    test('no pressure data without game indices', () {
      final summary = MatchSummary(
        points: [_ptS(Player.a, Player.a, 0)],
        finalState: _state(pointsA: 1),
      );
      expect(summary.hasPressureData, isFalse);
    });
  });

  group('MatchSummary match-tension analytics', () {
    // A → B → A lead swing: A leads 1–0, B leads 1–2 then 2–3, A ties 3–3,
    // then A pulls ahead 5–3. Differentials: +1,0,-1,-2,-1,0,+1,+2.
    List<ScoredPoint> swingGame() {
      var t = 0;
      ScoredPoint next(Player w) => _pt(w, PointReason.notReturned, t += 1000);
      return [
        next(Player.a), // +1
        next(Player.b), // 0
        next(Player.b), // -1
        next(Player.b), // -2  (B's biggest lead)
        next(Player.a), // -1
        next(Player.a), // 0
        next(Player.a), // +1
        next(Player.a), // +2
      ];
    }

    test('counts lead changes through ties', () {
      final summary = MatchSummary(
        points: swingGame(),
        finalState: _state(pointsA: 5, pointsB: 3),
      );
      // A ahead (start) → B ahead → A ahead = two lead changes.
      expect(summary.leadChanges, 2);
    });

    test('biggest lead per player', () {
      final summary = MatchSummary(
        points: swingGame(),
        finalState: _state(pointsA: 5, pointsB: 3),
      );
      expect(summary.largestLeadBy(Player.a), 2);
      expect(summary.largestLeadBy(Player.b), 2);
    });

    test('largest deficit overcome tracks comebacks', () {
      final summary = MatchSummary(
        points: swingGame(),
        finalState: _state(pointsA: 5, pointsB: 3),
      );
      // A trailed by 2 (1–3) and came back to win: comeback of 2.
      expect(summary.largestDeficitOvercomeBy(Player.a), 2);
      // B trailed 0–1 and recovered to 1–1 before pulling ahead: comeback of 1.
      expect(summary.largestDeficitOvercomeBy(Player.b), 1);
    });

    test('wire-to-wire match has no lead changes and no comeback', () {
      final summary = MatchSummary(
        points: [
          _pt(Player.a, PointReason.notReturned, 0),
          _pt(Player.a, PointReason.notReturned, 1000),
          _pt(Player.a, PointReason.notReturned, 2000),
        ],
        finalState: _state(pointsA: 3),
      );
      expect(summary.leadChanges, 0);
      expect(summary.largestLeadBy(Player.a), 3);
      expect(summary.largestDeficitOvercomeBy(Player.a), 0);
      expect(summary.largestDeficitOvercomeBy(Player.b), 0);
    });

    test('report surfaces lead changes and comeback lines', () {
      final report = MatchSummary(
        points: swingGame(),
        finalState: _state(pointsA: 5, pointsB: 3, over: false),
      ).report();
      expect(report, contains('Lead changes: 2'));
      expect(report, contains('biggest lead: 2'));
      expect(report, contains('overcame a 2-point deficit'));
    });

    test('report marks a wire-to-wire match', () {
      final report = MatchSummary(
        points: [_pt(Player.a, PointReason.notReturned, 0)],
        finalState: _state(pointsA: 1),
      ).report();
      expect(report, contains('Lead changes: 0 (wire-to-wire)'));
    });

    test('empty match has zero tension stats', () {
      final summary = MatchSummary(points: const [], finalState: _state());
      expect(summary.leadChanges, 0);
      expect(summary.largestLeadBy(Player.a), 0);
      expect(summary.largestDeficitOvercomeBy(Player.a), 0);
    });

    test('decisiveRally marks the point of no return after a comeback', () {
      // swingGame: A trails to −2 then wins 5–3; the permanent lead is only
      // taken on the last point (differential goes 0 → +1 at rally 7, then +2).
      // Series: 0,+1,0,−1,−2,−1,0,+1,+2 → winner A is level-or-behind through
      // index 6 (rally 6, score 3–3), so rally 7 takes the lead for good.
      final summary = MatchSummary(
        points: swingGame(),
        finalState: _state(gamesA: 3, pointsA: 5, pointsB: 3, over: true),
      );
      expect(summary.matchWinner, Player.a);
      expect(summary.decisiveRally, 7);
    });

    test('decisiveRally is 1 for a wire-to-wire winner', () {
      final summary = MatchSummary(
        points: [
          _pt(Player.a, PointReason.notReturned, 0),
          _pt(Player.a, PointReason.notReturned, 1000),
          _pt(Player.a, PointReason.notReturned, 2000),
        ],
        finalState: _state(gamesA: 3, pointsA: 3, over: true),
      );
      expect(summary.decisiveRally, 1);
    });

    test('decisiveRally is null while the match is in progress', () {
      final summary = MatchSummary(
        points: swingGame(),
        finalState: _state(pointsA: 5, pointsB: 3),
      );
      expect(summary.matchWinner, isNull);
      expect(summary.decisiveRally, isNull);
    });

    test('decisiveRally is null when the winner trails on total points', () {
      // B wins the match (over) but A leads on cumulative points 2–1, so B
      // never held the point lead — there is no point of no return.
      final summary = MatchSummary(
        points: [
          _pt(Player.a, PointReason.notReturned, 0),
          _pt(Player.b, PointReason.notReturned, 1000),
          _pt(Player.a, PointReason.notReturned, 2000),
        ],
        finalState: _state(gamesB: 3, over: true),
      );
      expect(summary.matchWinner, Player.b);
      expect(summary.decisiveRally, isNull);
    });

    test('report surfaces the decisive rally line for a completed match', () {
      final report = MatchSummary(
        points: swingGame(),
        finalState: _state(gamesA: 3, pointsA: 5, pointsB: 3, over: true),
      ).report();
      expect(report, contains('Player A took the lead for good at rally 7'));
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

    test('captures the game index on each awarded point', () {
      final controller = drivenController();
      // The demo never completes a game, so every point is in game 0.
      expect(controller.points.every((p) => p.gameIndex == 0), isTrue);
      final games = controller.summary.gameScores;
      expect(games, hasLength(1));
      // Game 0's running score matches the demo's 5-2.
      expect(games.single.pointsA, 5);
      expect(games.single.pointsB, 2);
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
