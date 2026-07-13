import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_insights.dart';
import 'package:pong_ai/core/analysis/match_summary.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';

MatchState _state({int pointsA = 0, int pointsB = 0, int gamesA = 0}) {
  return MatchState(
    pointsA: pointsA,
    pointsB: pointsB,
    gamesA: gamesA,
    gamesB: 0,
    server: Player.a,
    initialServer: Player.a,
    pointsPerGame: 11,
    bestOf: 5,
    isMatchOver: false,
  );
}

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
  group('MatchInsights', () {
    test('empty match has no data and reports it', () {
      final insights = MatchInsights(
        MatchSummary(points: const [], finalState: _state()),
      );
      expect(insights.hasData, isFalse);
      expect(insights.insightsFor(Player.a).hasData, isFalse);
      expect(insights.insightsFor(Player.a).focusTip, isNull);
      expect(insights.report(), contains('not enough data'));
    });

    test('picks the weakest dimension as the focus cue', () {
      // A holds 1 of 4 serves (0.25) but breaks 3 of 4 receives (0.75).
      final summary = MatchSummary(
        points: [
          _ptS(Player.a, Player.a, 0),
          _ptS(Player.b, Player.a, 1),
          _ptS(Player.b, Player.a, 2),
          _ptS(Player.b, Player.a, 3),
          _ptS(Player.a, Player.b, 4),
          _ptS(Player.a, Player.b, 5),
          _ptS(Player.a, Player.b, 6),
          _ptS(Player.b, Player.b, 7),
        ],
        finalState: _state(pointsA: 4, pointsB: 4),
      );
      final a = MatchInsights(summary).insightsFor(Player.a);

      expect(a.dimensions.length, 2);
      expect(a.weakest!.name, 'Serve effectiveness');
      expect(a.weakest!.score, closeTo(0.25, 1e-9));
      expect(a.strongest!.name, 'Return of serve');
      expect(a.focusTip, contains('Sharpen your serve'));
    });

    test('encourages when even the weakest dimension is solid', () {
      // A holds 3 of 4 serves (0.75) and breaks 3 of 4 receives (0.75).
      final summary = MatchSummary(
        points: [
          _ptS(Player.a, Player.a, 0),
          _ptS(Player.a, Player.a, 1),
          _ptS(Player.a, Player.a, 2),
          _ptS(Player.b, Player.a, 3),
          _ptS(Player.a, Player.b, 4),
          _ptS(Player.a, Player.b, 5),
          _ptS(Player.a, Player.b, 6),
          _ptS(Player.b, Player.b, 7),
        ],
        finalState: _state(pointsA: 6, pointsB: 2),
      );
      final a = MatchInsights(summary).insightsFor(Player.a);

      expect(a.weakest!.score, closeTo(0.75, 1e-9));
      expect(a.focusTip, contains('Well-rounded'));
    });

    test('surfaces game-point conversion as a coachable dimension', () {
      // A reaches 2 game points, converts 1 (rate 0.5) in a game to 12-10.
      final points = <ScoredPoint>[
        for (var i = 0; i < 9; i++) ...[
          _ptG(Player.a, 0, i * 2),
          _ptG(Player.b, 0, i * 2 + 1),
        ],
        _ptG(Player.a, 0, 100), // 10-9
        _ptG(Player.b, 0, 101), // A's game point saved -> 10-10
        _ptG(Player.a, 0, 102), // 11-10 deuce
        _ptG(Player.a, 0, 103), // A's game point converted -> 12-10
      ];
      final insights = MatchInsights(
        MatchSummary(points: points, finalState: _state(gamesA: 1)),
      );
      final a = insights.insightsFor(Player.a);

      expect(a.dimensions.map((d) => d.name), contains('Closing games'));
      final clutch = a.dimensions.firstWhere((d) => d.name == 'Closing games');
      expect(clutch.score, closeTo(0.5, 1e-9));
      expect(a.focusTip, contains('Close out games'));

      // B held no game point and served nothing, so it has no data.
      expect(insights.insightsFor(Player.b).hasData, isFalse);
      final report = insights.report();
      expect(report, contains('Coaching insights'));
      expect(report, contains('Closing games: 50%'));
      expect(report, contains('Player B — not enough data'));
    });
  });
}
