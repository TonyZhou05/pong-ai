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

      // B held no game point and served nothing, but it faced two of A's game
      // points and saved one (at 10-9), so it earns a defensive-clutch
      // dimension scored 1/2.
      final b = insights.insightsFor(Player.b);
      expect(b.dimensions.map((d) => d.name), ['Saving game points']);
      expect(b.dimensions.single.score, closeTo(0.5, 1e-9));
      final report = insights.report();
      expect(report, contains('Coaching insights'));
      expect(report, contains('Closing games: 50%'));
      expect(report, contains('Saving game points: 50%'));
    });

    test('coaches poor game-point defense as the focus', () {
      // A reaches game point once; B faces it and fails to save it (0.0),
      // while B holds half its own serves (0.5) — so B's weakest dimension
      // and thus focus is saving game points.
      final points = <ScoredPoint>[
        for (var i = 0; i < 9; i++) ...[
          ScoredPoint(
            winner: Player.a,
            reason: PointReason.notReturned,
            timestampMs: i * 2,
            gameIndex: 0,
            server: Player.b,
          ),
          ScoredPoint(
            winner: Player.b,
            reason: PointReason.notReturned,
            timestampMs: i * 2 + 1,
            gameIndex: 0,
            server: Player.b,
          ),
        ],
        // 9-9. A takes 10-9 (game point for A on the next B point).
        _ptG(Player.a, 0, 100),
        // 10-9: A's game point, B loses it -> not saved. A closes 11-9.
        _ptG(Player.a, 0, 101),
      ];
      final b = MatchInsights(
        MatchSummary(points: points, finalState: _state(gamesA: 1)),
      ).insightsFor(Player.b);

      final save = b.dimensions.firstWhere(
        (d) => d.name == 'Saving game points',
      );
      expect(save.score, closeTo(0.0, 1e-9));
      expect(b.weakest!.name, 'Saving game points');
      expect(b.focusTip, contains('Dig in when down game point'));
    });

    test('overall score is the mean of the assessed dimensions and grades it',
        () {
      // A holds 1 of 4 serves (0.25) and breaks 3 of 4 receives (0.75) ->
      // mean 0.5 -> grade D (0.5 < the 0.55 C cutoff).
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

      expect(a.overallScore, closeTo(0.5, 1e-9));
      expect(a.grade, 'D');
    });

    test('no data yields a null overall score and a dash grade', () {
      final a = MatchInsights(
        MatchSummary(points: const [], finalState: _state()),
      ).insightsFor(Player.a);
      expect(a.overallScore, isNull);
      expect(a.grade, '–');
    });

    test('head-to-head names the decisive dimension and reports it', () {
      // A holds 3 of 4 serves (0.75) and breaks 3 of 4 receives (0.75);
      // B holds 1 of 4 serves (0.25) and breaks 1 of 4 receives (0.25).
      final summary = MatchSummary(
        points: [
          _ptS(Player.a, Player.a, 0),
          _ptS(Player.a, Player.a, 1),
          _ptS(Player.a, Player.a, 2),
          _ptS(Player.b, Player.a, 3),
          _ptS(Player.b, Player.b, 4),
          _ptS(Player.a, Player.b, 5),
          _ptS(Player.a, Player.b, 6),
          _ptS(Player.a, Player.b, 7),
        ],
        finalState: _state(pointsA: 6, pointsB: 2),
      );
      final insights = MatchInsights(summary);

      final comps = insights.comparisons;
      expect(comps.map((c) => c.name), ['Serve effectiveness', 'Return of serve']);
      final serve = comps.first;
      expect(serve.scoreA, closeTo(0.75, 1e-9));
      expect(serve.scoreB, closeTo(0.25, 1e-9));
      expect(serve.leader, Player.a);
      expect(serve.gap, closeTo(0.5, 1e-9));
      expect(serve.scoreFor(Player.b), closeTo(0.25, 1e-9));

      // Serve and return gaps are equal (coupled), so the decisive one is the
      // earlier dimension, serve effectiveness.
      final decisive = insights.decisiveDimension;
      expect(decisive!.name, 'Serve effectiveness');

      expect(
        insights.report(),
        contains(
          'Match difference: Player A won the serve effectiveness battle '
          '(75% vs 25%).',
        ),
      );
    });

    test('a symmetric match has no decisive dimension', () {
      // Both players hold 1 of 4 serves and break 3 of 4 receives -> every
      // shared dimension ties, so there is no separating difference.
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
      final insights = MatchInsights(summary);

      expect(insights.comparisons.every((c) => c.leader == null), isTrue);
      expect(insights.decisiveDimension, isNull);
      expect(insights.report(), isNot(contains('Match difference')));
    });

    test('no shared dimension yields an empty comparison list', () {
      // Only A serves, so A has a serve dimension and B has a return dimension —
      // no dimension is shared by both players.
      final summary = MatchSummary(
        points: [
          _ptS(Player.a, Player.a, 0),
          _ptS(Player.a, Player.a, 1),
          _ptS(Player.b, Player.a, 2),
          _ptS(Player.a, Player.a, 3),
        ],
        finalState: _state(pointsA: 3, pointsB: 1),
      );
      final insights = MatchInsights(summary);

      expect(
        insights.insightsFor(Player.a).dimensions.map((d) => d.name),
        ['Serve effectiveness'],
      );
      expect(
        insights.insightsFor(Player.b).dimensions.map((d) => d.name),
        ['Return of serve'],
      );
      expect(insights.comparisons, isEmpty);
      expect(insights.decisiveDimension, isNull);
    });

    test('report includes each player grade alongside the focus line', () {
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
      final report = MatchInsights(summary).report();
      expect(report, contains('Player A — Grade D, Focus:'));
    });
  });
}
