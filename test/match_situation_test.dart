import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/scoring/match_situation.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';

MatchState _state({
  int pointsA = 0,
  int pointsB = 0,
  int gamesA = 0,
  int gamesB = 0,
  int pointsPerGame = 11,
  int bestOf = 5,
  bool isMatchOver = false,
}) {
  return MatchState(
    pointsA: pointsA,
    pointsB: pointsB,
    gamesA: gamesA,
    gamesB: gamesB,
    server: Player.a,
    initialServer: Player.a,
    pointsPerGame: pointsPerGame,
    bestOf: bestOf,
    isMatchOver: isMatchOver,
  );
}

void main() {
  group('MatchSituation game point', () {
    test('mid-game is no pressure', () {
      final s = MatchSituation(_state(pointsA: 5, pointsB: 3));
      expect(s.isGamePoint, isFalse);
      expect(s.pressure, PointPressure.none);
      expect(s.candidate, isNull);
      expect(s.label, isNull);
      expect(s.pointCount, 0);
    });

    test('10-7 is triple game point for A', () {
      final s = MatchSituation(_state(pointsA: 10, pointsB: 7));
      expect(s.candidate, Player.a);
      expect(s.isGamePoint, isTrue);
      expect(s.isMatchPoint, isFalse);
      expect(s.pressure, PointPressure.gamePoint);
      expect(s.pointCount, 3);
      expect(s.label, 'Triple game point A');
    });

    test('10-9 is single game point for A', () {
      final s = MatchSituation(_state(pointsA: 10, pointsB: 9));
      expect(s.pointCount, 1);
      expect(s.label, 'Game point A');
    });

    test('9-10 is single game point for B', () {
      final s = MatchSituation(_state(pointsA: 9, pointsB: 10));
      expect(s.candidate, Player.b);
      expect(s.label, 'Game point B');
    });

    test('deuce (10-10) is not game point for either side', () {
      final s = MatchSituation(_state(pointsA: 10, pointsB: 10));
      expect(s.isGamePoint, isFalse);
      expect(s.label, isNull);
    });

    test('advantage (11-10) is game point for the leader', () {
      final s = MatchSituation(_state(pointsA: 11, pointsB: 10));
      expect(s.candidate, Player.a);
      expect(s.pointCount, 1);
      expect(s.label, 'Game point A');
    });

    test('double game point reads "Double game point"', () {
      final s = MatchSituation(_state(pointsA: 10, pointsB: 8));
      expect(s.pointCount, 2);
      expect(s.label, 'Double game point A');
    });

    test('quadruple+ falls back to numeric plural', () {
      final s = MatchSituation(_state(pointsA: 10, pointsB: 6));
      expect(s.pointCount, 4);
      expect(s.label, '4 game points A');
    });
  });

  group('MatchSituation match point', () {
    test('game point in the deciding game is match point', () {
      // best-of-5 → 3 games to win; A leads 2 games and is 10-7.
      final s = MatchSituation(_state(pointsA: 10, pointsB: 7, gamesA: 2));
      expect(s.isGamePoint, isTrue);
      expect(s.isMatchPoint, isTrue);
      expect(s.pressure, PointPressure.matchPoint);
      expect(s.label, 'Triple match point A');
    });

    test('game point that only wins a game (not the match) is not match point',
        () {
      // A at 10-7 but only 1 game won → winning this game makes it 2, not 3.
      final s = MatchSituation(_state(pointsA: 10, pointsB: 7, gamesA: 1));
      expect(s.isMatchPoint, isFalse);
      expect(s.pressure, PointPressure.gamePoint);
    });

    test('best-of-3 match point at 1 game up', () {
      // best-of-3 → 2 games to win.
      final s = MatchSituation(
        _state(pointsA: 10, pointsB: 9, gamesA: 1, bestOf: 3),
      );
      expect(s.isMatchPoint, isTrue);
      expect(s.label, 'Match point A');
    });
  });

  test('a finished match reports no pressure', () {
    final s = MatchSituation(
      _state(pointsA: 11, pointsB: 5, gamesA: 3, isMatchOver: true),
    );
    expect(s.isGamePoint, isFalse);
    expect(s.candidate, isNull);
    expect(s.label, isNull);
  });

  test('agrees with the ScoringEngine win rule', () {
    // Drive a real engine to game point and confirm the next award wins.
    final engine = ScoringEngine();
    for (var i = 0; i < 10; i++) {
      engine.awardPoint(Player.a);
    }
    expect(MatchSituation(engine.state).candidate, Player.a);
    expect(MatchSituation(engine.state).isGamePoint, isTrue);
    engine.awardPoint(Player.a);
    expect(engine.state.gamesA, 1); // game actually completed
  });
}
