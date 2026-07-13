import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';

void main() {
  group('ScoringEngine — basic points & game win', () {
    test('awards points and wins a game 11-0', () {
      final engine = ScoringEngine();
      for (var i = 0; i < 11; i++) {
        engine.awardPoint(Player.a);
      }
      expect(engine.state.gamesA, 1);
      expect(engine.state.gamesB, 0);
      // Points reset for the next game.
      expect(engine.state.pointsA, 0);
      expect(engine.state.pointsB, 0);
    });

    test('does not win at 11 without a 2-point lead (deuce)', () {
      final engine = ScoringEngine();
      // Reach 10-10.
      for (var i = 0; i < 10; i++) {
        engine.awardPoint(Player.a);
        engine.awardPoint(Player.b);
      }
      // 11-10 is NOT a game win.
      engine.awardPoint(Player.a);
      expect(engine.state.gamesA, 0);
      expect(engine.state.pointsA, 11);
      expect(engine.state.pointsB, 10);

      // 12-10 wins.
      engine.awardPoint(Player.a);
      expect(engine.state.gamesA, 1);
      expect(engine.state.pointsA, 0);
    });
  });

  group('ScoringEngine — serve rotation', () {
    test('serve switches every 2 points before deuce', () {
      final engine = ScoringEngine(firstServer: Player.a);
      expect(engine.state.server, Player.a); // 0-0
      engine.awardPoint(Player.a); // 1-0 -> still A serves
      expect(engine.state.server, Player.a);
      engine.awardPoint(Player.b); // 1-1 -> switch to B
      expect(engine.state.server, Player.b);
      engine.awardPoint(Player.a); // 2-1 -> B still
      expect(engine.state.server, Player.b);
      engine.awardPoint(Player.a); // 3-1 -> switch back to A
      expect(engine.state.server, Player.a);
    });

    test('serve switches every point at deuce', () {
      final engine = ScoringEngine(firstServer: Player.a);
      for (var i = 0; i < 10; i++) {
        engine.awardPoint(Player.a);
        engine.awardPoint(Player.b);
      }
      // 10-10, total 20 -> initial server (A).
      expect(engine.state.pointsA, 10);
      expect(engine.state.server, Player.a);
      engine.awardPoint(Player.a); // 11-10 -> switch to B
      expect(engine.state.server, Player.b);
      engine.awardPoint(Player.b); // 11-11 -> switch to A
      expect(engine.state.server, Player.a);
    });
  });

  group('ScoringEngine — match win & serve handover between games', () {
    test('best-of-5 ends after 3 games; over ignores further points', () {
      final engine = ScoringEngine(bestOf: 5);
      for (var g = 0; g < 3; g++) {
        for (var i = 0; i < 11; i++) {
          engine.awardPoint(Player.a);
        }
      }
      expect(engine.state.gamesA, 3);
      expect(engine.state.isMatchOver, isTrue);

      final before = engine.state;
      engine.awardPoint(Player.b); // no-op once over
      expect(engine.state.pointsB, before.pointsB);
      expect(engine.state.gamesB, before.gamesB);
    });

    test('initial server alternates each new game', () {
      final engine = ScoringEngine(firstServer: Player.a);
      for (var i = 0; i < 11; i++) {
        engine.awardPoint(Player.a); // win game 1
      }
      // Game 2 should start with B serving.
      expect(engine.state.initialServer, Player.b);
      expect(engine.state.server, Player.b);
    });
  });

  group('ScoringEngine — undo', () {
    test('undo reverts the last awarded point', () {
      final engine = ScoringEngine();
      engine.awardPoint(Player.a);
      engine.awardPoint(Player.b);
      expect(engine.state.pointsA, 1);
      expect(engine.state.pointsB, 1);

      expect(engine.undo(), isTrue);
      expect(engine.state.pointsA, 1);
      expect(engine.state.pointsB, 0);

      expect(engine.undo(), isTrue);
      expect(engine.state.pointsA, 0);
      expect(engine.undo(), isFalse); // nothing left
    });
  });
}
