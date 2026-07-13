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

  group('ScoringEngine — setFirstServer', () {
    test('sets the first server before play and drives the rotation', () {
      final engine = ScoringEngine();
      expect(engine.state.server, Player.a); // default

      expect(engine.setFirstServer(Player.b), isTrue);
      expect(engine.state.server, Player.b);
      expect(engine.state.initialServer, Player.b);

      // Serve still switches every 2 points, now starting from B.
      engine.awardPoint(Player.a);
      expect(engine.state.server, Player.b);
      engine.awardPoint(Player.a);
      expect(engine.state.server, Player.a);
    });

    test('is rejected once a point has been scored', () {
      final engine = ScoringEngine();
      engine.awardPoint(Player.a);
      expect(engine.setFirstServer(Player.b), isFalse);
      expect(engine.state.initialServer, Player.a);
    });
  });

  group('ScoringEngine — setMatchFormat', () {
    test('changes the format before play, preserving the server', () {
      final engine = ScoringEngine();
      expect(engine.state.bestOf, 5); // default
      expect(engine.state.pointsPerGame, 11);

      engine.setFirstServer(Player.b);
      expect(engine.setMatchFormat(bestOf: 3), isTrue);
      expect(engine.state.bestOf, 3);
      expect(engine.state.pointsPerGame, 11); // untouched
      expect(engine.state.server, Player.b); // preserved
      expect(engine.state.initialServer, Player.b);

      // Best-of-3 is won at 2 games.
      for (var i = 0; i < 11; i++) {
        engine.awardPoint(Player.a);
      }
      expect(engine.state.gamesA, 1);
      for (var i = 0; i < 11; i++) {
        engine.awardPoint(Player.a);
      }
      expect(engine.state.isMatchOver, isTrue);
    });

    test('can change both game length and best-of together', () {
      final engine = ScoringEngine();
      expect(engine.setMatchFormat(pointsPerGame: 21, bestOf: 7), isTrue);
      expect(engine.state.pointsPerGame, 21);
      expect(engine.state.bestOf, 7);
    });

    test('rejects an invalid (even) best-of and leaves state unchanged', () {
      final engine = ScoringEngine();
      expect(engine.setMatchFormat(bestOf: 4), isFalse);
      expect(engine.state.bestOf, 5);
    });

    test('is rejected once a point has been scored', () {
      final engine = ScoringEngine();
      engine.awardPoint(Player.a);
      expect(engine.setMatchFormat(bestOf: 3), isFalse);
      expect(engine.state.bestOf, 5);
    });
  });

  group('ScoringEngine — reset (rematch)', () {
    test('clears the score but keeps format and first server', () {
      final engine = ScoringEngine(
        firstServer: Player.b,
        pointsPerGame: 21,
        bestOf: 3,
      );
      engine.awardPoint(Player.a);
      engine.awardPoint(Player.b);
      engine.awardPoint(Player.b);

      engine.reset();

      expect(engine.state.pointsA, 0);
      expect(engine.state.pointsB, 0);
      expect(engine.state.gamesA, 0);
      expect(engine.state.gamesB, 0);
      expect(engine.state.isMatchOver, isFalse);
      // Format and first server survive the reset.
      expect(engine.state.pointsPerGame, 21);
      expect(engine.state.bestOf, 3);
      expect(engine.state.server, Player.b);
      expect(engine.state.initialServer, Player.b);
      // The previous match's points can't be undone back into the new one.
      expect(engine.undo(), isFalse);
    });

    test('lets the format be re-picked after a reset', () {
      final engine = ScoringEngine();
      engine.awardPoint(Player.a);
      expect(engine.setMatchFormat(bestOf: 3), isFalse); // locked mid-match

      engine.reset();
      expect(engine.setMatchFormat(bestOf: 3), isTrue); // unlocked again
      expect(engine.state.bestOf, 3);
    });
  });
}
