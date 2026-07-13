import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_announcer.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';

MatchState _state({
  int pointsA = 0,
  int pointsB = 0,
  int gamesA = 0,
  int gamesB = 0,
  Player server = Player.a,
  bool over = false,
}) {
  return MatchState(
    pointsA: pointsA,
    pointsB: pointsB,
    gamesA: gamesA,
    gamesB: gamesB,
    server: server,
    initialServer: Player.a,
    pointsPerGame: 11,
    bestOf: 5,
    isMatchOver: over,
  );
}

void main() {
  group('MatchAnnouncer', () {
    test('first observation seeds baseline and announces nothing', () {
      final announcer = MatchAnnouncer();
      expect(announcer.onState(_state()), isNull);
    });

    test('calls a within-game point with the leader named first', () {
      final announcer = MatchAnnouncer()..onState(_state());
      expect(announcer.onState(_state(pointsA: 1)), 'Player A, 1–0.');
      expect(announcer.onState(_state(pointsA: 1, pointsB: 1)), '1 all.');
      expect(
        announcer.onState(_state(pointsA: 1, pointsB: 2)),
        'Player B, 2–1.',
      );
    });

    test('announces a completed game with the games standing', () {
      final announcer = MatchAnnouncer()..onState(_state(pointsA: 10, pointsB: 8));
      // Game point converts: points reset, games 1-0.
      expect(
        announcer.onState(_state(gamesA: 1)),
        'Game to Player A. Player A leads 1 games to 0.',
      );
    });

    test('games-all standing uses singular/plural correctly', () {
      final announcer = MatchAnnouncer()..onState(_state(gamesA: 1, pointsB: 10));
      expect(
        announcer.onState(_state(gamesA: 1, gamesB: 1)),
        'Game to Player B. 1 game all.',
      );
    });

    test('announces match over, taking precedence over the game call', () {
      final announcer =
          MatchAnnouncer()..onState(_state(gamesA: 2, gamesB: 2, pointsA: 10));
      expect(
        announcer.onState(_state(gamesA: 3, gamesB: 2, over: true)),
        'Match to Player A, 3 games to 2.',
      );
    });

    test('announces nothing when the score steps backward (undo)', () {
      final announcer = MatchAnnouncer()..onState(_state(pointsA: 5, pointsB: 3));
      expect(announcer.onState(_state(pointsA: 4, pointsB: 3)), isNull);
      // ...and the baseline tracks the undone state, so the re-scored point
      // still announces.
      expect(announcer.onState(_state(pointsA: 5, pointsB: 3)), 'Player A, 5–3.');
    });

    test('announces nothing on an unchanged state', () {
      final announcer = MatchAnnouncer()..onState(_state(pointsA: 2));
      expect(announcer.onState(_state(pointsA: 2)), isNull);
    });

    test('reset re-seeds so a fresh match at 0–0 is not announced', () {
      final announcer =
          MatchAnnouncer()..onState(_state(gamesA: 3, gamesB: 1, over: true));
      announcer.reset();
      // Feeding the fresh 0-0 state seeds again (no spurious "game" call).
      expect(announcer.onState(_state()), isNull);
      expect(announcer.onState(_state(pointsB: 1)), 'Player B, 1–0.');
    });

    test('suffixes a game-point cue when a side is one point away', () {
      final announcer = MatchAnnouncer()..onState(_state(pointsA: 9, pointsB: 9));
      expect(
        announcer.onState(_state(pointsA: 10, pointsB: 9)),
        'Player A, 10–9. Game point Player A.',
      );
    });

    test('voices multiple consecutive chances (double/triple game point)', () {
      final announcer =
          MatchAnnouncer()..onState(_state(pointsB: 9, pointsA: 7));
      expect(
        announcer.onState(_state(pointsB: 10, pointsA: 7)),
        'Player B, 10–7. Triple game point Player B.',
      );
    });

    test('suffixes a match-point cue in the deciding-game climax', () {
      final announcer = MatchAnnouncer()
        ..onState(_state(gamesA: 2, gamesB: 2, pointsA: 9, pointsB: 9));
      expect(
        announcer.onState(_state(gamesA: 2, gamesB: 2, pointsA: 10, pointsB: 9)),
        'Player A, 10–9. Match point Player A.',
      );
    });

    test('adds a game-point cue at deuce advantage', () {
      final announcer =
          MatchAnnouncer()..onState(_state(pointsA: 10, pointsB: 10));
      expect(
        announcer.onState(_state(pointsA: 11, pointsB: 10)),
        'Player A, 11–10. Game point Player A.',
      );
    });

    test('no pressure cue when neither side is one point away', () {
      final announcer = MatchAnnouncer()..onState(_state(pointsA: 5, pointsB: 3));
      expect(
        announcer.onState(_state(pointsA: 6, pointsB: 3)),
        'Player A, 6–3.',
      );
    });

    test('point call names the new server when serve rotates', () {
      final announcer = MatchAnnouncer()
        ..onState(_state(pointsA: 1, pointsB: 1, server: Player.a));
      expect(
        announcer.onState(_state(pointsA: 2, pointsB: 1, server: Player.b)),
        'Player A, 2–1. Player B to serve.',
      );
    });

    test('no serve cue when the server is unchanged', () {
      final announcer = MatchAnnouncer()
        ..onState(_state(pointsA: 1, server: Player.a));
      expect(
        announcer.onState(_state(pointsA: 1, pointsB: 1, server: Player.a)),
        '1 all.',
      );
    });

    test('serve cue precedes the pressure cue, which stays the final word', () {
      final announcer = MatchAnnouncer()
        ..onState(_state(pointsA: 9, pointsB: 8, server: Player.b));
      expect(
        announcer.onState(_state(pointsA: 10, pointsB: 8, server: Player.a)),
        'Player A, 10–8. Player A to serve. Double game point Player A.',
      );
    });

    test('serve cue surfaces on a real ScoringEngine serve rotation', () {
      final engine = ScoringEngine();
      final announcer = MatchAnnouncer()..onState(engine.state);
      final calls = <String>[];
      for (var i = 0; i < 4; i++) {
        engine.awardPoint(i.isEven ? Player.a : Player.b);
        final call = announcer.onState(engine.state);
        if (call != null) calls.add(call);
      }
      // Serve switches every 2 points before deuce, so a "to serve" cue lands.
      expect(calls.any((c) => c.contains('to serve')), isTrue);
    });

    test('drives through a real ScoringEngine game to the game call', () {
      final engine = ScoringEngine();
      final announcer = MatchAnnouncer()..onState(engine.state);
      final calls = <String>[];
      for (var i = 0; i < 11; i++) {
        engine.awardPoint(Player.a);
        final call = announcer.onState(engine.state);
        if (call != null) calls.add(call);
      }
      // 10 point calls then a game call.
      expect(calls.first, 'Player A, 1–0.');
      expect(calls.last, contains('Game to Player A'));
    });
  });
}
