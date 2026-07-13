import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';

/// Feed a sequence of events and return every decision the referee reached.
List<PointDecision> _run(RallyReferee ref, List<TrackerEvent> events) {
  final out = <PointDecision>[];
  for (final e in events) {
    final d = ref.update(e);
    if (d != null) out.add(d);
  }
  return out;
}

// Event builders keyed only by what the referee cares about.
BounceEvent _bounce(int t, TableSide side) =>
    BounceEvent(t, side == TableSide.left ? 0.25 : 0.75, 0.5, side);
NetCrossEvent _cross(int t, TableSide from) =>
    NetCrossEvent(t, from, from.other);
BallLostEvent _lost(int t) => BallLostEvent(t);

void main() {
  group('RallyReferee — side/player mapping', () {
    test('left player defaults to A, right to B', () {
      final ref = RallyReferee();
      expect(ref.playerOn(TableSide.left), Player.a);
      expect(ref.playerOn(TableSide.right), Player.b);
    });

    test('leftPlayer can be flipped', () {
      final ref = RallyReferee(leftPlayer: Player.b);
      expect(ref.playerOn(TableSide.left), Player.b);
      expect(ref.playerOn(TableSide.right), Player.a);
    });
  });

  group('RallyReferee — double bounce', () {
    test('two bounces on the same side awards the opponent', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.right),
        _bounce(50, TableSide.right), // right never returned it
      ]);
      expect(decisions, hasLength(1));
      expect(decisions.single.winner, Player.a); // left player wins
      expect(decisions.single.reason, PointReason.doubleBounce);
      expect(decisions.single.timestampMs, 50);
    });

    test('a bounce, a crossing, then a bounce back is a legal rally', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.left),
        _cross(30, TableSide.left),
        _bounce(60, TableSide.right),
        _cross(90, TableSide.right),
        _bounce(120, TableSide.left),
      ]);
      expect(decisions, isEmpty);
    });

    test('a crossing between two same-side bounces is not a double bounce', () {
      // e.g. ball bounces right, is returned back to right after a rally exchange.
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.right),
        _cross(30, TableSide.right),
        _bounce(60, TableSide.left),
        _cross(90, TableSide.left),
        _bounce(120, TableSide.right),
      ]);
      expect(decisions, isEmpty);
    });
  });

  group('RallyReferee — ball lost', () {
    test('bounce then loss with no return awards the opponent', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.left),
        _lost(200),
      ]);
      expect(decisions, hasLength(1));
      expect(decisions.single.winner, Player.b); // right wins, left didn't return
      expect(decisions.single.reason, PointReason.notReturned);
    });

    test('loss while in flight after a crossing is undetermined', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.left),
        _cross(30, TableSide.left), // ball now flying toward the right
        _lost(60), // smashed off the end / out of view
      ]);
      expect(decisions, hasLength(1));
      expect(decisions.single.winner, isNull);
      expect(decisions.single.reason, PointReason.outOfPlay);
      expect(decisions.single.isDecisive, isFalse);
    });

    test('loss before any bounce is undetermined', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [_lost(10)]);
      expect(decisions.single.reason, PointReason.outOfPlay);
      expect(decisions.single.winner, isNull);
    });
  });

  group('RallyReferee — end changes', () {
    test('switchEnds flips the side→player mapping', () {
      final ref = RallyReferee();
      expect(ref.leftPlayer, Player.a);
      expect(ref.playerOn(TableSide.left), Player.a);

      ref.switchEnds();
      expect(ref.leftPlayer, Player.b);
      expect(ref.playerOn(TableSide.left), Player.b);
      expect(ref.playerOn(TableSide.right), Player.a);

      ref.switchEnds(); // back to the start
      expect(ref.leftPlayer, Player.a);
    });

    test('a double bounce is attributed across an end change', () {
      final ref = RallyReferee();
      // Before the switch: left never returned it → the right player (B) wins.
      final before = _run(ref, [
        _bounce(0, TableSide.left),
        _bounce(50, TableSide.left),
      ]);
      expect(before.single.winner, Player.b);

      ref.switchEnds();
      // After the switch the same physical left half is now player B, so an
      // unreturned bounce there awards A instead.
      final after = _run(ref, [
        _bounce(100, TableSide.left),
        _bounce(150, TableSide.left),
      ]);
      expect(after.single.winner, Player.a);
    });

    test('reset does not undo an end change', () {
      final ref = RallyReferee();
      ref.switchEnds();
      ref.reset();
      expect(ref.leftPlayer, Player.b);
    });
  });

  group('RallyReferee — rally reset', () {
    test('referee resets after a decision so the next rally is independent', () {
      final ref = RallyReferee();
      // Rally 1: double bounce on the right.
      _run(ref, [_bounce(0, TableSide.right), _bounce(50, TableSide.right)]);
      // Rally 2: a legal exchange should NOT be flagged using stale state.
      final decisions = _run(ref, [
        _bounce(100, TableSide.right),
        _cross(130, TableSide.right),
        _bounce(160, TableSide.left),
      ]);
      expect(decisions, isEmpty);
    });
  });
}
