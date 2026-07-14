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

  group('RallyReferee — out of bounds (double crossing, no bounce)', () {
    test('two crossings with no bounce between, then loss: the first shot '
        'flew out — point to its receiver', () {
      final ref = RallyReferee();
      // B returns from the right; the ball crosses into the left, never
      // lands, and dead-ball drift carries it back across before the track
      // dies. B's shot missed the table.
      final decisions = _run(ref, [
        _bounce(0, TableSide.right), // B's incoming ball lands legally
        _cross(100, TableSide.right), // B's return crosses into the left...
        _cross(600, TableSide.left), // ...and drifts back without landing
        _lost(1500),
      ]);
      final d = decisions.single;
      expect(d.reason, PointReason.outOfBounds);
      expect(d.winner, Player.a, reason: 'the receiver on the left wins');
      expect(
        d.timestampMs,
        100,
        reason: 'stamped when the out shot crossed, not when the track died',
      );
    });

    test('a bounce between two crossings is a legal exchange — no award', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _cross(0, TableSide.right),
        _bounce(50, TableSide.left), // the return landed: play goes on
        _cross(100, TableSide.left),
        _bounce(150, TableSide.right),
      ]);
      expect(decisions, isEmpty);
    });

    test('a single crossing then loss stays undetermined', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.right),
        _cross(100, TableSide.right),
        _lost(1000),
      ]);
      expect(decisions.single.reason, PointReason.outOfPlay);
      expect(decisions.single.winner, isNull);
    });

    test('three crossings without a bounce still award at the first', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _cross(0, TableSide.left), // into the right: the shot that went out
        _cross(300, TableSide.right),
        _cross(600, TableSide.left),
        _lost(1500),
      ]);
      final d = decisions.single;
      expect(d.reason, PointReason.outOfBounds);
      expect(d.winner, Player.b, reason: 'first crossing headed to the right');
      expect(d.timestampMs, 0);
    });
  });

  group('RallyReferee — out of bounds (lost past the table edge)', () {
    test('crossed into a side, lost beyond that side\'s edge: out', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.left),
        _cross(100, TableSide.left), // heads into the right side...
        // ...and the track dies already past the right edge: flew long.
        const BallLostEvent(1000, lostOutside: TableSide.right),
      ]);
      final d = decisions.single;
      expect(d.reason, PointReason.outOfBounds);
      expect(d.winner, Player.b, reason: 'the right-side receiver wins');
    });

    test('lost beyond the edge it came FROM does not trigger the rule', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.left),
        _cross(100, TableSide.left),
        const BallLostEvent(1000, lostOutside: TableSide.left),
      ]);
      // Falls through to the in-flight-loss path: undetermined.
      expect(decisions.single.reason, PointReason.outOfPlay);
      expect(decisions.single.winner, isNull);
    });

    test('a plain in-flight loss (no exit info) stays undetermined', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.left),
        _cross(100, TableSide.left),
        _lost(1000),
      ]);
      expect(decisions.single.reason, PointReason.outOfPlay);
    });
  });

  group('RallyReferee — serve gate (requireServe)', () {
    test('between-point passes decide nothing until a serve initiates play',
        () {
      final ref = RallyReferee(requireServe: true);
      // A casual knock across that gets caught: cross, then the ball is lost.
      final decisions = _run(ref, [
        _cross(0, TableSide.left),
        _lost(500),
      ]);
      expect(decisions, isEmpty, reason: 'no rally was ever initiated');
      expect(ref.rallyInProgress, isFalse);
    });

    test('the serve signature (bounce then crossing from it) arms the rally',
        () {
      final ref = RallyReferee(requireServe: true);
      final decisions = _run(ref, [
        _bounce(0, TableSide.left), // server's own-side bounce
        _cross(50, TableSide.left), // ...crossing from it: serve!
        _bounce(100, TableSide.right),
        _bounce(400, TableSide.right), // double bounce -> decided
      ]);
      expect(ref.rallyInProgress, isFalse, reason: 're-armed after decision');
      expect(decisions.single.reason, PointReason.doubleBounce);
      expect(decisions.single.winner, Player.a);
    });

    test('a crossing that lands also arms the rally (missed serve bounce)',
        () {
      final ref = RallyReferee(requireServe: true);
      final decisions = _run(ref, [
        _cross(0, TableSide.left), // serve crossing (own bounce untracked)
        _bounce(50, TableSide.right), // ...lands: play is live
        _lost(1000), // right side never returned it
      ]);
      expect(decisions.single.reason, PointReason.notReturned);
      expect(decisions.single.winner, Player.a);
    });

    test('after a decision the gate re-arms: leftovers decide nothing', () {
      final ref = RallyReferee(requireServe: true);
      _run(ref, [
        _bounce(0, TableSide.left),
        _cross(50, TableSide.left),
        _bounce(100, TableSide.right),
        _bounce(400, TableSide.right), // rally 1 decided
      ]);
      // Post-point knock-around: bounce on one side, then lost.
      final decisions = _run(ref, [
        _bounce(600, TableSide.left),
        _lost(1500),
      ]);
      expect(decisions, isEmpty);
    });

    test('off by default: streams score from the first event', () {
      final ref = RallyReferee();
      expect(ref.rallyInProgress, isTrue);
      final decisions = _run(ref, [
        _bounce(0, TableSide.right),
        _bounce(50, TableSide.right),
      ]);
      expect(decisions.single.reason, PointReason.doubleBounce);
    });
  });

  group('RallyReferee — exit attribution by crossing origin', () {
    NetCrossEvent crossWith(
      int t,
      TableSide from, {
      bool? near,
      bool offFrame = false,
    }) =>
        NetCrossEvent(
          t,
          from,
          from.other,
          originNearPlayer: near,
          originOffFrame: offFrame,
        );

    test('an off-frame return that exits past its target is out: receiver '
        'wins', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.right),
        crossWith(100, TableSide.right, near: true), // shot into the left
        // The left player chased it off-frame and returned it; the return
        // then dies past the right baseline.
        crossWith(700, TableSide.left, near: true, offFrame: true),
        const BallLostEvent(1500, lostOutside: TableSide.right),
      ]);
      final d = decisions.single;
      expect(d.reason, PointReason.outOfBounds);
      expect(d.winner, Player.b, reason: 'receiver on the exit side');
    });

    test('a near-player recross conflicting with dead-drift evidence is '
        'undecidable: prompt', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.right),
        crossWith(100, TableSide.right, near: true),
        // Reversal happened at a visible player — could be a return, could
        // be a dead ball rebounding where they stand.
        crossWith(700, TableSide.left, near: true),
        const BallLostEvent(1500, lostOutside: TableSide.right),
      ]);
      expect(decisions.single.reason, PointReason.outOfPlay);
      expect(decisions.single.winner, isNull);
    });

    test('a dead-drift recross (reversal in open space) keeps the '
        'first-crossing attribution', () {
      final ref = RallyReferee();
      final decisions = _run(ref, [
        _bounce(0, TableSide.right),
        crossWith(100, TableSide.right, near: true), // the shot that went out
        crossWith(700, TableSide.left, near: false), // drift back
        const BallLostEvent(1500, lostOutside: TableSide.right),
      ]);
      final d = decisions.single;
      expect(d.reason, PointReason.outOfBounds);
      expect(
        d.winner,
        Player.a,
        reason: 'first unanswered crossing (into the left) was the fault',
      );
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
