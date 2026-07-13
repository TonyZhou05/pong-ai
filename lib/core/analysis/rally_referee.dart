/// Pure-Dart refereeing: turns [TrackerEvent]s into point-award decisions.
///
/// This is the missing middle layer between [BallTracker] (which reports what
/// the ball *did* — bounces, net crossings, losses) and [ScoringEngine] (which
/// only understands "player X won the point"). The referee watches a rally
/// unfold and, when it can confidently attribute a fault, names the point
/// winner and why.
///
/// It has no Flutter or vision dependencies, so the officiating logic can be
/// unit-tested against synthetic rallies and replayed benchmark clips.
///
/// Only faults that are *reliably* visible from the ball trajectory are scored
/// automatically:
/// * a double bounce on one side (that player failed to hit the ball), and
/// * a ball that bounced on a side and then went out of play without being
///   returned over the net (that player failed to return it).
///
/// A ball lost while still in flight (e.g. smashed off the end of the table) is
/// genuinely ambiguous from the ball path alone, so it is surfaced as an
/// undetermined decision for the UI to resolve rather than guessed at.
library;

import '../scoring/scoring_engine.dart';
import 'ball_tracker.dart';

/// Why the referee ended the rally and (when known) who won the point.
enum PointReason {
  /// The ball bounced twice on the same side without a net crossing between —
  /// that side never returned it.
  doubleBounce,

  /// The ball bounced on a side then left play without crossing back over the
  /// net — that side failed to return it.
  notReturned,

  /// The ball was lost while in flight (after a crossing, or before any bounce).
  /// The path alone cannot say whose fault it was; [PointDecision.winner] is
  /// null and the UI should ask the user.
  outOfPlay,

  /// The ball crossed the net and play then ended without it *ever* touching
  /// the receiving side's table — it crossed back (dead-ball drift) or was
  /// lost. A legal shot must land before anything else happens, so the shot
  /// flew out of bounds: point to the receiving side's player.
  outOfBounds,

  /// The point was entered by the user, not inferred from the ball path — e.g.
  /// the vision pipeline missed a rally entirely and the user tapped a
  /// "+point" button to keep the score correct.
  manual,
}

/// The referee's verdict at the end of a rally.
class PointDecision {
  const PointDecision({
    required this.winner,
    required this.reason,
    required this.timestampMs,
  });

  /// The player awarded the point, or null when [reason] is
  /// [PointReason.outOfPlay] and the winner could not be inferred.
  final Player? winner;

  final PointReason reason;
  final int timestampMs;

  /// Whether the referee could name a winner (safe to auto-score).
  bool get isDecisive => winner != null;

  @override
  String toString() =>
      'PointDecision(${winner ?? 'undetermined'}, $reason, @$timestampMs)';
}

/// Watches one rally's [TrackerEvent]s and decides the point.
///
/// Feed events (as produced by [BallTracker.update]) one at a time to [update].
/// It returns a [PointDecision] on the event that ends the rally and then
/// resets itself for the next rally; otherwise it returns null.
class RallyReferee {
  RallyReferee({Player leftPlayer = Player.a, this.requireServe = false})
      : _initialLeftPlayer = leftPlayer;

  /// When true, events do not score (or even count as rally activity) until a
  /// rally has visibly been *initiated*: either the serve signature — a bounce
  /// on side S followed by a crossing from S — or, for tracks that miss the
  /// serve's own-side bounce, a crossing into a side followed by a bounce on
  /// that side (the ball demonstrably landed in play). Without the gate,
  /// players casually knocking the ball to each other between points read as
  /// rally activity: bounces inflate the rally counter and a caught pass can
  /// fizzle into a bogus decision. After every decision the gate re-arms, so
  /// between-point noise stays ignored until the next serve. Off by default
  /// (scripted clips and the historical behaviour treat every stream as
  /// in-rally from the first event).
  final bool requireServe;

  /// Whether the current rally has been initiated (see [requireServe]).
  bool _rallyStarted = false;

  /// Whether rally activity is currently live: always true without
  /// [requireServe]; otherwise true once the serve signature was seen and
  /// until the rally's decision. Consumers (e.g. the bounce counter) use this
  /// to ignore between-point ball motion.
  bool get rallyInProgress => !requireServe || _rallyStarted;

  /// Pre-rally observations while waiting for a serve ([requireServe] only).
  TableSide? _preBounceSide;
  TableSide? _preCrossTo;

  /// Which player occupied the left half of the table at the start of the match.
  /// The live [leftPlayer] flips away from this each time the players
  /// [switchEnds].
  final Player _initialLeftPlayer;

  /// Whether the players have swapped ends an odd number of times, so the
  /// physical left/right halves now map to the opposite scoring [Player]s.
  bool _endsSwapped = false;

  /// The player currently occupying the left half of the table (split by the
  /// net). The other player is on the right. This maps the tracker's spatial
  /// [TableSide] onto the scoring engine's [Player] identities, accounting for
  /// any end changes ([switchEnds]). Exposed so movement analytics can attribute
  /// detected people to the same [Player] identities.
  Player get leftPlayer =>
      _endsSwapped ? _initialLeftPlayer.other : _initialLeftPlayer;

  /// Swap which scoring [Player] each physical half of the table belongs to.
  ///
  /// In table tennis the players change ends between games, but the phone (and
  /// therefore the camera's left/right) stays put — so after an end change the
  /// person now on the left is the *other* [Player]. The [MatchController]
  /// calls this when a game completes so the side→player attribution keeps
  /// awarding points to the correct player across games. Rally state
  /// ([reset]) is unaffected; the swap persists for the rest of the match.
  void switchEnds() => _endsSwapped = !_endsSwapped;

  /// Restore the original side→player mapping (as if no [switchEnds] had
  /// happened). Called when starting a rematch so the new match's side
  /// attribution begins from the players' starting ends rather than wherever
  /// the previous match's end changes left it.
  void resetEnds() => _endsSwapped = false;

  /// The side of the most recent legal bounce this rally, or null before one.
  TableSide? _lastBounceSide;

  /// Whether the ball has crossed the net since [_lastBounceSide] was recorded.
  bool _crossedSinceBounce = false;

  /// Net crossings since the last bounce. Two or more crossings with no bounce
  /// between them are impossible in legal play — the *first* of them was a
  /// shot that never landed (what "crossed back" is dead-ball drift after the
  /// ball flew long). Counted so [_onBallLost] can award the out-of-bounds
  /// point instead of surfacing an undetermined prompt.
  int _crossesSinceBounce = 0;

  /// Destination side and timestamp of the first crossing since the last
  /// bounce — the shot that, if never answered by a bounce, went out.
  TableSide? _firstUnansweredCrossTo;
  int _firstUnansweredCrossT = 0;

  /// The player on a given side of the table.
  Player playerOn(TableSide side) =>
      side == TableSide.left ? leftPlayer : leftPlayer.other;

  /// Process one rally event; returns a verdict if it ends the rally.
  PointDecision? update(TrackerEvent event) {
    if (requireServe && !_rallyStarted) return _onPreRally(event);
    return switch (event) {
      NetCrossEvent() => _onNetCross(event),
      BounceEvent() => _onBounce(event),
      BallLostEvent() => _onBallLost(event),
    };
  }

  /// Watches for the rally to be initiated (see [requireServe]); never scores.
  PointDecision? _onPreRally(TrackerEvent event) {
    switch (event) {
      case BounceEvent e:
        if (_preCrossTo == e.side) {
          // The ball crossed the net and has now landed on the receiving
          // side: play is live (covers serves whose own-side bounce the
          // track missed).
          _rallyStarted = true;
          _lastBounceSide = e.side;
          _crossedSinceBounce = false;
        } else {
          _preBounceSide = e.side;
        }
      case NetCrossEvent e:
        if (_preBounceSide == e.from) {
          // The serve signature: a bounce on S then a crossing from S.
          _rallyStarted = true;
          _lastBounceSide = e.from;
          _crossedSinceBounce = true;
          _crossesSinceBounce = 1;
          _firstUnansweredCrossTo = e.to;
          _firstUnansweredCrossT = e.timestampMs;
        } else {
          _preCrossTo = e.to;
        }
      case BallLostEvent():
        // Whatever that was (a warm-up pass, a caught ball), it ended
        // without ever becoming a rally. Forget it, decide nothing.
        _preBounceSide = null;
        _preCrossTo = null;
    }
    return null;
  }

  PointDecision? _onNetCross(NetCrossEvent event) {
    // A successful crossing: the ball is now heading to the other side and the
    // previous bounce has been answered.
    if (_crossesSinceBounce == 0) {
      _firstUnansweredCrossTo = event.to;
      _firstUnansweredCrossT = event.timestampMs;
    }
    _crossesSinceBounce++;
    _crossedSinceBounce = true;
    return null;
  }

  PointDecision? _onBounce(BounceEvent event) {
    final sameSide = _lastBounceSide == event.side;
    if (sameSide && !_crossedSinceBounce) {
      // Two bounces on one side with no return in between: that side lost.
      return _decide(
        winner: playerOn(event.side).other,
        reason: PointReason.doubleBounce,
        timestampMs: event.timestampMs,
      );
    }
    // A legal bounce; remember it and wait for the return.
    _lastBounceSide = event.side;
    _crossedSinceBounce = false;
    _crossesSinceBounce = 0;
    _firstUnansweredCrossTo = null;
    return null;
  }

  PointDecision? _onBallLost(BallLostEvent event) {
    // The ball crossed the net two or more times since the last bounce and
    // play then stopped. A legal shot lands before anything else happens, so
    // the *first* of those crossings was a shot that missed the table — it
    // flew out of bounds, and the later crossing(s) were dead-ball drift.
    // Point to the receiver it was heading at, stamped when the shot crossed.
    if (_crossesSinceBounce >= 2 && _firstUnansweredCrossTo != null) {
      return _decide(
        winner: playerOn(_firstUnansweredCrossTo!),
        reason: PointReason.outOfBounds,
        timestampMs: _firstUnansweredCrossT,
      );
    }
    // The ball crossed into a side, never bounced, and was last tracked
    // already past that side's outer edge: it flew long over the baseline.
    // Out of bounds — point to the receiver it was heading at.
    if (_firstUnansweredCrossTo != null &&
        event.lostOutside == _firstUnansweredCrossTo) {
      return _decide(
        winner: playerOn(_firstUnansweredCrossTo!),
        reason: PointReason.outOfBounds,
        timestampMs: event.timestampMs,
      );
    }
    if (_lastBounceSide != null && !_crossedSinceBounce) {
      // The ball bounced on a side and then vanished without coming back over
      // the net — that side never returned it.
      return _decide(
        winner: playerOn(_lastBounceSide!).other,
        reason: PointReason.notReturned,
        timestampMs: event.timestampMs,
      );
    }
    // Lost in flight (after a single crossing, or before any bounce): the
    // path alone cannot attribute it — undetermined, the UI asks the user.
    return _decide(
      winner: null,
      reason: PointReason.outOfPlay,
      timestampMs: event.timestampMs,
    );
  }

  PointDecision _decide({
    required Player? winner,
    required PointReason reason,
    required int timestampMs,
  }) {
    reset();
    return PointDecision(
      winner: winner,
      reason: reason,
      timestampMs: timestampMs,
    );
  }

  /// Forget the current rally's state (called automatically after a decision).
  /// With [requireServe], the gate re-arms: the next rally must again be
  /// visibly initiated before events count.
  void reset() {
    _lastBounceSide = null;
    _crossedSinceBounce = false;
    _crossesSinceBounce = 0;
    _firstUnansweredCrossTo = null;
    _rallyStarted = false;
    _preBounceSide = null;
    _preCrossTo = null;
  }
}
