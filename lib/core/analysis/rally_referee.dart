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
  RallyReferee({Player leftPlayer = Player.a}) : _leftPlayer = leftPlayer;

  /// Which player occupies the left half of the table (split by the net). The
  /// other player is on the right. This maps the tracker's spatial
  /// [TableSide] onto the scoring engine's [Player] identities.
  final Player _leftPlayer;

  /// The player on the left half of the table (net-split). Exposed so movement
  /// analytics can attribute detected people to the same [Player] identities.
  Player get leftPlayer => _leftPlayer;

  /// The side of the most recent legal bounce this rally, or null before one.
  TableSide? _lastBounceSide;

  /// Whether the ball has crossed the net since [_lastBounceSide] was recorded.
  bool _crossedSinceBounce = false;

  /// The player on a given side of the table.
  Player playerOn(TableSide side) =>
      side == TableSide.left ? _leftPlayer : _leftPlayer.other;

  /// Process one rally event; returns a verdict if it ends the rally.
  PointDecision? update(TrackerEvent event) {
    return switch (event) {
      NetCrossEvent() => _onNetCross(),
      BounceEvent() => _onBounce(event),
      BallLostEvent() => _onBallLost(event),
    };
  }

  PointDecision? _onNetCross() {
    // A successful crossing: the ball is now heading to the other side and the
    // previous bounce has been answered.
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
    return null;
  }

  PointDecision? _onBallLost(BallLostEvent event) {
    if (_lastBounceSide != null && !_crossedSinceBounce) {
      // The ball bounced on a side and then vanished without coming back over
      // the net — that side never returned it.
      return _decide(
        winner: playerOn(_lastBounceSide!).other,
        reason: PointReason.notReturned,
        timestampMs: event.timestampMs,
      );
    }
    // Lost in flight (after a crossing, or before any bounce): undetermined.
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
  void reset() {
    _lastBounceSide = null;
    _crossedSinceBounce = false;
  }
}
