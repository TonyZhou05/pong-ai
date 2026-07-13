/// Pure-Dart post-match performance analysis.
///
/// The [RallyReferee] already explains *why* every point ended (a double
/// bounce, a ball that was never returned, or an open-play resolution). On its
/// own each verdict is transient — applied to the score and forgotten. This
/// layer retains the sequence of awarded points ([ScoredPoint]) and derives the
/// performance story from it ([MatchSummary]): who won, how many points came
/// from forcing the opponent into errors versus open play, each player's
/// longest run, and a human-readable report.
///
/// Like the rest of `core/`, it has no Flutter or vision dependencies, so the
/// analytics can be unit-tested against a synthetic point log.
library;

import '../scoring/scoring_engine.dart';
import 'rally_referee.dart';

/// One point awarded during a match, tagged with why the rally ended.
///
/// This is the durable record the referee's transient [PointDecision] becomes
/// once a winner is known (either auto-attributed or resolved by the user).
class ScoredPoint {
  const ScoredPoint({
    required this.winner,
    required this.reason,
    required this.timestampMs,
  });

  final Player winner;
  final PointReason reason;
  final int timestampMs;

  @override
  String toString() => 'ScoredPoint($winner, $reason, @$timestampMs)';
}

/// Aggregated performance analysis over a match's [ScoredPoint] log.
class MatchSummary {
  const MatchSummary({required this.points, required this.finalState});

  /// The ordered log of every awarded point.
  final List<ScoredPoint> points;

  /// The score at the moment the summary was taken.
  final MatchState finalState;

  /// Total points played (rallies with a decided winner).
  int get totalPoints => points.length;

  /// Points won by [p] across the whole match.
  int pointsWonBy(Player p) =>
      points.where((point) => point.winner == p).length;

  /// Points [p] won grouped by why the rally ended.
  Map<PointReason, int> reasonBreakdownFor(Player p) {
    final counts = <PointReason, int>{
      for (final reason in PointReason.values) reason: 0,
    };
    for (final point in points) {
      if (point.winner == p) counts[point.reason] = counts[point.reason]! + 1;
    }
    return counts;
  }

  /// Points [p] won because the opponent failed to keep the ball in play — a
  /// double bounce or a shot that was never returned. These reflect [p]
  /// forcing errors (pressure, placement, pace).
  int forcedErrorsWonBy(Player p) {
    final breakdown = reasonBreakdownFor(p);
    return breakdown[PointReason.doubleBounce]! +
        breakdown[PointReason.notReturned]!;
  }

  /// Points [p] won that were settled in open play (lost-in-flight rallies the
  /// user resolved manually, e.g. a smash winner or an out-of-play call).
  int openPlayPointsWonBy(Player p) =>
      reasonBreakdownFor(p)[PointReason.outOfPlay]!;

  /// The longest run of consecutive points won by [p].
  int longestStreakFor(Player p) {
    var longest = 0;
    var current = 0;
    for (final point in points) {
      if (point.winner == p) {
        current += 1;
        if (current > longest) longest = current;
      } else {
        current = 0;
      }
    }
    return longest;
  }

  /// Wall-clock span between the first and last recorded point, in ms.
  int get durationMs => points.length < 2
      ? 0
      : points.last.timestampMs - points.first.timestampMs;

  /// The match winner, or null if the match is not over yet.
  Player? get matchWinner {
    if (!finalState.isMatchOver) return null;
    return finalState.gamesA > finalState.gamesB ? Player.a : Player.b;
  }

  static String _fmtDuration(int ms) {
    final totalSeconds = ms ~/ 1000;
    final minutes = totalSeconds ~/ 60;
    final seconds = totalSeconds % 60;
    return '$minutes:${seconds.toString().padLeft(2, '0')}';
  }

  static String _name(Player p) => p == Player.a ? 'Player A' : 'Player B';

  /// A deterministic, human-readable performance report.
  String report() {
    final lines = <String>['Match summary'];

    final winner = matchWinner;
    if (winner != null) {
      lines.add(
        '${_name(winner)} wins '
        '${finalState.gamesA}–${finalState.gamesB} in games.',
      );
    } else {
      lines.add(
        'In progress — games ${finalState.gamesA}–${finalState.gamesB}, '
        'points ${finalState.pointsA}–${finalState.pointsB}.',
      );
    }
    lines.add('$totalPoints points played over ${_fmtDuration(durationMs)}.');

    for (final player in Player.values) {
      lines
        ..add('')
        ..add('${_name(player)} — ${pointsWonBy(player)} pts')
        ..add('  • ${forcedErrorsWonBy(player)} won on forced errors')
        ..add('  • ${openPlayPointsWonBy(player)} won in open play')
        ..add('  • longest run: ${longestStreakFor(player)}');
    }

    return lines.join('\n');
  }
}
