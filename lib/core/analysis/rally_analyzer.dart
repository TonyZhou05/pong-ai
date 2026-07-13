/// Pure-Dart rally-length analytics.
///
/// Match analysis apps (e.g. "Ball AI") headline *rally length* — how many
/// times the ball crossed the net before the point ended — because it captures
/// the character of a match: short first-strike points vs long grinding
/// exchanges. Nothing in the pipeline recorded it yet: the [BallTracker] already
/// emits a [NetCrossEvent] per return and the [RallyReferee] already delimits
/// rallies by producing a [PointDecision], so this layer just folds those two
/// signals into per-rally records and an aggregate [RallyStats].
///
/// Like the rest of `core/`, it has no Flutter or vision-plugin dependencies and
/// is driven entirely by [TrackerEvent]s + [PointDecision]s, so it is
/// unit-testable against synthetic rallies and replayed benchmark clips.
library;

import '../scoring/scoring_engine.dart';
import 'ball_tracker.dart';
import 'rally_referee.dart';

/// One completed rally: how many strokes it lasted and how long it took.
class Rally {
  const Rally({
    required this.strokeCount,
    required this.durationMs,
    required this.winner,
    required this.reason,
  });

  /// Number of times the ball crossed the net during the rally — the standard
  /// proxy for rally length (a serve that is not returned is one stroke).
  final int strokeCount;

  /// Wall-clock span from the rally's first observed event to the deciding one.
  final int durationMs;

  /// The point winner, or null when the referee could not attribute it
  /// ([PointReason.outOfPlay]).
  final Player? winner;

  final PointReason reason;

  @override
  String toString() =>
      'Rally($strokeCount strokes, ${durationMs}ms, ${winner ?? 'undetermined'})';
}

/// A rally's length category, used for the win-breakdown analytics.
enum RallyLength {
  /// Decided in at most 2 strokes (serve / first-strike points).
  short,

  /// 3–5 strokes.
  medium,

  /// 6 or more strokes (long, grinding exchanges).
  long,
}

/// Which length bucket a rally's stroke count falls into.
RallyLength rallyLengthOf(int strokeCount) {
  if (strokeCount <= 2) return RallyLength.short;
  if (strokeCount <= 5) return RallyLength.medium;
  return RallyLength.long;
}

/// Aggregated rally-length feedback over a match's [Rally]s.
class RallyStats {
  const RallyStats(this.rallies);

  final List<Rally> rallies;

  int get rallyCount => rallies.length;

  int get totalStrokes =>
      rallies.fold(0, (sum, r) => sum + r.strokeCount);

  double get averageStrokes =>
      rallies.isEmpty ? 0 : totalStrokes / rallies.length;

  /// The longest rally by stroke count.
  int get longestStrokes =>
      rallies.isEmpty ? 0 : rallies.map((r) => r.strokeCount).reduce(_max);

  double get averageDurationMs => rallies.isEmpty
      ? 0
      : rallies.fold(0, (sum, r) => sum + r.durationMs) / rallies.length;

  /// Rallies decided in at most 2 strokes (serve/first-strike points).
  int get shortRallies => rallies.where((r) => r.strokeCount <= 2).length;

  /// Rallies of 3–5 strokes.
  int get mediumRallies =>
      rallies.where((r) => r.strokeCount >= 3 && r.strokeCount <= 5).length;

  /// Rallies of 6 or more strokes (long, grinding exchanges).
  int get longRallies => rallies.where((r) => r.strokeCount >= 6).length;

  /// Rallies with a decided winner ([Rally.winner] non-null). A rally the
  /// referee left undetermined ([PointReason.outOfPlay]) is excluded from the
  /// win-breakdown denominators.
  int get decidedRallies => rallies.where((r) => r.winner != null).length;

  /// Rallies [p] won (across all lengths). Mines [Rally.winner], which the
  /// aggregate previously never consumed.
  int ralliesWonBy(Player p) => rallies.where((r) => r.winner == p).length;

  /// Rallies of [length] that [p] won — the "who thrives in short first-strike
  /// vs long grinding exchanges" breakdown. Long-rally dominance in particular
  /// is a headline endurance/consistency signal.
  int ralliesWonByLength(Player p, RallyLength length) => rallies
      .where((r) => r.winner == p && rallyLengthOf(r.strokeCount) == length)
      .length;

  /// Whether any rally carried a decided winner, i.e. the win breakdown is
  /// meaningful for this match.
  bool get hasWinData => rallies.any((r) => r.winner != null);

  static int _max(int a, int b) => a > b ? a : b;

  /// A deterministic, human-readable rally-length report.
  String report() {
    if (rallies.isEmpty) {
      return 'Rally analysis\nNo rallies recorded yet.';
    }
    final lines = [
      'Rally analysis',
      '$rallyCount rallies — avg ${averageStrokes.toStringAsFixed(1)} strokes, '
          'longest $longestStrokes.',
      'Avg rally length: ${(averageDurationMs / 1000).toStringAsFixed(1)}s.',
      '  • $shortRallies short (≤2)',
      '  • $mediumRallies medium (3–5)',
      '  • $longRallies long (≥6)',
    ];
    if (hasWinData) {
      lines
        ..add('Rally wins — '
            'A: ${ralliesWonBy(Player.a)}, B: ${ralliesWonBy(Player.b)}.')
        ..add('  • long (≥6) wins — '
            'A: ${ralliesWonByLength(Player.a, RallyLength.long)}, '
            'B: ${ralliesWonByLength(Player.b, RallyLength.long)}');
    }
    return lines.join('\n');
  }
}

/// Incrementally folds a match's rally events into [Rally] records.
///
/// Feed each [TrackerEvent] (as produced by [BallTracker.update]) to [observe],
/// then call [endRally] with the [RallyReferee]'s [PointDecision] the moment a
/// rally is decided. Read the accumulated [rallies] / aggregate [stats] at any
/// time.
class RallyAnalyzer {
  final List<Rally> _rallies = [];

  int? _firstEventMs;
  int _crossings = 0;

  /// All completed rallies so far, in order.
  List<Rally> get rallies => List.unmodifiable(_rallies);

  /// A live aggregate over the rallies recorded so far.
  RallyStats get stats => RallyStats(List.of(_rallies));

  /// Record one event of the current (in-progress) rally.
  void observe(TrackerEvent event) {
    _firstEventMs ??= event.timestampMs;
    if (event is NetCrossEvent) _crossings++;
  }

  /// Close out the current rally with the referee's [decision] and start a fresh
  /// one. The rally spans from its first observed event to [decision]; if no
  /// events preceded the decision its duration is zero.
  void endRally(PointDecision decision) {
    final start = _firstEventMs ?? decision.timestampMs;
    _rallies.add(
      Rally(
        strokeCount: _crossings,
        durationMs: decision.timestampMs - start,
        winner: decision.winner,
        reason: decision.reason,
      ),
    );
    _resetCurrent();
  }

  /// Forget all rally state (e.g. to start a new match).
  void reset() {
    _rallies.clear();
    _resetCurrent();
  }

  void _resetCurrent() {
    _firstEventMs = null;
    _crossings = 0;
  }
}
