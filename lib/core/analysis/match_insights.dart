/// Pure-Dart **per-player coaching insights** for a completed match.
///
/// [MatchSummary] measures the match from every angle — serve holds, return
/// breaks, game-point conversion — but, exactly like the training side before
/// [TrainingFeedback], each number is surfaced on its own and a player still has
/// to decide *what to work on*. The training path got a prioritized "focus next"
/// cue in iteration 51; the match path never did, so turning the per-player
/// match metrics into one actionable weakness was a genuine parallel gap.
///
/// [MatchInsights] folds a finished [MatchSummary] into scored coachable
/// dimensions *for each player* (serve effectiveness, return of serve,
/// closing games, and saving game points under pressure), then names each
/// player's weakest dimension as
/// their focus and their strongest as a confirmed strength. Like the rest of
/// `core/analysis/`, it is pure Dart derived only from the summary, so it is
/// unit-testable with no Flutter or vision plugin.
library;

import '../scoring/scoring_engine.dart';
import 'match_summary.dart';

/// One coachable aspect of a player's match, scored in `[0, 1]` (higher is
/// better) with an actionable cue used when it is that player's weak point.
class InsightDimension {
  const InsightDimension({
    required this.name,
    required this.score,
    required this.tip,
  });

  /// Short human label, e.g. `Serve effectiveness`.
  final String name;

  /// Quality of this dimension in `[0, 1]` (1 = ideal, 0 = poor).
  final double score;

  /// A concrete coaching cue to improve this dimension.
  final String tip;
}

/// The prioritized coaching read for a single player: the coachable dimensions
/// that could be assessed, plus the weakest (focus) and strongest (strength).
class PlayerInsights {
  const PlayerInsights(this.player, this.dimensions);

  final Player player;

  /// The coachable dimensions assessed for this player, in a fixed order (serve
  /// first). Empty when the match log carried none of the required signals for
  /// this player (no serve data and no receive/game-point situations).
  final List<InsightDimension> dimensions;

  /// A dimension at/above this is considered already solid, so a player whose
  /// weakest dimension still clears the bar earns encouragement, not a fix-it.
  static const double _goodEnough = 0.6;

  bool get hasData => dimensions.isNotEmpty;

  /// The weakest coachable dimension (lowest score; ties resolve to the earlier
  /// one, i.e. serve before return before clutch), or null with no data.
  InsightDimension? get weakest {
    if (dimensions.isEmpty) return null;
    var worst = dimensions.first;
    for (final d in dimensions) {
      if (d.score < worst.score) worst = d;
    }
    return worst;
  }

  /// The strongest coachable dimension (highest score), or null with no data.
  InsightDimension? get strongest {
    if (dimensions.isEmpty) return null;
    var best = dimensions.first;
    for (final d in dimensions) {
      if (d.score > best.score) best = d;
    }
    return best;
  }

  /// The player's overall match rating in `[0, 1]` — the mean of every assessed
  /// coachable dimension. Null when there is no data to assess.
  double? get overallScore {
    if (dimensions.isEmpty) return null;
    var sum = 0.0;
    for (final d in dimensions) {
      sum += d.score;
    }
    return sum / dimensions.length;
  }

  /// An A–F letter grade for the player's match, from [overallScore]. Mirrors
  /// the training-mode session grade so both modes surface one headline rating.
  /// Returns `'–'` when there is no data to assess.
  String get grade {
    final s = overallScore;
    if (s == null) return '–';
    if (s >= 0.85) return 'A';
    if (s >= 0.7) return 'B';
    if (s >= 0.55) return 'C';
    if (s >= 0.4) return 'D';
    return 'F';
  }

  /// The single most useful thing this player should work on next — the weakest
  /// dimension's cue, or encouragement when everything already clears
  /// [_goodEnough]. Null only when there is no data to assess.
  String? get focusTip {
    final worst = weakest;
    if (worst == null) return null;
    if (worst.score >= _goodEnough) {
      return 'Well-rounded match — keep it up and add more pace.';
    }
    return worst.tip;
  }
}

/// Turns a match's per-player aggregate metrics into prioritized coaching cues.
class MatchInsights {
  MatchInsights(this.summary);

  final MatchSummary summary;

  /// Whether any player has at least one assessable coaching dimension.
  bool get hasData =>
      Player.values.any((p) => insightsFor(p).dimensions.isNotEmpty);

  /// The coaching read for [p], computed from the summary's per-player metrics.
  PlayerInsights insightsFor(Player p) =>
      PlayerInsights(p, _score(summary, p));

  static List<InsightDimension> _score(MatchSummary summary, Player p) {
    final dims = <InsightDimension>[];

    // Serve effectiveness: fraction of own-serve points won (serve holds).
    final serveRate = summary.serveWinRateFor(p);
    if (serveRate != null) {
      dims.add(
        InsightDimension(
          name: 'Serve effectiveness',
          score: serveRate,
          tip: 'Sharpen your serve — add spin and placement to win more '
              'service points.',
        ),
      );
    }

    // Return of serve: fraction of the opponent's serves that this player
    // broke. Receive points played = points the opponent served.
    final receivePlayed = summary.servePointsPlayedBy(p.other);
    if (receivePlayed > 0) {
      dims.add(
        InsightDimension(
          name: 'Return of serve',
          score: summary.receivePointsWonBy(p) / receivePlayed,
          tip: 'Attack the return — take the initiative when receiving serve.',
        ),
      );
    }

    // Clutch: fraction of held game points converted to close out the game.
    final clutch = summary.gamePointConversionRateFor(p);
    if (clutch != null) {
      dims.add(
        InsightDimension(
          name: 'Closing games',
          score: clutch,
          tip: 'Close out games — stay aggressive on your game-point chances.',
        ),
      );
    }

    // Defensive clutch: fraction of faced game points saved (denying the
    // opponent the game). The complement to closing — a player who repeatedly
    // gets broken when down game point has a save-under-pressure weakness.
    final saveRate = summary.gamePointSaveRateFor(p);
    if (saveRate != null) {
      dims.add(
        InsightDimension(
          name: 'Saving game points',
          score: saveRate,
          tip: 'Dig in when down game point — stay patient and force one more '
              'rally to save it.',
        ),
      );
    }

    return dims;
  }

  /// A deterministic, human-readable coaching section mirroring the other match
  /// text reports, with a focus line and dimension breakdown per player.
  String report() {
    final lines = <String>['Coaching insights'];
    if (!hasData) {
      lines.add('  • not enough data yet');
      return lines.join('\n');
    }
    for (final p in Player.values) {
      final insights = insightsFor(p);
      final name = p == Player.a ? 'Player A' : 'Player B';
      if (!insights.hasData) {
        lines.add('$name — not enough data');
        continue;
      }
      lines.add(
        '$name — Grade ${insights.grade}, Focus: ${insights.focusTip!}',
      );
      for (final d in insights.dimensions) {
        lines.add('  • ${d.name}: ${(d.score * 100).round()}%');
      }
    }
    return lines.join('\n');
  }
}
