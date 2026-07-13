/// Pure-Dart **coaching feedback** for a training session.
///
/// Every prior training layer *measures* the drill — depth, placement
/// consistency, lateral grouping, tempo/rhythm, pace — and surfaces each number
/// on its own. But a player reading a wall of percentages still has to decide
/// *what to actually work on next*. That prioritization — pick the weakest
/// coachable dimension and turn it into one actionable cue — was never done:
/// [TrackingQualityAnalyzer] only advises on phone placement / detection health,
/// not stroke technique, and [TrainingSummary.report] just lists the metrics.
///
/// [TrainingFeedback] folds a finished [TrainingSummary] (plus the
/// [TrainingConfig] that defines the target) into scored coachable dimensions,
/// then names the weakest as the *focus* and the strongest as the confirmed
/// *strength*. Like the rest of `core/training/`, it is pure Dart derived only
/// from the summary, so it is unit-testable with no Flutter / vision plugin.
library;

import 'shot_analyzer.dart';

/// One coachable aspect of a player's shots, scored in `[0, 1]` (higher is
/// better) with an actionable cue used when it is the session's weak point.
class FeedbackDimension {
  const FeedbackDimension({
    required this.name,
    required this.score,
    required this.tip,
  });

  /// Short human label, e.g. `Placement accuracy`.
  final String name;

  /// Quality of this dimension in `[0, 1]` (1 = ideal, 0 = poor).
  final double score;

  /// A concrete coaching cue to improve this dimension.
  final String tip;
}

/// Turns a training session's aggregate metrics into a prioritized coaching cue.
class TrainingFeedback {
  TrainingFeedback(this.summary, {this.config = const TrainingConfig()})
      : dimensions = _score(summary, config);

  final TrainingSummary summary;
  final TrainingConfig config;

  /// The coachable dimensions that could be assessed for this session, in a
  /// fixed order (placement accuracy first). Empty when no shots were recorded.
  final List<FeedbackDimension> dimensions;

  /// A dimension whose score is at/above this is considered already dialed in,
  /// so a session where even the weakest dimension clears the bar earns a
  /// "keep it up" message rather than a fix-it cue.
  static const double _goodEnough = 0.8;

  bool get hasData => dimensions.isNotEmpty;

  /// The weakest coachable dimension (lowest score; ties resolve to the earlier
  /// one, i.e. placement accuracy before consistency), or null with no shots.
  FeedbackDimension? get weakest {
    if (dimensions.isEmpty) return null;
    var worst = dimensions.first;
    for (final d in dimensions) {
      if (d.score < worst.score) worst = d;
    }
    return worst;
  }

  /// The strongest coachable dimension (highest score), or null with no shots.
  FeedbackDimension? get strongest {
    if (dimensions.isEmpty) return null;
    var best = dimensions.first;
    for (final d in dimensions) {
      if (d.score > best.score) best = d;
    }
    return best;
  }

  /// The single most useful thing to work on next — the weakest dimension's
  /// cue, or an encouragement when everything already clears [_goodEnough].
  /// Null only when there are no shots to assess.
  String? get focusTip {
    final worst = weakest;
    if (worst == null) return null;
    if (worst.score >= _goodEnough) {
      return 'Solid all around — keep the same drill and push the pace.';
    }
    return worst.tip;
  }

  /// Build the ordered, scored dimension list for a session. A single shot only
  /// supports the placement-accuracy judgement (consistency/rhythm need at least
  /// two shots to mean anything), so those are omitted below that threshold.
  static List<FeedbackDimension> _score(
    TrainingSummary summary,
    TrainingConfig config,
  ) {
    if (summary.shots.isEmpty) return const [];
    final dims = <FeedbackDimension>[];

    // Placement accuracy: how close the average landing depth is to target.
    final depthError = (summary.averageDepth - config.targetDepth).abs();
    final placement = (1 - depthError / config.depthTolerance).clamp(0.0, 1.0);
    final tooShort = summary.averageDepth < config.targetDepth;
    dims.add(
      FeedbackDimension(
        name: 'Placement accuracy',
        score: placement,
        tip: tooShort
            ? 'Land the ball deeper — aim closer to the far baseline.'
            : 'Bring the ball in shorter — you are overshooting the target.',
      ),
    );

    // On-table accuracy: the most fundamental coachable dimension — did the ball
    // even stay on the table? Placement/consistency/pace only score the strokes
    // that *landed* on the target half, so a player who keeps missing the table
    // entirely gets coached on the depth precision of their few good shots
    // instead of their real weakness. Assessed only when a miss was recorded
    // (missedShots > 0), matching the on-table-accuracy report line's guard, so
    // a clean session (no misses) is unchanged.
    if (summary.missedShots > 0) {
      dims.add(
        FeedbackDimension(
          name: 'On-table accuracy',
          score: summary.onTableRate,
          tip: 'Keep the ball on the table — control the stroke so more shots '
              'land in before working on placement.',
        ),
      );
    }

    if (summary.shots.length >= 2) {
      dims.add(
        FeedbackDimension(
          name: 'Depth consistency',
          score: summary.consistency,
          tip: 'Group your shots — repeat the same landing depth every stroke.',
        ),
      );
      dims.add(
        FeedbackDimension(
          name: 'Lateral consistency',
          score: summary.lateralConsistency,
          tip: 'Tighten your side-to-side aim into one spot on the table.',
        ),
      );
      dims.add(
        FeedbackDimension(
          name: 'Rhythm',
          score: summary.rhythmConsistency,
          tip: 'Even out your timing — keep a steady tempo between shots.',
        ),
      );
    }

    // Shot pace: the physical power dimension, assessable only when a km/h scale
    // exists (calibrated ruler + a shot that carried real x-motion). Placement,
    // consistency and rhythm are all accuracy metrics — hitting on target but
    // soft still leaves pace uncoached. Scored as average pace against the
    // target, plateauing at full marks once the target pace is reached (driving
    // harder than target is not penalized; overshooting shows up as placement).
    if (summary.averageSpeedKmh > 0) {
      final pace = (summary.averageSpeedKmh / config.targetSpeedKmh)
          .clamp(0.0, 1.0)
          .toDouble();
      dims.add(
        FeedbackDimension(
          name: 'Shot pace',
          score: pace,
          tip: 'Drive through the ball — add pace to put the opponent under '
              'pressure.',
        ),
      );
    }
    return dims;
  }

  /// A deterministic, human-readable coaching section mirroring the other
  /// training text reports.
  String report() {
    if (dimensions.isEmpty) {
      return 'Coaching\nNo shots recorded yet.';
    }
    final lines = <String>['Coaching'];
    final worst = weakest!;
    final best = strongest!;
    lines.add('Focus next: ${focusTip!}');
    // Only call out a strength distinct from the focus area.
    if (best.name != worst.name && best.score >= _goodEnough) {
      lines.add('Strength: ${best.name.toLowerCase()} is dialed in.');
    }
    for (final d in dimensions) {
      lines.add('  • ${d.name}: ${(d.score * 100).round()}%');
    }
    return lines.join('\n');
  }
}
