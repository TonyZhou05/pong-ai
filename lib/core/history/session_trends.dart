/// Across-session progression / trends over the saved session history.
///
/// [SessionHistoryStore] (iteration 45) persists each match / training report
/// and [SessionHistoryScreen] (iteration 46/47) lists them one-by-one — but
/// nothing ever looked at the *collection*. Both structured exporters
/// (`buildMatchReportJson`, `buildTrainingReportJson`) name this as their whole
/// reason for existing: "diff pace/placement/rhythm across sessions", "diffed
/// across sessions". A single drill's grade tells you how today went; whether
/// you are actually *improving* only shows up across many sessions.
///
/// [SessionTrends] folds a list of [StoredSession]s (the exact records the store
/// hands back) into training-progression metrics — score improvement over time,
/// the best session, latest-vs-first deltas — plus a small match tally. Like the
/// rest of `core/`, it is pure Dart (parses the stored JSON maps only, no
/// Flutter / vision / plugin) and unit-testable end-to-end.
library;

import 'session_history_store.dart';

/// One training session reduced to the headline metrics a progression view
/// needs, parsed out of its stored `buildTrainingReportJson` map.
class TrainingTrendPoint {
  const TrainingTrendPoint({
    required this.id,
    required this.savedAt,
    required this.shotCount,
    required this.averageScore,
    required this.overallGrade,
    this.depthConsistency,
    this.maxSpeedKmh,
    this.rhythmConsistency,
    this.focusArea,
  });

  final String id;
  final DateTime savedAt;
  final int shotCount;
  final double averageScore;
  final String overallGrade;

  /// Placement consistency (population stddev of landing depth); lower is
  /// tighter. Null if the stored report predates the field.
  final double? depthConsistency;

  /// Peak physical shot speed in km/h, null if the session recorded no scaled
  /// pace (e.g. an all-slow drill or an older report).
  final double? maxSpeedKmh;

  /// Metronome rhythm score in [0,1]; null if the drill had < 2 shots.
  final double? rhythmConsistency;

  /// The coaching *focus* the session flagged as its weakest dimension (e.g.
  /// `Placement accuracy`), from the persisted `coaching.focus` field. Null if
  /// the report recorded no coachable data (no shots) or predates the field.
  final String? focusArea;

  /// Parse a stored training session, or null if the report shape is not a
  /// recognizable training export (so a corrupt / foreign record is skipped).
  static TrainingTrendPoint? fromStored(StoredSession session) {
    if (session.kind != SessionKind.training) return null;
    final s = session.report['session'];
    if (s is! Map) return null;
    final avg = _asDouble(s['averageScore']);
    final shots = _asInt(s['shotCount']);
    final grade = s['overallGrade'];
    if (avg == null || shots == null || grade is! String) return null;

    final placement = session.report['placement'];
    final pace = session.report['pace'];
    final tempo = session.report['tempo'];
    final coaching = session.report['coaching'];
    final focus = coaching is Map ? coaching['focus'] : null;
    return TrainingTrendPoint(
      id: session.id,
      savedAt: session.savedAt,
      shotCount: shots,
      averageScore: avg,
      overallGrade: grade,
      depthConsistency:
          placement is Map ? _asDouble(placement['depthConsistency']) : null,
      maxSpeedKmh: pace is Map ? _asDouble(pace['maxSpeedKmh']) : null,
      rhythmConsistency:
          tempo is Map ? _asDouble(tempo['rhythmConsistency']) : null,
      focusArea: focus is String ? focus : null,
    );
  }
}

/// Cross-session progression over a saved-session history.
class SessionTrends {
  const SessionTrends({
    required this.trainingSessions,
    required this.matchCount,
  });

  /// Every parseable training session, oldest first (so index 0 is where the
  /// player started and the last is their most recent drill).
  final List<TrainingTrendPoint> trainingSessions;

  /// How many stored sessions were matches. Matches pit Player A vs B rather
  /// than a single tracked user, so there is no personal win-rate to trend;
  /// the count still situates the training history in the whole record.
  final int matchCount;

  /// Fold the store's records (in any order) into trends. Training sessions are
  /// sorted oldest-first by save time (ties broken by id) so deltas read as
  /// first → latest.
  factory SessionTrends.fromSessions(Iterable<StoredSession> sessions) {
    final training = <TrainingTrendPoint>[];
    var matches = 0;
    for (final session in sessions) {
      if (session.kind == SessionKind.match) {
        matches++;
        continue;
      }
      final point = TrainingTrendPoint.fromStored(session);
      if (point != null) training.add(point);
    }
    training.sort((a, b) {
      final byTime = a.savedAt.compareTo(b.savedAt);
      return byTime != 0 ? byTime : a.id.compareTo(b.id);
    });
    return SessionTrends(trainingSessions: training, matchCount: matches);
  }

  int get trainingCount => trainingSessions.length;

  /// Whether there are enough training sessions (>= 2) for a first→latest
  /// improvement to be meaningful.
  bool get hasTrainingTrend => trainingSessions.length >= 2;

  TrainingTrendPoint? get firstSession =>
      trainingSessions.isEmpty ? null : trainingSessions.first;

  TrainingTrendPoint? get latestSession =>
      trainingSessions.isEmpty ? null : trainingSessions.last;

  /// The training session with the highest average score (ties broken toward
  /// the more recent one), or null if there are none.
  TrainingTrendPoint? get bestSession {
    TrainingTrendPoint? best;
    for (final p in trainingSessions) {
      if (best == null || p.averageScore >= best.averageScore) best = p;
    }
    return best;
  }

  /// Mean average-score across all training sessions, null if there are none.
  double? get meanScore {
    if (trainingSessions.isEmpty) return null;
    final total =
        trainingSessions.fold<double>(0, (sum, p) => sum + p.averageScore);
    return total / trainingSessions.length;
  }

  /// Latest average score minus the first: positive means the player's shot
  /// quality has improved over the tracked history. Null with < 2 sessions.
  double? get scoreImprovement {
    if (!hasTrainingTrend) return null;
    return trainingSessions.last.averageScore -
        trainingSessions.first.averageScore;
  }

  /// Fastest shot (km/h) recorded across every session that scaled pace, or
  /// null if none did — the personal-best radar number.
  double? get bestMaxSpeedKmh {
    double? best;
    for (final p in trainingSessions) {
      final s = p.maxSpeedKmh;
      if (s == null) continue;
      if (best == null || s > best) best = s;
    }
    return best;
  }

  /// Change in peak physical shot speed (km/h) from the first to the latest
  /// session that recorded a scaled pace: returns `latest − first`, so a
  /// positive value means the player is hitting harder over the tracked
  /// history. Null unless at least two sessions carry a km/h speed.
  double? get speedImprovement {
    final series = _metricSeries((p) => p.maxSpeedKmh);
    if (series.length < 2) return null;
    return series.last - series.first;
  }

  /// Change in landing-placement consistency (population stddev of shot depth)
  /// from the first to the latest session that recorded it. Depth stddev is
  /// *lower-is-tighter*, so this returns `first − latest`: a positive value
  /// means placement got tighter (improved). Null unless at least two sessions
  /// carry the metric.
  double? get depthConsistencyImprovement {
    final series = _metricSeries((p) => p.depthConsistency);
    if (series.length < 2) return null;
    return series.first - series.last;
  }

  /// Change in metronome rhythm consistency ([0,1], *higher-is-steadier*) from
  /// the first to the latest session that recorded it: returns `latest − first`,
  /// so a positive value means the drill tempo got steadier. Null unless at
  /// least two sessions carry the metric.
  double? get rhythmConsistencyImprovement {
    final series = _metricSeries((p) => p.rhythmConsistency);
    if (series.length < 2) return null;
    return series.last - series.first;
  }

  /// How many training sessions across the history flagged each coaching focus
  /// area as their weakest dimension, e.g. `{Placement accuracy: 3, Rhythm: 1}`.
  /// Sessions with no recorded focus (no shots / older report) are excluded.
  Map<String, int> get focusCounts {
    final counts = <String, int>{};
    for (final p in trainingSessions) {
      final focus = p.focusArea;
      if (focus == null) continue;
      counts[focus] = (counts[focus] ?? 0) + 1;
    }
    return counts;
  }

  /// The coaching focus area that comes up most often across the saved drills —
  /// the player's *persistent* weak point. Ties break toward the area whose most
  /// recent session is newer (so the current struggle wins). Null if no session
  /// recorded a focus.
  String? get recurringFocus {
    final counts = focusCounts;
    if (counts.isEmpty) return null;
    // Latest occurrence index per focus, for deterministic recency tie-breaking.
    final latestIndex = <String, int>{};
    for (var i = 0; i < trainingSessions.length; i++) {
      final focus = trainingSessions[i].focusArea;
      if (focus != null) latestIndex[focus] = i;
    }
    String? best;
    var bestCount = 0;
    var bestRecency = -1;
    counts.forEach((focus, count) {
      final recency = latestIndex[focus] ?? -1;
      if (count > bestCount ||
          (count == bestCount && recency > bestRecency)) {
        best = focus;
        bestCount = count;
        bestRecency = recency;
      }
    });
    return best;
  }

  /// How many sessions flagged [recurringFocus] as their weak point (0 if none).
  int get recurringFocusCount {
    final focus = recurringFocus;
    return focus == null ? 0 : (focusCounts[focus] ?? 0);
  }

  /// Whether a coaching focus recurs across at least two drills — enough to read
  /// as a persistent weak point worth calling out rather than a one-off.
  bool get hasRecurringFocus => recurringFocusCount >= 2;

  /// The recorded values of a nullable per-session metric, in session order
  /// (oldest first), skipping sessions that did not record it.
  List<double> _metricSeries(double? Function(TrainingTrendPoint) select) {
    final out = <double>[];
    for (final p in trainingSessions) {
      final v = select(p);
      if (v != null) out.add(v);
    }
    return out;
  }

  /// A short human-readable progression summary, mirroring the text-report style
  /// of the per-session analytics.
  String report() {
    final lines = <String>['Progress across saved sessions'];
    lines.add(
      'Sessions: $trainingCount training'
      '${matchCount > 0 ? ', $matchCount match' : ''}',
    );
    if (trainingSessions.isEmpty) {
      lines.add('No training drills saved yet.');
      return lines.join('\n');
    }

    final mean = meanScore!;
    lines.add('Average shot score: ${_pct(mean)}');

    final improvement = scoreImprovement;
    if (improvement != null) {
      final first = trainingSessions.first.averageScore;
      final latest = trainingSessions.last.averageScore;
      final verb = improvement > 0.0005
          ? 'up'
          : (improvement < -0.0005 ? 'down' : 'flat');
      lines.add(
        'Trend: ${_pct(first)} → ${_pct(latest)} '
        '($verb ${_signedPct(improvement)})',
      );
    }

    final best = bestSession!;
    lines.add(
      'Best session: grade ${best.overallGrade} '
      '(${_pct(best.averageScore)}, ${best.shotCount} shots)',
    );

    final topSpeed = bestMaxSpeedKmh;
    if (topSpeed != null) {
      lines.add('Fastest shot: ${topSpeed.toStringAsFixed(1)} km/h');
    }

    final speedTrend = speedImprovement;
    if (speedTrend != null) {
      final verb = speedTrend > 0.05
          ? 'up ${speedTrend.toStringAsFixed(1)} km/h'
          : (speedTrend < -0.05
              ? 'down ${speedTrend.abs().toStringAsFixed(1)} km/h'
              : 'flat');
      lines.add('Shot speed: $verb');
    }

    final depthTrend = depthConsistencyImprovement;
    if (depthTrend != null) {
      final verb = depthTrend > 0.0005
          ? 'tighter'
          : (depthTrend < -0.0005 ? 'looser' : 'flat');
      lines.add('Placement consistency: $verb');
    }

    final rhythmTrend = rhythmConsistencyImprovement;
    if (rhythmTrend != null) {
      final verb = rhythmTrend > 0.0005
          ? 'up ${_signedPct(rhythmTrend)}'
          : (rhythmTrend < -0.0005 ? 'down ${_signedPct(rhythmTrend)}' : 'flat');
      lines.add('Rhythm consistency: $verb');
    }

    if (hasRecurringFocus) {
      lines.add(
        'Recurring focus: $recurringFocus '
        '($recurringFocusCount of $trainingCount drills)',
      );
    }
    return lines.join('\n');
  }

  static String _pct(double v) => '${(v * 100).round()}%';

  static String _signedPct(double v) {
    final rounded = (v * 100).round();
    return '${rounded >= 0 ? '+' : ''}$rounded%';
  }
}

double? _asDouble(Object? v) => v is num ? v.toDouble() : null;

int? _asInt(Object? v) => v is num ? v.toInt() : null;
