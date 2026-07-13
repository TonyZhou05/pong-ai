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
    this.onTableRate,
    this.longestOnTargetStreak,
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

  /// Fraction of attempted strokes that landed on the table ([0,1], *higher-is-
  /// better*), from the persisted `session.onTableRate` field. Null if the drill
  /// attempted no strokes or the report predates the field.
  final double? onTableRate;

  /// Best run of consecutive on-target (good-or-better) shots in the session,
  /// from the persisted `session.longestOnTargetStreak` field. Null if the report
  /// predates the field (iteration 78).
  final int? longestOnTargetStreak;

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
      onTableRate: _asDouble(s['onTableRate']),
      longestOnTargetStreak: _asInt(s['longestOnTargetStreak']),
    );
  }
}

/// One saved match reduced to the cumulative-total metrics a "career" view
/// needs, parsed out of its stored `buildMatchReportJson` map. Every field
/// beyond id/time is nullable so a match saved before a given analytics layer
/// existed (or one with no ball tracked) still parses and counts.
class MatchTrendPoint {
  const MatchTrendPoint({
    required this.id,
    required this.savedAt,
    this.totalPoints,
    this.durationMs,
    this.maxBallSpeedKmh,
    this.longestRallyStrokes,
    this.averageRallyStrokes,
    this.winner,
  });

  final String id;
  final DateTime savedAt;

  /// Points played in the match, null if the report predates the summary block.
  final int? totalPoints;

  /// Match wall-clock duration in ms, null if unrecorded.
  final int? durationMs;

  /// Peak ball speed in km/h, null if the match tracked no scaled ball speed.
  final double? maxBallSpeedKmh;

  /// Longest rally in strokes, null if no rally data was recorded.
  final int? longestRallyStrokes;

  /// Mean rally length (strokes) in the match, null if no rally data was
  /// recorded — the per-match *typical* rally, complementing the peak
  /// [longestRallyStrokes].
  final double? averageRallyStrokes;

  /// The match winner key (`A` / `B`), null if the match did not finish.
  final String? winner;

  /// Parse a stored match session, or null if it is not a match record.
  static MatchTrendPoint? fromStored(StoredSession session) {
    if (session.kind != SessionKind.match) return null;
    final report = session.report;
    final summary = report['summary'];
    final ballSpeed = report['ballSpeed'];
    final rallies = report['rallies'];
    final score = report['score'];
    final winner = score is Map ? score['winner'] : null;
    return MatchTrendPoint(
      id: session.id,
      savedAt: session.savedAt,
      totalPoints: summary is Map ? _asInt(summary['totalPoints']) : null,
      durationMs: summary is Map ? _asInt(summary['durationMs']) : null,
      maxBallSpeedKmh:
          ballSpeed is Map ? _asDouble(ballSpeed['maxKmh']) : null,
      longestRallyStrokes:
          rallies is Map ? _asInt(rallies['longestStrokes']) : null,
      averageRallyStrokes:
          rallies is Map ? _asDouble(rallies['averageStrokes']) : null,
      winner: winner is String ? winner : null,
    );
  }
}

/// Cross-session progression over a saved-session history.
class SessionTrends {
  const SessionTrends({
    required this.trainingSessions,
    required this.matchSessions,
  });

  /// Every parseable training session, oldest first (so index 0 is where the
  /// player started and the last is their most recent drill).
  final List<TrainingTrendPoint> trainingSessions;

  /// Every saved match, oldest first. Matches pit Player A vs B rather than a
  /// single tracked user, so there is no personal win-rate to trend — but the
  /// collection still holds meaningful cumulative "career" totals (points
  /// played, fastest ball ever tracked, longest rally) worth surfacing.
  final List<MatchTrendPoint> matchSessions;

  /// Fold the store's records (in any order) into trends. Sessions are sorted
  /// oldest-first by save time (ties broken by id) so deltas read as
  /// first → latest.
  factory SessionTrends.fromSessions(Iterable<StoredSession> sessions) {
    final training = <TrainingTrendPoint>[];
    final matches = <MatchTrendPoint>[];
    for (final session in sessions) {
      if (session.kind == SessionKind.match) {
        final m = MatchTrendPoint.fromStored(session);
        if (m != null) matches.add(m);
        continue;
      }
      final point = TrainingTrendPoint.fromStored(session);
      if (point != null) training.add(point);
    }
    training.sort((a, b) {
      final byTime = a.savedAt.compareTo(b.savedAt);
      return byTime != 0 ? byTime : a.id.compareTo(b.id);
    });
    matches.sort((a, b) {
      final byTime = a.savedAt.compareTo(b.savedAt);
      return byTime != 0 ? byTime : a.id.compareTo(b.id);
    });
    return SessionTrends(trainingSessions: training, matchSessions: matches);
  }

  int get trainingCount => trainingSessions.length;

  /// How many stored sessions were matches.
  int get matchCount => matchSessions.length;

  /// Whether any match has been saved, so the cumulative match totals are worth
  /// surfacing.
  bool get hasMatchData => matchSessions.isNotEmpty;

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

  /// Total shots played across every saved training drill — a cumulative
  /// "career" practice-volume stat, the training-side twin of
  /// [totalMatchPoints]. 0 when no drill has been saved.
  int get totalShotsPracticed {
    var total = 0;
    for (final p in trainingSessions) {
      total += p.shotCount;
    }
    return total;
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

  /// Change in on-table accuracy (fraction of strokes kept on the table, [0,1],
  /// *higher-is-better*) from the first to the latest session that recorded it:
  /// returns `latest − first`, so a positive value means the player is missing
  /// the table less often. Null unless at least two sessions carry the metric.
  double? get accuracyImprovement {
    final series = _metricSeries((p) => p.onTableRate);
    if (series.length < 2) return null;
    return series.last - series.first;
  }

  /// Best on-table accuracy ([0,1]) recorded across every session that tracked
  /// it — the personal-best consistency number. Null if none did.
  double? get bestOnTableRate {
    double? best;
    for (final p in trainingSessions) {
      final r = p.onTableRate;
      if (r == null) continue;
      if (best == null || r > best) best = r;
    }
    return best;
  }

  /// Longest on-target (good-or-better) streak recorded across every session
  /// that tracked it — the personal-best "in a row" number, the streak analog of
  /// [bestOnTableRate]. Null if no session recorded a streak.
  int? get bestOnTargetStreak {
    int? best;
    for (final p in trainingSessions) {
      final s = p.longestOnTargetStreak;
      if (s == null) continue;
      if (best == null || s > best) best = s;
    }
    return best;
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

  /// Total points played across every saved match that recorded a point total —
  /// a cumulative "career" volume stat. Null if no match recorded one.
  int? get totalMatchPoints {
    var total = 0;
    var any = false;
    for (final m in matchSessions) {
      final p = m.totalPoints;
      if (p == null) continue;
      total += p;
      any = true;
    }
    return any ? total : null;
  }

  /// Total wall-clock play time (ms) summed across every saved match that
  /// recorded a duration — a cumulative "career" time-on-table stat, the
  /// match-side twin of [totalShotsPracticed]. Null if no match recorded one.
  int? get totalMatchDurationMs {
    var total = 0;
    var any = false;
    for (final m in matchSessions) {
      final d = m.durationMs;
      if (d == null) continue;
      total += d;
      any = true;
    }
    return any ? total : null;
  }

  /// Fastest ball speed (km/h) tracked across every saved match — the match-side
  /// personal-best radar number. Null if no match recorded a scaled speed.
  double? get fastestMatchBallSpeedKmh {
    double? best;
    for (final m in matchSessions) {
      final s = m.maxBallSpeedKmh;
      if (s == null) continue;
      if (best == null || s > best) best = s;
    }
    return best;
  }

  /// Longest rally (in strokes) tracked across every saved match. Null if no
  /// match recorded rally data.
  int? get longestMatchRallyStrokes {
    int? best;
    for (final m in matchSessions) {
      final s = m.longestRallyStrokes;
      if (s == null) continue;
      if (best == null || s > best) best = s;
    }
    return best;
  }

  /// Typical rally length (strokes) across saved matches — the unweighted mean
  /// of each match's own average rally, so it reads as "how long a rally usually
  /// runs" over the tracked history, complementing the peak
  /// [longestMatchRallyStrokes]. Null if no match recorded rally data.
  double? get averageMatchRallyStrokes {
    var total = 0.0;
    var n = 0;
    for (final m in matchSessions) {
      final a = m.averageRallyStrokes;
      if (a == null) continue;
      total += a;
      n++;
    }
    return n == 0 ? null : total / n;
  }

  /// How many saved matches finished with a recorded winner (an in-progress or
  /// pre-winner-field match contributes nothing to the head-to-head record).
  int get decidedMatchCount {
    var n = 0;
    for (final m in matchSessions) {
      if (m.winner == 'A' || m.winner == 'B') n++;
    }
    return n;
  }

  /// Whether at least one saved match finished, so the A-vs-B head-to-head
  /// record is worth surfacing.
  bool get hasMatchWinRecord => decidedMatchCount > 0;

  /// How many finished matches Player [key] (`A` / `B`) won across the saved
  /// history — the cumulative head-to-head record. The phone tracks a fixed
  /// left/right seat rather than a named person, so this is a seat-vs-seat
  /// tally, but for a recurring two-player pairing it is exactly the running
  /// series score they would keep by hand.
  int matchWinsBy(String key) {
    var n = 0;
    for (final m in matchSessions) {
      if (m.winner == key) n++;
    }
    return n;
  }

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
      _appendMatchSection(lines);
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

    lines.add('Total shots practiced: $totalShotsPracticed');

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

    final accuracyTrend = accuracyImprovement;
    if (accuracyTrend != null) {
      final verb = accuracyTrend > 0.0005
          ? 'up ${_signedPct(accuracyTrend)}'
          : (accuracyTrend < -0.0005
              ? 'down ${_signedPct(accuracyTrend)}'
              : 'flat');
      lines.add('On-table accuracy: $verb');
    }

    final bestStreak = bestOnTargetStreak;
    if (bestStreak != null && bestStreak >= 2) {
      lines.add('Best on-target streak: $bestStreak in a row');
    }

    if (hasRecurringFocus) {
      lines.add(
        'Recurring focus: $recurringFocus '
        '($recurringFocusCount of $trainingCount drills)',
      );
    }
    _appendMatchSection(lines);
    return lines.join('\n');
  }

  /// Append the cumulative match "career" totals, if any match is saved.
  void _appendMatchSection(List<String> lines) {
    if (!hasMatchData) return;
    lines.add('Matches: $matchCount played');
    if (hasMatchWinRecord) {
      lines.add(
        'Head-to-head: A ${matchWinsBy('A')}–${matchWinsBy('B')} B',
      );
    }
    final points = totalMatchPoints;
    if (points != null) lines.add('Points contested: $points');
    final playtime = totalMatchDurationMs;
    if (playtime != null) lines.add('Total play time: ${_fmtDuration(playtime)}');
    final rally = longestMatchRallyStrokes;
    if (rally != null) lines.add('Longest rally: $rally strokes');
    final avgRally = averageMatchRallyStrokes;
    if (avgRally != null) {
      lines.add('Average rally: ${avgRally.toStringAsFixed(1)} strokes');
    }
    final speed = fastestMatchBallSpeedKmh;
    if (speed != null) {
      lines.add('Fastest ball: ${speed.toStringAsFixed(1)} km/h');
    }
  }

  /// Format a millisecond duration as `Mm Ss` (e.g. `12m 03s`) for the
  /// cumulative career play-time line.
  static String _fmtDuration(int ms) {
    final totalSeconds = ms ~/ 1000;
    final minutes = totalSeconds ~/ 60;
    final seconds = totalSeconds % 60;
    return '${minutes}m ${seconds.toString().padLeft(2, '0')}s';
  }

  static String _pct(double v) => '${(v * 100).round()}%';

  static String _signedPct(double v) {
    final rounded = (v * 100).round();
    return '${rounded >= 0 ? '+' : ''}$rounded%';
  }
}

double? _asDouble(Object? v) => v is num ? v.toDouble() : null;

int? _asInt(Object? v) => v is num ? v.toInt() : null;
