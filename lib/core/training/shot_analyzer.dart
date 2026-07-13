/// Pure-Dart shot-quality analysis for **training mode**.
///
/// In training the player stands on one side of the table and drives balls
/// against a rebound net on the far side. There is no opponent and no score to
/// referee; instead the value is *feedback* — how fast, how deep and how
/// consistently the player is placing each stroke.
///
/// This layer reuses the same [BallTracker] the match pipeline is built on:
/// every outgoing stroke shows up as a net-cross from the player's side
/// followed by a [BounceEvent] on the target side. From that bounce we derive a
/// [Shot] (its target-side depth and the peak approach speed of the flight) and
/// grade it against a [TrainingConfig]. Aggregated across a session the shots
/// become a [TrainingSummary] describing pace, placement and consistency.
///
/// Like the rest of `core/`, it has no Flutter or vision-plugin dependencies,
/// so the whole thing is unit-testable against synthetic ball trajectories.
library;

import 'dart:math' as math;

import '../analysis/ball_speed.dart' show kTableLengthMeters;
import '../analysis/ball_tracker.dart';
import '../vision/detection.dart';

/// How good a single stroke was, bucketed from its [Shot.score].
enum ShotGrade { poor, fair, good, excellent }

/// One graded stroke: an outgoing ball that bounced on the target side.
class Shot {
  const Shot({
    required this.timestampMs,
    required this.speed,
    required this.depth,
    required this.score,
    this.lateral = 0.5,
    this.speedKmh = 0,
  });

  /// Time of the target-side bounce that completed the stroke.
  final int timestampMs;

  /// Peak horizontal approach speed of the flight, in normalized units/second.
  final double speed;

  /// Peak horizontal approach speed scaled to real-world **km/h** via the
  /// calibrated table ruler (the ITTF 2.74 m length spans the frame x-axis from
  /// a side camera), the physical companion to the normalized [speed]. Unlike
  /// [speed] — which saturates against an arbitrary reference — this is a
  /// meaningful radar-gun-style pace a player can read. Defaults to `0` (no
  /// scale known).
  final double speedKmh;

  /// Where the ball landed on the target half: `0` at the net, `1` at the far
  /// baseline. Clamped to `[0, 1]`.
  final double depth;

  /// Where the ball landed across the table's near/far depth: `0` at the
  /// surface's near (top) edge, `1` at its far (bottom) edge. Clamped to
  /// `[0, 1]`. Together with [depth] this locates the bounce on the target half
  /// for the training shot-map. Defaults to the table's centre (`0.5`).
  final double lateral;

  /// Combined quality in `[0, 1]` (placement accuracy blended with pace).
  final double score;

  /// The score bucketed into a coarse grade.
  ShotGrade get grade {
    if (score >= 0.8) return ShotGrade.excellent;
    if (score >= 0.6) return ShotGrade.good;
    if (score >= 0.4) return ShotGrade.fair;
    return ShotGrade.poor;
  }

  @override
  String toString() =>
      'Shot(@$timestampMs, depth=${depth.toStringAsFixed(2)}, '
      'speed=${speed.toStringAsFixed(2)}, ${grade.name})';
}

/// Tunable definition of what a "good" training shot looks like.
class TrainingConfig {
  const TrainingConfig({
    this.geometry = const TableGeometry(),
    this.playerSide = TableSide.left,
    this.targetDepth = 0.75,
    this.depthTolerance = 0.35,
    this.referenceSpeed = 1.5,
    this.placementWeight = 0.6,
    this.tableLengthMeters = kTableLengthMeters,
    this.targetSpeedKmh = 30,
  })  : assert(targetDepth >= 0 && targetDepth <= 1),
        assert(depthTolerance > 0),
        assert(referenceSpeed > 0),
        assert(placementWeight >= 0 && placementWeight <= 1),
        assert(tableLengthMeters > 0),
        assert(targetSpeedKmh > 0);

  /// The table layout (where the net sits) within the normalized frame.
  final TableGeometry geometry;

  /// The half the player hits *from*; the rebound net is on the other half.
  final TableSide playerSide;

  /// Desired landing depth on the target half (`0` net .. `1` baseline).
  final double targetDepth;

  /// Depth error at which placement quality falls to zero.
  final double depthTolerance;

  /// Approach speed (units/second) that earns full marks for pace.
  final double referenceSpeed;

  /// Blend of placement vs pace in the final score (`1` = placement only).
  final double placementWeight;

  /// Physical table length (metres) the frame x-span maps to, the ruler for
  /// scaling a shot's along-table pace to real-world km/h (default ITTF 2.74 m).
  final double tableLengthMeters;

  /// Along-table shot pace (km/h) that earns full marks for the coaching
  /// "Shot pace" dimension — the physical-units power target a driving drill
  /// aims to reach (default ~30 km/h for a solid recreational drive). Only used
  /// by [TrainingFeedback] and only when a physical km/h scale is available.
  final double targetSpeedKmh;

  /// The half the ball should land on (opposite the player).
  TableSide get targetSide => playerSide.other;

  /// Returns a copy with the given fields replaced. Used by the live training
  /// screen to switch [playerSide] from the pre-session picker before any shot
  /// is graded.
  TrainingConfig copyWith({
    TableGeometry? geometry,
    TableSide? playerSide,
    double? targetDepth,
    double? depthTolerance,
    double? referenceSpeed,
    double? placementWeight,
    double? tableLengthMeters,
    double? targetSpeedKmh,
  }) {
    return TrainingConfig(
      geometry: geometry ?? this.geometry,
      playerSide: playerSide ?? this.playerSide,
      targetDepth: targetDepth ?? this.targetDepth,
      depthTolerance: depthTolerance ?? this.depthTolerance,
      referenceSpeed: referenceSpeed ?? this.referenceSpeed,
      placementWeight: placementWeight ?? this.placementWeight,
      tableLengthMeters: tableLengthMeters ?? this.tableLengthMeters,
      targetSpeedKmh: targetSpeedKmh ?? this.targetSpeedKmh,
    );
  }

  /// Metres each normalized x-unit represents, given the table's frame x-span —
  /// the along-table ruler for [Shot.speedKmh].
  double get metersPerUnitX => tableLengthMeters / (geometry.right - geometry.left);
}

/// Aggregated feedback over a training session's [Shot]s.
class TrainingSummary {
  const TrainingSummary(this.shots, {this.missedShots = 0})
      : assert(missedShots >= 0);

  final List<Shot> shots;

  /// Outgoing strokes that crossed the net but never landed on the target half
  /// — the ball went off the table (long/wide) instead of bouncing in. These
  /// are *not* [Shot]s (they have no landing to grade), but they are attempts,
  /// so they set the denominator for [onTableRate]. Defaults to `0`.
  final int missedShots;

  int get shotCount => shots.length;

  /// Every outgoing stroke attempted this session: the ones that landed on the
  /// table ([shotCount]) plus the ones that missed it ([missedShots]).
  int get attemptedShots => shotCount + missedShots;

  /// Fraction of attempted strokes that actually landed on the target half, in
  /// `[0, 1]` — the headline "in %" a coach watches. `0` when nothing was
  /// attempted. A player grouping every ball on the table scores `1`.
  double get onTableRate =>
      attemptedShots == 0 ? 0 : shotCount / attemptedShots;

  double get averageSpeed => _mean(shots.map((s) => s.speed));

  /// Mean real-world peak pace across the session, in km/h (0 when empty).
  double get averageSpeedKmh => _mean(shots.map((s) => s.speedKmh));

  /// The fastest single shot in the session, in km/h (0 when empty).
  double get maxSpeedKmh =>
      shots.isEmpty ? 0 : shots.map((s) => s.speedKmh).reduce(math.max);

  double get averageDepth => _mean(shots.map((s) => s.depth));

  double get averageLateral => _mean(shots.map((s) => s.lateral));

  double get averageScore => _mean(shots.map((s) => s.score));

  /// Number of shots that earned [grade].
  int gradeCount(ShotGrade grade) =>
      shots.where((s) => s.grade == grade).length;

  /// Whether a stroke counts as "on target" for streak purposes: it landed with
  /// at least a [ShotGrade.good] — a well-placed drive — mirroring how the match
  /// summary counts a run of points *won*.
  static bool _onTarget(Shot s) => s.grade.index >= ShotGrade.good.index;

  /// The longest run of consecutive on-target ([ShotGrade.good] or better) shots
  /// in the session — the headline "in a row" streak a coach or gamified drill
  /// tracks, the training analog of `MatchSummary.longestStreakFor`. A single
  /// off-target stroke resets the count. `0` when empty or nothing landed well.
  int get longestOnTargetStreak {
    var longest = 0;
    var current = 0;
    for (final s in shots) {
      if (_onTarget(s)) {
        current += 1;
        if (current > longest) longest = current;
      } else {
        current = 0;
      }
    }
    return longest;
  }

  /// The run of on-target shots still "alive" at the end of the session — the
  /// trailing streak, for a live "N in a row" readout. `0` when the last stroke
  /// missed the [ShotGrade.good] bar.
  int get currentOnTargetStreak {
    var current = 0;
    for (final s in shots.reversed) {
      if (!_onTarget(s)) break;
      current += 1;
    }
    return current;
  }

  /// Minimum shots before an intra-session trend is meaningful: each half needs
  /// at least two shots to average.
  static const int _trendMinShots = 4;

  /// Whether the session has enough shots ([_trendMinShots]) to split into two
  /// halves and measure a within-session trend.
  bool get hasScoreTrend => shots.length >= _trendMinShots;

  /// Average shot quality over the *first* half of the session's shots, in
  /// `[0, 1]`. `0` until [hasScoreTrend]. For an odd shot count the middle shot
  /// is excluded so the two halves stay equal-sized.
  double get firstHalfAverageScore {
    if (!hasScoreTrend) return 0;
    final half = shots.length ~/ 2;
    return _mean(shots.take(half).map((s) => s.score));
  }

  /// Average shot quality over the *second* half of the session's shots, in
  /// `[0, 1]`. `0` until [hasScoreTrend].
  double get secondHalfAverageScore {
    if (!hasScoreTrend) return 0;
    final half = shots.length ~/ 2;
    return _mean(shots.skip(shots.length - half).map((s) => s.score));
  }

  /// The change in average shot quality from the first half of the session to
  /// the second, in score points (`[-1, 1]`). Positive = the player warmed up
  /// and improved as the drill went on; negative = quality faded through the
  /// session (a fatigue / concentration-drop signal a coach watches for).
  /// `null` until [hasScoreTrend]. This is the *within-session* analog of
  /// SessionTrends' cross-session improvement — every other metric here is a
  /// whole-session aggregate that hides whether the player rose or faded.
  double? get scoreTrend =>
      hasScoreTrend ? secondHalfAverageScore - firstHalfAverageScore : null;

  /// How repeatable the placement was, in `[0, 1]`: `1` means every ball landed
  /// at the same depth, `0` means depths were spread across the whole half.
  /// Derived from the population standard deviation of shot depth.
  double get consistency {
    if (shots.length < 2) return shots.isEmpty ? 0 : 1;
    final mean = averageDepth;
    final variance =
        _mean(shots.map((s) => math.pow(s.depth - mean, 2).toDouble()));
    final std = math.sqrt(variance);
    // A std of 0.5 spans an entire table half; treat that as fully inconsistent.
    return (1 - std / 0.5).clamp(0.0, 1.0);
  }

  /// How repeatable the *across-table* (lateral) placement was, in `[0, 1]`:
  /// `1` means every ball landed at the same lateral position, `0` means they
  /// were spread across the whole width. Derived from the population standard
  /// deviation of [Shot.lateral], the companion to [consistency] for the depth
  /// axis — together they say whether the player is grouping shots into a spot.
  double get lateralConsistency {
    if (shots.length < 2) return shots.isEmpty ? 0 : 1;
    final mean = averageLateral;
    final variance =
        _mean(shots.map((s) => math.pow(s.lateral - mean, 2).toDouble()));
    final std = math.sqrt(variance);
    // A std of 0.5 spans the whole width; treat that as fully inconsistent.
    return (1 - std / 0.5).clamp(0.0, 1.0);
  }

  /// Wall-clock span between the first and last shot, in ms.
  int get durationMs =>
      shots.length < 2 ? 0 : shots.last.timestampMs - shots.first.timestampMs;

  /// The gaps (ms) between consecutive shots, in order. Empty for `<2` shots.
  List<int> get shotIntervalsMs {
    if (shots.length < 2) return const [];
    return [
      for (var i = 1; i < shots.length; i++)
        shots[i].timestampMs - shots[i - 1].timestampMs,
    ];
  }

  /// Mean time between consecutive shots, in ms. `0` for `<2` shots.
  double get averageIntervalMs {
    final gaps = shotIntervalsMs;
    if (gaps.isEmpty) return 0;
    return _mean(gaps.map((g) => g.toDouble()));
  }

  /// Drill cadence: how many shots the player produced per minute at the
  /// observed tempo. `0` until at least two shots establish an interval.
  double get shotsPerMinute {
    final avg = averageIntervalMs;
    if (avg <= 0) return 0;
    return 60000 / avg;
  }

  /// How metronomic the drill tempo was, in `[0, 1]`: `1` means every gap
  /// between shots was identical (a perfectly steady rhythm), `0` means the
  /// gaps were wildly irregular. Derived from the coefficient of variation
  /// (stddev / mean) of the inter-shot intervals — the tempo companion to the
  /// depth/lateral placement [consistency] metrics.
  double get rhythmConsistency {
    final gaps = shotIntervalsMs;
    if (gaps.length < 2) return gaps.isEmpty ? 0 : 1;
    final mean = averageIntervalMs;
    if (mean <= 0) return 0;
    final variance =
        _mean(gaps.map((g) => math.pow(g - mean, 2).toDouble()));
    final std = math.sqrt(variance);
    return (1 - std / mean).clamp(0.0, 1.0);
  }

  /// An A–F letter grade for the whole session, from [averageScore].
  String get overallGrade {
    if (shots.isEmpty) return '–';
    final s = averageScore;
    if (s >= 0.85) return 'A';
    if (s >= 0.7) return 'B';
    if (s >= 0.55) return 'C';
    if (s >= 0.4) return 'D';
    return 'F';
  }

  /// A short human phrase for a first→second-half score [trend] (score points).
  static String _trendLabel(double trend) {
    final pts = (trend * 100).round();
    if (pts >= 4) return 'warming up (+$pts% quality through the drill)';
    if (pts <= -4) return 'fading ($pts% — watch for fatigue)';
    return 'steady';
  }

  static double _mean(Iterable<double> xs) {
    final list = xs.toList();
    if (list.isEmpty) return 0;
    return list.reduce((a, b) => a + b) / list.length;
  }

  /// A deterministic, human-readable training report.
  String report() {
    if (shots.isEmpty) {
      return 'Training summary\nNo shots recorded yet.';
    }
    final pct = (averageScore * 100).round();
    return [
      'Training summary',
      '$shotCount shots — grade $overallGrade ($pct%).',
      if (missedShots > 0)
        'On-table accuracy: ${(onTableRate * 100).round()}% '
            '($shotCount of $attemptedShots on the table).',
      'Avg depth: ${(averageDepth * 100).round()}% of the far half.',
      'Avg pace: ${averageSpeed.toStringAsFixed(2)} units/s.',
      if (maxSpeedKmh > 0)
        'Ball speed: ${maxSpeedKmh.toStringAsFixed(0)} km/h top, '
            '${averageSpeedKmh.toStringAsFixed(0)} km/h avg.',
      'Depth consistency: ${(consistency * 100).round()}%.',
      'Lateral consistency: ${(lateralConsistency * 100).round()}%.',
      if (shots.length >= 2) ...[
        'Tempo: ${shotsPerMinute.toStringAsFixed(1)} shots/min.',
        'Rhythm consistency: ${(rhythmConsistency * 100).round()}%.',
      ],
      if (longestOnTargetStreak >= 2)
        'Best on-target streak: $longestOnTargetStreak in a row.',
      if (scoreTrend != null) 'Session trend: ${_trendLabel(scoreTrend!)}.',
      '  • ${gradeCount(ShotGrade.excellent)} excellent',
      '  • ${gradeCount(ShotGrade.good)} good',
      '  • ${gradeCount(ShotGrade.fair)} fair',
      '  • ${gradeCount(ShotGrade.poor)} poor',
    ].join('\n');
  }
}

/// Incrementally consumes ball detections and emits graded [Shot]s.
///
/// Feed it one [FrameResult] per frame via [onFrame]; it returns the [Shot] that
/// completed on that frame, or `null`. Internally it runs a [BallTracker] to
/// spot net-crosses and target-side bounces, and measures the peak horizontal
/// speed of each outgoing flight for pace scoring.
class ShotAnalyzer {
  ShotAnalyzer({TrainingConfig config = const TrainingConfig()})
      : config = config,
        _tracker = BallTracker(geometry: config.geometry);

  final TrainingConfig config;
  final BallTracker _tracker;

  /// The internal ball tracker, exposed so a live overlay can draw the
  /// Kalman-predicted "ghost" ball ([BallTracker.estimateBallAt]) through
  /// detector dropouts — the same seam the match controller's tracker exposes.
  BallTracker get tracker => _tracker;

  final List<Shot> _shots = [];
  int _misses = 0;

  BallSample? _prev;
  bool _outgoing = false;
  double _peakSpeed = 0;
  double? _lastSpeedKmh;

  /// All shots recorded so far, in order.
  List<Shot> get shots => List.unmodifiable(_shots);

  /// Outgoing strokes this session that crossed the net but never landed on the
  /// target half — i.e. the ball went off the table without bouncing in. Fed
  /// into [TrainingSummary.missedShots] so on-table accuracy can be reported.
  int get missCount => _misses;

  /// The most recent per-frame ball speed, scaled to real-world **km/h** via the
  /// table ruler — the live "radar gun" reading a training overlay can flash
  /// beside the ball as it flies, the practice-mode companion to the whole-shot
  /// [Shot.speedKmh]. `null` before any two-frame reading exists and after a
  /// [BallLostEvent] clears the trajectory (so a detector dropout doesn't leave a
  /// stale number). Callers should still only display it while the ball is in
  /// view, mirroring the live match screen's readout gating.
  double? get currentSpeedKmh => _lastSpeedKmh;

  /// A live summary over the shots recorded so far.
  TrainingSummary get summary =>
      TrainingSummary(List.of(_shots), missedShots: _misses);

  /// Feed one frame; returns the shot completed on it, if any.
  Shot? onFrame(FrameResult frame) {
    final events = _tracker.update(frame);

    // Track our own peak horizontal speed for the current outgoing flight.
    final ball = frame.ball;
    final prev = _prev;
    double? frameSpeed;
    if (ball != null) {
      final sample = BallSample(
        frame.timestampMs,
        ball.box.centerX,
        ball.box.centerY,
      );
      if (prev != null && sample.timestampMs > prev.timestampMs) {
        final dt = (sample.timestampMs - prev.timestampMs) / 1000.0;
        frameSpeed = (sample.x - prev.x).abs() / dt;
        _lastSpeedKmh = frameSpeed * config.metersPerUnitX * 3.6;
      }
      _prev = sample;
    }

    Shot? completed;
    for (final event in events) {
      if (event is NetCrossEvent && event.to == config.targetSide) {
        // A fresh outgoing stroke begins; restart the pace measurement so a
        // rebound return doesn't leak into the next shot's speed.
        _outgoing = true;
        _peakSpeed = 0;
      } else if (event is BallLostEvent) {
        // An outgoing stroke that was still in flight when the ball vanished
        // crossed the net but never landed on the target half — it went off
        // the table. Count it as a missed attempt (it is not a gradable shot).
        if (_outgoing) _misses++;
        _outgoing = false;
        _peakSpeed = 0;
        _lastSpeedKmh = null;
      } else if (event is BounceEvent &&
          event.side == config.targetSide &&
          _outgoing) {
        completed = _recordShot(event);
      }
    }

    if (_outgoing && frameSpeed != null && frameSpeed > _peakSpeed) {
      _peakSpeed = frameSpeed;
    }

    return completed;
  }

  Shot _recordShot(BounceEvent bounce) {
    final depth = _depthOf(bounce.x);
    final lateral = _lateralOf(bounce.y);
    final placement =
        (1 - (depth - config.targetDepth).abs() / config.depthTolerance)
            .clamp(0.0, 1.0);
    final pace = (_peakSpeed / config.referenceSpeed).clamp(0.0, 1.0);
    final score =
        config.placementWeight * placement + (1 - config.placementWeight) * pace;

    final shot = Shot(
      timestampMs: bounce.timestampMs,
      speed: _peakSpeed,
      speedKmh: _peakSpeed * config.metersPerUnitX * 3.6,
      depth: depth,
      lateral: lateral,
      score: score,
    );
    _shots.add(shot);
    _outgoing = false;
    _peakSpeed = 0;
    return shot;
  }

  /// Normalized landing depth on the target half: `0` at the net, `1` at the
  /// far baseline.
  double _depthOf(double x) {
    final netX = config.geometry.netX;
    final d = config.targetSide == TableSide.right
        ? (x - netX) / (1 - netX)
        : (netX - x) / netX;
    return d.clamp(0.0, 1.0);
  }

  /// Normalized lateral landing position across the table's near/far depth:
  /// `0` at the surface's near (top) edge, `1` at its far (bottom) edge.
  double _lateralOf(double y) {
    final geo = config.geometry;
    final span = geo.bottom - geo.top;
    if (span <= 0) return 0.5;
    return ((y - geo.top) / span).clamp(0.0, 1.0);
  }

  /// Forget all session state (e.g. to start a new drill).
  void reset() {
    _tracker.reset();
    _shots.clear();
    _misses = 0;
    _prev = null;
    _outgoing = false;
    _peakSpeed = 0;
    _lastSpeedKmh = null;
  }
}
