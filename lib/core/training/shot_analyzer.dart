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
  });

  /// Time of the target-side bounce that completed the stroke.
  final int timestampMs;

  /// Peak horizontal approach speed of the flight, in normalized units/second.
  final double speed;

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
  })  : assert(targetDepth >= 0 && targetDepth <= 1),
        assert(depthTolerance > 0),
        assert(referenceSpeed > 0),
        assert(placementWeight >= 0 && placementWeight <= 1);

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

  /// The half the ball should land on (opposite the player).
  TableSide get targetSide => playerSide.other;
}

/// Aggregated feedback over a training session's [Shot]s.
class TrainingSummary {
  const TrainingSummary(this.shots);

  final List<Shot> shots;

  int get shotCount => shots.length;

  double get averageSpeed => _mean(shots.map((s) => s.speed));

  double get averageDepth => _mean(shots.map((s) => s.depth));

  double get averageScore => _mean(shots.map((s) => s.score));

  /// Number of shots that earned [grade].
  int gradeCount(ShotGrade grade) =>
      shots.where((s) => s.grade == grade).length;

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

  /// Wall-clock span between the first and last shot, in ms.
  int get durationMs =>
      shots.length < 2 ? 0 : shots.last.timestampMs - shots.first.timestampMs;

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
      'Avg depth: ${(averageDepth * 100).round()}% of the far half.',
      'Avg pace: ${averageSpeed.toStringAsFixed(2)} units/s.',
      'Consistency: ${(consistency * 100).round()}%.',
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

  final List<Shot> _shots = [];

  BallSample? _prev;
  bool _outgoing = false;
  double _peakSpeed = 0;

  /// All shots recorded so far, in order.
  List<Shot> get shots => List.unmodifiable(_shots);

  /// A live summary over the shots recorded so far.
  TrainingSummary get summary => TrainingSummary(List.of(_shots));

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
        _outgoing = false;
        _peakSpeed = 0;
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
    _prev = null;
    _outgoing = false;
    _peakSpeed = 0;
  }
}
