/// The vertical slice that turns a stream of vision frames into a live score.
///
/// [MatchController] wires the three previously-isolated pure-Dart layers into
/// one pipeline:
///
/// ```
/// FrameResult ──▶ BallTracker ──▶ RallyReferee ──▶ ScoringEngine
///  (vision)        (events)        (decisions)       (the score)
/// ```
///
/// It has no Flutter or plugin dependencies: feed it [FrameResult]s (from the
/// live `ultralytics_yolo` runtime or a replayed benchmark clip) and read
/// [score]. This makes the whole match loop unit-testable end-to-end without a
/// camera.
library;

import '../scoring/scoring_engine.dart';
import '../vision/detection.dart';
import 'ball_tracker.dart';
import 'match_summary.dart';
import 'rally_referee.dart';
import 'table_calibrator.dart';

class MatchController {
  MatchController({
    BallTracker? tracker,
    RallyReferee? referee,
    ScoringEngine? engine,
    this.calibrator,
  })  : _tracker = tracker ?? BallTracker(),
        referee = referee ?? RallyReferee(),
        engine = engine ?? ScoringEngine();

  BallTracker _tracker;

  /// The trajectory tracker. When an auto-[calibrator] is supplied it is
  /// rebuilt (with the same tuning) once calibration completes, so this getter
  /// always reflects the geometry currently in force.
  BallTracker get tracker => _tracker;

  final RallyReferee referee;
  final ScoringEngine engine;

  /// Optional auto-calibrator. When provided, the controller spends a warm-up
  /// phase feeding frames to it (scoring nothing) until it can infer the
  /// [TableGeometry] from where the ball and players actually are; only then
  /// does it rebuild [tracker] with that geometry and start scoring. This lets
  /// the user just place the phone table-side instead of hand-marking corners.
  final TableCalibrator? calibrator;

  bool _calibrated = false;

  /// Whether the controller is still in the calibration warm-up (no points are
  /// scored yet). Always false when no [calibrator] was supplied.
  bool get isCalibrating => calibrator != null && !_calibrated;

  /// The table geometry currently driving rally detection.
  TableGeometry get geometry => _tracker.geometry;

  /// Decisions the referee could not attribute a winner to
  /// ([PointReason.outOfPlay]); the UI can surface these for the user to
  /// resolve manually.
  final List<PointDecision> _undetermined = [];
  List<PointDecision> get undetermined => List.unmodifiable(_undetermined);

  /// The ordered log of every point actually awarded, in [engine.awardPoint]
  /// order. Kept in sync with the score (including [undo]) so it can drive the
  /// post-match [summary].
  final List<ScoredPoint> _points = [];
  List<ScoredPoint> get points => List.unmodifiable(_points);

  /// The current match score.
  MatchState get score => engine.state;

  /// Performance analysis over the points scored so far.
  MatchSummary get summary =>
      MatchSummary(points: points, finalState: engine.state);

  void _record(Player winner, PointReason reason, int timestampMs) {
    _points.add(
      ScoredPoint(winner: winner, reason: reason, timestampMs: timestampMs),
    );
  }

  /// Feed one vision frame through the pipeline.
  ///
  /// Returns every [PointDecision] reached on this frame (usually none, and at
  /// most one per rally-ending event). Decisive decisions are applied to the
  /// [engine] automatically; undetermined ones are collected in [undetermined].
  List<PointDecision> onFrame(FrameResult frame) {
    // Warm-up: accumulate observations and defer all scoring until the geometry
    // is trusted. Once it is, rebuild the tracker and score from here onward.
    final cal = calibrator;
    if (cal != null && !_calibrated) {
      cal.observe(frame);
      final geometry = cal.calibrate();
      if (geometry == null) return const [];
      _applyCalibration(geometry);
    }

    final decisions = <PointDecision>[];
    for (final event in _tracker.update(frame)) {
      final decision = referee.update(event);
      if (decision == null) continue;

      if (decision.isDecisive) {
        engine.awardPoint(decision.winner!);
        _record(decision.winner!, decision.reason, decision.timestampMs);
      } else {
        _undetermined.add(decision);
      }
      decisions.add(decision);

      // A rally just ended; start the next one from a clean trajectory.
      _tracker.reset();
    }
    return decisions;
  }

  /// Swap in a tracker built on the calibrated [geometry], preserving the
  /// original tracker's tuning, and mark calibration complete.
  void _applyCalibration(TableGeometry geometry) {
    _tracker = BallTracker(
      geometry: geometry,
      minBounceSpeed: _tracker.minBounceSpeed,
      maxGapFrames: _tracker.maxGapFrames,
    );
    _calibrated = true;
  }

  /// Manually award an [undetermined] point the referee could not attribute
  /// (e.g. a ball smashed out of play). Removes it from [undetermined] and
  /// applies it to the score. No-op if [decision] is not pending.
  void resolveUndetermined(PointDecision decision, Player winner) {
    if (_undetermined.remove(decision)) {
      engine.awardPoint(winner);
      _record(winner, decision.reason, decision.timestampMs);
    }
  }

  /// Undo the most recent scored point. Returns true if something was undone.
  bool undo() {
    final undone = engine.undo();
    if (undone && _points.isNotEmpty) _points.removeLast();
    return undone;
  }
}
