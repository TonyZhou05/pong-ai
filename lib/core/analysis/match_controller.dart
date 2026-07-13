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

class MatchController {
  MatchController({
    BallTracker? tracker,
    RallyReferee? referee,
    ScoringEngine? engine,
  })  : tracker = tracker ?? BallTracker(),
        referee = referee ?? RallyReferee(),
        engine = engine ?? ScoringEngine();

  final BallTracker tracker;
  final RallyReferee referee;
  final ScoringEngine engine;

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
    final decisions = <PointDecision>[];
    for (final event in tracker.update(frame)) {
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
      tracker.reset();
    }
    return decisions;
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
