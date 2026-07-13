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
import 'ball_speed.dart';
import 'ball_tracker.dart';
import 'bounce_placement.dart';
import 'match_summary.dart';
import 'player_movement.dart';
import 'rally_analyzer.dart';
import 'rally_referee.dart';
import 'table_calibrator.dart';
import 'tracking_quality.dart';

class MatchController {
  MatchController({
    BallTracker? tracker,
    RallyReferee? referee,
    ScoringEngine? engine,
    this.calibrator,
  })  : _tracker = tracker ?? BallTracker(),
        referee = referee ?? RallyReferee(),
        engine = engine ?? ScoringEngine() {
    _movement = PlayerMovementAnalyzer(
      geometry: _tracker.geometry,
      leftPlayer: this.referee.leftPlayer,
    );
    _placement = BouncePlacementAnalyzer(geometry: _tracker.geometry);
    _ballSpeed = BallSpeedEstimator(geometry: _tracker.geometry);
  }

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

  /// Player movement / footwork analytics accumulated from the pose model over
  /// the (post-calibration) frames scored so far. Rebuilt on the calibrated
  /// geometry so its net-split side assignment matches the referee.
  late PlayerMovementAnalyzer _movement;

  /// Footwork / positioning metrics for [player] over the match so far.
  PlayerMovementStats movementFor(Player player) => _movement.statsFor(player);

  /// Every recorded foot-position sample for [player] over the match so far —
  /// the raw material for a positioning / court-coverage heatmap.
  List<FramePoint> positionsFor(Player player) => _movement.positionsFor(player);

  /// Rally-length analytics (strokes/duration per point) over the match so far.
  /// Like the movement analytics it is live-only — it is not rewound by [undo].
  final RallyAnalyzer _rallies = RallyAnalyzer();

  /// Aggregate rally-length statistics accumulated so far.
  RallyStats get rallyStats => _rallies.stats;

  /// Bounce-placement / shot-map analytics accumulated from the tracker's
  /// bounce events. Rebuilt on the calibrated geometry so net/edge-relative
  /// placement lines up with the inferred table. Live-only like the movement and
  /// rally analytics — not rewound by [undo].
  late BouncePlacementAnalyzer _placement;

  /// Placement distribution of every bounce recorded on [side] so far.
  SidePlacementStats placementFor(TableSide side) =>
      _placement.statsFor(side);

  /// Real-world ball-speed analytics estimated from the ball's along-table
  /// motion against the calibrated geometry. Rebuilt on the calibrated geometry
  /// so the metres-per-unit ruler matches the inferred table. Live-only like the
  /// movement/rally/placement analytics — not rewound by [undo].
  late BallSpeedEstimator _ballSpeed;

  /// The fastest ball speed (km/h) observed so far, or 0 when no data.
  double get maxBallSpeedKmh => _ballSpeed.maxKmh;

  /// The mean observed ball speed (km/h), or 0 when no data.
  double get averageBallSpeedKmh => _ballSpeed.averageKmh;

  /// Whether any ball-speed reading has been accumulated.
  bool get hasBallSpeedData => _ballSpeed.hasData;

  /// Tracking-quality / detection-health analytics accumulated over *every*
  /// frame (including calibration warm-up — detection health is independent of
  /// scoring). Unlike the geometry-dependent analytics it is never rebuilt on
  /// calibration, so it captures how well the phone placement tracked the whole
  /// session. Live-only like the movement/rally/placement analytics.
  final TrackingQualityAnalyzer _tracking = TrackingQualityAnalyzer();

  /// Detection-health metrics for the session so far — how reliably the ball
  /// and both players were tracked, for a phone-placement quality read-out.
  TrackingQualityAnalyzer get trackingQuality => _tracking;

  void _record(
    Player winner,
    PointReason reason,
    int timestampMs,
    Player server,
    int gameIndex,
  ) {
    _points.add(
      ScoredPoint(
        winner: winner,
        reason: reason,
        timestampMs: timestampMs,
        server: server,
        gameIndex: gameIndex,
      ),
    );
  }

  /// Feed one vision frame through the pipeline.
  ///
  /// Returns every [PointDecision] reached on this frame (usually none, and at
  /// most one per rally-ending event). Decisive decisions are applied to the
  /// [engine] automatically; undetermined ones are collected in [undetermined].
  List<PointDecision> onFrame(FrameResult frame) {
    // Detection-health accounting runs on every frame, including calibration
    // warm-up, since it measures how well the phone placement tracks the ball
    // and players regardless of whether we are scoring yet.
    _tracking.observe(frame);

    // Warm-up: accumulate observations and defer all scoring until the geometry
    // is trusted. Once it is, rebuild the tracker and score from here onward.
    final cal = calibrator;
    if (cal != null && !_calibrated) {
      cal.observe(frame);
      final geometry = cal.calibrate();
      if (geometry == null) return const [];
      _applyCalibration(geometry);
    }

    // Mine this frame's player poses for footwork/positioning analytics, and
    // its ball position for real-world speed. Runs only once scoring is live
    // (past any calibration warm-up), so the geometry used to attribute players
    // to sides and to scale ball speed to metres is the calibrated one.
    _movement.observe(frame);
    _ballSpeed.observe(frame);

    final decisions = <PointDecision>[];
    for (final event in _tracker.update(frame)) {
      _rallies.observe(event);
      _placement.observe(event);
      final decision = referee.update(event);
      if (decision == null) continue;

      if (decision.isDecisive) {
        // Capture who served *before* awarding — awardPoint advances the serve
        // rotation, so state.server after the call is the next server, not this
        // rally's. Likewise the game index is the number of games completed
        // before this point (awarding may complete the game).
        final server = engine.state.server;
        final gameIndex = engine.state.gamesA + engine.state.gamesB;
        engine.awardPoint(decision.winner!);
        _record(
          decision.winner!,
          decision.reason,
          decision.timestampMs,
          server,
          gameIndex,
        );
      } else {
        _undetermined.add(decision);
      }
      decisions.add(decision);
      _rallies.endRally(decision);

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
      maxJump: _tracker.maxJump,
    );
    // Rebuild movement analytics on the calibrated net line so player-to-side
    // attribution matches the now-inferred geometry (nothing was scored during
    // warm-up, so no movement is lost).
    _movement = PlayerMovementAnalyzer(
      geometry: geometry,
      leftPlayer: referee.leftPlayer,
    );
    // Rebuild placement analytics on the calibrated net/edges so bounce
    // depth-from-net and lateral coordinates are measured against the inferred
    // table (nothing was scored during warm-up, so no bounces are lost).
    _placement = BouncePlacementAnalyzer(geometry: geometry);
    // Rebuild the speed estimator on the calibrated table span so its
    // metres-per-unit ruler reflects the inferred table width in the frame.
    _ballSpeed = BallSpeedEstimator(geometry: geometry);
    _calibrated = true;
  }

  /// Manually award an [undetermined] point the referee could not attribute
  /// (e.g. a ball smashed out of play). Removes it from [undetermined] and
  /// applies it to the score. No-op if [decision] is not pending.
  void resolveUndetermined(PointDecision decision, Player winner) {
    if (_undetermined.remove(decision)) {
      final server = engine.state.server;
      final gameIndex = engine.state.gamesA + engine.state.gamesB;
      engine.awardPoint(winner);
      _record(winner, decision.reason, decision.timestampMs, server, gameIndex);
    }
  }

  /// Undo the most recent scored point. Returns true if something was undone.
  bool undo() {
    final undone = engine.undo();
    if (undone && _points.isNotEmpty) _points.removeLast();
    return undone;
  }
}
