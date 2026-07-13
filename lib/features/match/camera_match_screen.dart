import 'dart:async';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../../core/analysis/ball_tracker.dart';
import '../../core/analysis/match_controller.dart';
import '../../core/analysis/rally_referee.dart';
import '../../core/analysis/table_calibrator.dart';
import '../../core/scoring/scoring_engine.dart';
import '../../core/vision/detection.dart';
import '../../core/vision/yolo_vision_service.dart';

/// The **live camera** match screen: runs on-device `ultralytics_yolo`
/// inference over the phone's camera and drives the real
/// tracker → referee → scoring pipeline from it.
///
/// This is the production counterpart to [MatchScreen] (which replays scripted
/// frames): here a [YOLOView] platform view produces detections, its
/// `onStreamingData` callback feeds a [YoloVisionService], and that service's
/// [FrameResult] stream drives a [MatchController]. The camera preview fills the
/// screen and the scoreboard/referee calls are overlaid on top — the "Ball AI"
/// live view.
///
/// Placed table-side, the [MatchController] first self-calibrates the table
/// geometry from a warm-up window (via [TableCalibrator]) before it starts
/// scoring, so the user just props up the phone and plays.
///
/// The camera preview is injected via [cameraPreviewBuilder] and the frame
/// source via [visionService] so widget tests can drive the pipeline headlessly
/// without a platform view or camera.
class CameraMatchScreen extends StatefulWidget {
  const CameraMatchScreen({
    super.key,
    this.visionService,
    this.cameraPreviewBuilder,
    this.matchControllerBuilder,
    this.modelPath = 'yolo11n',
    this.task = YOLOTask.detect,
  });

  /// The camera-backed frame source. Defaults to a fresh [YoloVisionService].
  final YoloVisionService? visionService;

  /// Builds the camera preview widget. Defaults to a real [YOLOView] wired to
  /// [visionService]. Injectable so tests can substitute a headless stand-in.
  final Widget Function(BuildContext, YoloVisionService)? cameraPreviewBuilder;

  /// Builds the scoring pipeline. Defaults to a [MatchController] with a
  /// [TableCalibrator] warm-up. Injectable so tests can drive scoring without
  /// waiting for auto-calibration.
  final MatchController Function()? matchControllerBuilder;

  /// The on-device model to run. Defaults to the COCO `yolo11n` detector, which
  /// labels both `person` (players) and `sports ball` (the ball) in a single
  /// pass — the two classes [YoloVisionService]'s adapter maps into a
  /// [FrameResult]. Swap for a fine-tuned ping-pong-ball model to improve ball
  /// recall (see docs/ARCHITECTURE.md).
  final String modelPath;

  /// The inference task. [YOLOTask.detect] yields both player and ball boxes;
  /// [YOLOTask.pose] adds player keypoints (for footwork analytics) but drops
  /// the ball, so detection is the default for auto-scoring.
  final YOLOTask task;

  @override
  State<CameraMatchScreen> createState() => _CameraMatchScreenState();
}

class _CameraMatchScreenState extends State<CameraMatchScreen> {
  late final YoloVisionService _vision;
  late final MatchController _controller;

  StreamSubscription<FrameResult>? _sub;
  final List<PointDecision> _recentCalls = [];

  @override
  void initState() {
    super.initState();
    _vision = widget.visionService ?? YoloVisionService();
    _controller = widget.matchControllerBuilder?.call() ??
        MatchController(
          calibrator: TableCalibrator(),
          // Real on-device detections carry false positives (a round object or
          // bright logo across the table). Gate them against the Kalman
          // prediction so a spurious detection can't teleport the trajectory and
          // manufacture a bogus point. 0.4 (~40% of the frame) is conservative:
          // it clears normal play and gentle bounces, catching only gross jumps.
          tracker: BallTracker(maxJump: 0.4),
        );
    _startVision();
  }

  Future<void> _startVision() async {
    await _vision.load();
    _sub = _vision.frames.listen(_onFrame);
    await _vision.start();
  }

  void _onFrame(FrameResult frame) {
    final decisions = _controller.onFrame(frame);
    if (!mounted) return;
    setState(() {
      _recentCalls.addAll(decisions);
      if (_recentCalls.length > 5) {
        _recentCalls.removeRange(0, _recentCalls.length - 5);
      }
      if (_controller.score.isMatchOver) _vision.stop();
    });
  }

  void _resolve(PointDecision decision, Player winner) {
    setState(() => _controller.resolveUndetermined(decision, winner));
  }

  void _undo() {
    if (_controller.undo()) setState(() {});
  }

  Widget _buildCameraPreview(BuildContext context) {
    final builder = widget.cameraPreviewBuilder;
    if (builder != null) return builder(context, _vision);
    return YOLOView(
      modelPath: widget.modelPath,
      task: widget.task,
      onStreamingData: _vision.onStreamingData,
    );
  }

  @override
  void dispose() {
    _sub?.cancel();
    _vision.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = _controller.score;
    final pending = _controller.undetermined;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Match'),
        actions: [
          IconButton(
            icon: const Icon(Icons.undo),
            tooltip: 'Undo last point',
            onPressed: _undo,
          ),
        ],
      ),
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            _buildCameraPreview(context),
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: _LiveScoreboard(
                state: state,
                calibrating: _controller.isCalibrating,
              ),
            ),
            if (pending.isNotEmpty)
              Positioned(
                bottom: 0,
                left: 0,
                right: 0,
                child: _UndeterminedPrompt(
                  decision: pending.first,
                  onPick: (winner) => _resolve(pending.first, winner),
                ),
              )
            else
              Positioned(
                bottom: 0,
                left: 0,
                right: 0,
                child: _LiveCallFeed(
                  calls: _recentCalls,
                  matchOver: state.isMatchOver,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// Translucent scoreboard overlaid on the camera preview.
class _LiveScoreboard extends StatelessWidget {
  const _LiveScoreboard({required this.state, required this.calibrating});

  final MatchState state;
  final bool calibrating;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      color: Colors.black54,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        children: [
          if (calibrating)
            Text(
              'Calibrating table… hold the phone steady',
              style: theme.textTheme.labelMedium
                  ?.copyWith(color: Colors.white70),
            ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _Side(
                label: 'A',
                points: state.pointsA,
                games: state.gamesA,
                serving: state.server == Player.a && !state.isMatchOver,
              ),
              Text(
                '–',
                style: theme.textTheme.displaySmall
                    ?.copyWith(color: Colors.white),
              ),
              _Side(
                label: 'B',
                points: state.pointsB,
                games: state.gamesB,
                serving: state.server == Player.b && !state.isMatchOver,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Side extends StatelessWidget {
  const _Side({
    required this.label,
    required this.points,
    required this.games,
    required this.serving,
  });

  final String label;
  final int points;
  final int games;
  final bool serving;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Player $label',
              style: theme.textTheme.labelLarge?.copyWith(color: Colors.white),
            ),
            const SizedBox(width: 6),
            Icon(
              Icons.sports_baseball,
              size: 14,
              color: serving ? theme.colorScheme.primary : Colors.transparent,
            ),
          ],
        ),
        Text(
          '$points',
          style: theme.textTheme.displaySmall?.copyWith(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          'Games: $games',
          style: theme.textTheme.bodySmall?.copyWith(color: Colors.white70),
        ),
      ],
    );
  }
}

/// Rolling referee-call feed overlaid on the camera preview.
class _LiveCallFeed extends StatelessWidget {
  const _LiveCallFeed({required this.calls, required this.matchOver});

  final List<PointDecision> calls;
  final bool matchOver;

  static String _describe(PointDecision d) {
    final who = switch (d.winner) {
      Player.a => 'Player A',
      Player.b => 'Player B',
      null => 'Undetermined',
    };
    final reason = switch (d.reason) {
      PointReason.doubleBounce => 'double bounce',
      PointReason.notReturned => 'not returned',
      PointReason.outOfPlay => 'out of play',
    };
    return '$who — $reason';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      color: Colors.black54,
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            matchOver ? 'Match over' : 'Recent calls',
            style: theme.textTheme.titleSmall?.copyWith(color: Colors.white),
          ),
          const SizedBox(height: 4),
          if (calls.isEmpty)
            Text(
              'Waiting for the first rally…',
              style: theme.textTheme.bodySmall?.copyWith(color: Colors.white70),
            )
          else
            for (final c in calls.reversed)
              Text(
                '• ${_describe(c)}',
                style:
                    theme.textTheme.bodySmall?.copyWith(color: Colors.white70),
              ),
        ],
      ),
    );
  }
}

/// Shown when the referee cannot attribute a point; the user decides.
class _UndeterminedPrompt extends StatelessWidget {
  const _UndeterminedPrompt({required this.decision, required this.onPick});

  final PointDecision decision;
  final void Function(Player winner) onPick;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      color: Colors.black87,
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Who won this point?',
            style: TextStyle(color: Colors.white),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: FilledButton(
                  onPressed: () => onPick(Player.a),
                  child: const Text('Player A'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton(
                  onPressed: () => onPick(Player.b),
                  child: const Text('Player B'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
