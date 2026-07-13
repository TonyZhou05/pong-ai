import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../../core/analysis/ball_tracker.dart';
import '../../core/analysis/match_controller.dart';
import '../../core/analysis/match_report.dart';
import '../../core/analysis/match_report_json.dart';
import '../../core/analysis/rally_referee.dart';
import '../../core/analysis/table_calibrator.dart';
import '../../core/history/history_store_provider.dart';
import '../../core/history/session_history_store.dart';
import '../../core/scoring/scoring_engine.dart';
import '../../core/vision/detection.dart';
import '../../core/vision/vision_model_profile.dart';
import '../../core/vision/yolo_vision_service.dart';
import '../summary/momentum_chart.dart';
import '../summary/player_map.dart';
import '../summary/shot_map.dart';

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
    this.model = defaultVisionModel,
    this.historyStoreLoader = defaultSessionHistoryStore,
  });

  /// The camera-backed frame source. Defaults to one whose adapter decodes
  /// [model]'s output (its label set + confidence thresholds).
  final YoloVisionService? visionService;

  /// Builds the camera preview widget. Defaults to a real [YOLOView] wired to
  /// [visionService]. Injectable so tests can substitute a headless stand-in.
  final Widget Function(BuildContext, YoloVisionService)? cameraPreviewBuilder;

  /// Builds the scoring pipeline. Defaults to a [MatchController] with a
  /// [TableCalibrator] warm-up. Injectable so tests can drive scoring without
  /// waiting for auto-calibration.
  final MatchController Function()? matchControllerBuilder;

  /// The on-device model to run. Defaults to the stock COCO detector
  /// ([cocoDetectProfile]), which labels both `person` (players) and
  /// `sports ball` (the ball) in a single pass. Swap to [pingPongDetectProfile]
  /// once a fine-tuned model is bundled to improve ball recall (see
  /// docs/ARCHITECTURE.md); the profile carries both the model path and the
  /// matching decode config so the swap is a single coherent choice.
  final VisionModelProfile model;

  /// Resolves the store the end-of-match "Save to history" action writes to.
  /// Defaults to the on-device documents-directory store; tests inject an
  /// in-memory fake.
  final Future<SessionHistoryStore> Function() historyStoreLoader;

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
    _vision = widget.visionService ?? widget.model.createVisionService();
    _controller = widget.matchControllerBuilder?.call() ??
        MatchController(
          calibrator: TableCalibrator(),
          // A real match changes ends between games while the phone stays put,
          // so flip the side→player mapping each completed game to keep
          // attributing bounces to the right player. (Off for scripted demos,
          // which never physically switch ends.)
          switchEndsBetweenGames: true,
          // Real on-device detections carry false positives (a round object or
          // bright logo across the table). Gate them against the Kalman
          // prediction so a spurious detection can't teleport the trajectory and
          // manufacture a bogus point. 0.4 (~40% of the frame) is conservative:
          // it clears normal play and gentle bounces, catching only gross jumps.
          tracker: BallTracker(maxJump: 0.4),
          // A stationary player's detected feet wobble a few pixels each frame;
          // summing that raw jitter across a match inflates the footwork
          // distance. Ignore sub-~1%-of-frame per-frame drift so only real
          // movement is counted. (Off for scripted demos with exact motions.)
          movementJitterThreshold: 0.01,
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
      modelPath: widget.model.modelPath,
      task: widget.model.task,
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
            else if (state.isMatchOver)
              Positioned(
                bottom: 0,
                left: 0,
                right: 0,
                child: _MatchOverPanel(
                  controller: _controller,
                  historyStoreLoader: widget.historyStoreLoader,
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

/// End-of-match summary overlaid on the frozen camera preview once the match is
/// over. This is the live-camera counterpart to [MatchScreen]'s summary panel:
/// it surfaces the winner, key analytics, and the same Save-to-history / Copy
/// report / Export JSON actions so the production camera path can persist and
/// share a match, not just show a "Match over" line.
class _MatchOverPanel extends StatelessWidget {
  const _MatchOverPanel({
    required this.controller,
    required this.historyStoreLoader,
  });

  final MatchController controller;
  final Future<SessionHistoryStore> Function() historyStoreLoader;

  static String _name(Player p) => p == Player.a ? 'Player A' : 'Player B';

  Future<void> _saveToHistory(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    final store = await historyStoreLoader();
    await store.save(
      kind: SessionKind.match,
      report: buildMatchReportJson(controller),
    );
    messenger.showSnackBar(
      const SnackBar(content: Text('Saved to history')),
    );
  }

  Future<void> _copyReport(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    await Clipboard.setData(ClipboardData(text: buildMatchReport(controller)));
    messenger.showSnackBar(
      const SnackBar(content: Text('Report copied to clipboard')),
    );
  }

  Future<void> _exportJson(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    await Clipboard.setData(
      ClipboardData(text: matchReportJsonString(controller)),
    );
    messenger.showSnackBar(
      const SnackBar(content: Text('JSON summary copied to clipboard')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final summary = controller.summary;
    final winner = summary.matchWinner;
    return Container(
      width: double.infinity,
      color: Colors.black87,
      padding: const EdgeInsets.all(16),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              winner == null
                  ? 'Match over'
                  : '${_name(winner)} wins the match',
              style: theme.textTheme.titleMedium?.copyWith(color: Colors.white),
            ),
            const SizedBox(height: 4),
            Text(
              '${summary.totalPoints} points played',
              style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white70),
            ),
            if (summary.gameScores.isNotEmpty)
              Text(
                'Games: ${summary.gameScores.join(', ')}',
                style:
                    theme.textTheme.bodyMedium?.copyWith(color: Colors.white70),
              ),
            if (controller.hasBallSpeedData)
              Text(
                'Top ball speed: '
                '${controller.maxBallSpeedKmh.toStringAsFixed(1)} km/h',
                style:
                    theme.textTheme.bodyMedium?.copyWith(color: Colors.white70),
              ),
            if (controller.trackingQuality.hasData)
              Text(
                'Tracking quality: grade ${controller.trackingQuality.grade}',
                style:
                    theme.textTheme.bodyMedium?.copyWith(color: Colors.white70),
              ),
            if (summary.totalPoints > 0) ...[
              const SizedBox(height: 8),
              Text(
                'Momentum',
                style:
                    theme.textTheme.labelLarge?.copyWith(color: Colors.white),
              ),
              const SizedBox(height: 4),
              MomentumChartView(points: summary.points),
            ],
            if (controller.placementFor(TableSide.left).count +
                    controller.placementFor(TableSide.right).count >
                0) ...[
              const SizedBox(height: 8),
              Text(
                'Shot map',
                style:
                    theme.textTheme.labelLarge?.copyWith(color: Colors.white),
              ),
              const SizedBox(height: 4),
              ShotMapView(
                left: controller.placementFor(TableSide.left),
                right: controller.placementFor(TableSide.right),
              ),
            ],
            if (Player.values
                .any((p) => controller.positionsFor(p).isNotEmpty)) ...[
              const SizedBox(height: 8),
              Text(
                'Player coverage',
                style:
                    theme.textTheme.labelLarge?.copyWith(color: Colors.white),
              ),
              const SizedBox(height: 4),
              PlayerPositionMapView(
                positions: {
                  for (final p in Player.values) p: controller.positionsFor(p),
                },
                netX: controller.geometry.netX,
              ),
            ],
            const SizedBox(height: 8),
            Row(
              children: [
                for (final player in Player.values)
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _name(player),
                          style: theme.textTheme.labelLarge
                              ?.copyWith(color: Colors.white),
                        ),
                        Text(
                          '${summary.pointsWonBy(player)} pts won',
                          style: theme.textTheme.bodySmall
                              ?.copyWith(color: Colors.white70),
                        ),
                        Text(
                          '${summary.forcedErrorsWonBy(player)} forced errors',
                          style: theme.textTheme.bodySmall
                              ?.copyWith(color: Colors.white70),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: Wrap(
                spacing: 8,
                children: [
                  OutlinedButton.icon(
                    icon: const Icon(Icons.save_alt, size: 18),
                    label: const Text('Save to history'),
                    onPressed: () => _saveToHistory(context),
                  ),
                  OutlinedButton.icon(
                    icon: const Icon(Icons.data_object, size: 18),
                    label: const Text('Export JSON'),
                    onPressed: () => _exportJson(context),
                  ),
                  OutlinedButton.icon(
                    icon: const Icon(Icons.copy, size: 18),
                    label: const Text('Copy report'),
                    onPressed: () => _copyReport(context),
                  ),
                ],
              ),
            ),
          ],
        ),
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
