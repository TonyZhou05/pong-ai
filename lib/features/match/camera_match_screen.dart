import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../../core/analysis/ball_tracker.dart';
import '../../core/analysis/match_controller.dart';
import '../../core/analysis/match_insights.dart';
import '../../core/analysis/match_report.dart';
import '../../core/analysis/match_report_json.dart';
import '../../core/analysis/rally_referee.dart';
import '../../core/analysis/table_calibrator.dart';
import '../../core/history/history_store_provider.dart';
import '../../core/history/session_history_store.dart';
import '../../core/scoring/match_situation.dart';
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

  /// The most recent frame's detections, drawn as a live tracking overlay on the
  /// camera preview so the user can see what the pipeline is following.
  FrameResult? _lastFrame;

  /// Kalman-extrapolated ball position for a frame whose detector lost the ball,
  /// so the overlay keeps drawing a dimmed "ghost" ball through motion-blur
  /// dropouts instead of blinking out. Null when the ball is visible or the
  /// trajectory has been dropped.
  ({double x, double y})? _predictedBall;

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
      _lastFrame = frame;
      _predictedBall = frame.ball == null
          ? _controller.tracker.estimateBallAt(frame.timestampMs)
          : null;
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

  void _manualPoint(Player winner) {
    setState(() => _controller.awardManualPoint(winner));
    if (_controller.score.isMatchOver) _vision.stop();
  }

  void _setFirstServer(Player p) {
    if (_controller.setFirstServer(p)) setState(() {});
  }

  void _setBestOf(int bestOf) {
    if (_controller.setMatchFormat(bestOf: bestOf)) setState(() {});
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
            // The live tracking overlay: player boxes, the tracked (or predicted)
            // ball, and the calibrated net line drawn coordinate-aligned on top of
            // the camera preview — the "Ball AI" live view of what's being
            // followed. Non-interactive so overlays below (prompt/panel) still get
            // taps.
            Positioned.fill(
              child: IgnorePointer(
                child: _LiveTrackingOverlay(
                  frame: _lastFrame,
                  predictedBall: _predictedBall,
                  netX: _controller.geometry.netX,
                  showNet: !_controller.isCalibrating,
                  // Live "Ball AI"-style km/h readout beside the tracked ball,
                  // shown only when the ball is actually in view (a dropout
                  // leaves the last reading stale).
                  currentSpeedKmh: _controller.currentBallSpeedKmh,
                ),
              ),
            ),
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: _LiveScoreboard(
                state: state,
                calibrating: _controller.isCalibrating,
                // Let the user record who actually serves first while the match
                // hasn't started, so the serve indicator and serve analytics
                // aren't stuck assuming Player A.
                onPickServer:
                    _controller.matchNotStarted ? _setFirstServer : null,
                // Let the user pick the match length before it starts.
                onPickBestOf:
                    _controller.matchNotStarted ? _setBestOf : null,
              ),
            ),
            // Live "reposition the phone" nudge: while the match is on, if the
            // trailing-window detection health is poor (ball/players frequently
            // out of frame) surface the placement hint so the user can fix the
            // phone position instead of only learning it failed at match end.
            if (!state.isMatchOver &&
                _controller.trackingQuality.isPlacementPoor)
              Align(
                alignment: const Alignment(0, -0.35),
                child: IgnorePointer(
                  child: _PlacementWarning(
                    hint: _controller.trackingQuality.recentHint,
                  ),
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
                  // Hand-award a missed point once scoring is live (not during
                  // calibration warm-up).
                  onManualPoint:
                      _controller.isCalibrating ? null : _manualPoint,
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
  const _LiveScoreboard({
    required this.state,
    required this.calibrating,
    this.onPickServer,
    this.onPickBestOf,
  });

  final MatchState state;
  final bool calibrating;

  /// Called when the user taps a player to set who serves first. Null once the
  /// match has started (the first server can no longer change).
  final void Function(Player)? onPickServer;

  /// Called when the user picks the best-of series length. Null once the match
  /// has started (the format can no longer change).
  final void Function(int)? onPickBestOf;

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
          if (MatchSituation(state).bannerLabel case final banner?)
            _PointPressureBanner(
              label: banner,
              matchPoint: MatchSituation(state).isMatchPoint,
            ),
          if (onPickServer != null)
            _ServerPicker(server: state.server, onPick: onPickServer!),
          if (onPickBestOf != null)
            _FormatPicker(bestOf: state.bestOf, onPick: onPickBestOf!),
        ],
      ),
    );
  }
}

/// A flashing "you're one away" cue shown on the live scoreboard whenever a
/// side reaches game point or match point.
class _PointPressureBanner extends StatelessWidget {
  const _PointPressureBanner({required this.label, required this.matchPoint});

  final String label;
  final bool matchPoint;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = matchPoint ? Colors.redAccent : Colors.amber;
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.9),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          label.toUpperCase(),
          style: theme.textTheme.labelMedium?.copyWith(
            color: Colors.black,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.5,
          ),
        ),
      ),
    );
  }
}

/// Lets the user pick the match length (best-of series) before the match
/// starts, so a casual game can be best-of-3 and a full match best-of-7 instead
/// of always being the default best-of-5.
class _FormatPicker extends StatelessWidget {
  const _FormatPicker({required this.bestOf, required this.onPick});

  final int bestOf;
  final void Function(int) onPick;

  static const _options = [3, 5, 7];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            'Best of:',
            style: theme.textTheme.labelMedium?.copyWith(color: Colors.white70),
          ),
          const SizedBox(width: 8),
          for (final n in _options)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 2),
              child: ChoiceChip(
                label: Text('$n'),
                selected: bestOf == n,
                onSelected: (_) => onPick(n),
              ),
            ),
        ],
      ),
    );
  }
}

/// Lets the user record who serves the first point before the match starts, so
/// the serve indicator and serve/receive analytics aren't stuck assuming A.
class _ServerPicker extends StatelessWidget {
  const _ServerPicker({required this.server, required this.onPick});

  final Player server;
  final void Function(Player) onPick;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            'First server:',
            style: theme.textTheme.labelMedium?.copyWith(color: Colors.white70),
          ),
          const SizedBox(width: 8),
          for (final p in Player.values)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 2),
              child: ChoiceChip(
                label: Text(p == Player.a ? 'A' : 'B'),
                selected: server == p,
                onSelected: (_) => onPick(p),
              ),
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

/// Draws the pipeline's current detections over the live camera preview:
/// each tracked player's bounding box, the ball (or a dimmed Kalman-predicted
/// "ghost" ball when the detector lost it), and the calibrated net line. All
/// coordinates are the same normalized `[0,1]` frame space the detections use,
/// so the overlay lines up with the preview (the same normalized space the
/// plugin's own boxes use). This is the live counterpart to the demo
/// [MatchScreen]'s top-down `_TableView`.
class _LiveTrackingOverlay extends StatelessWidget {
  const _LiveTrackingOverlay({
    required this.frame,
    required this.predictedBall,
    required this.netX,
    required this.showNet,
    required this.currentSpeedKmh,
  });

  final FrameResult? frame;
  final ({double x, double y})? predictedBall;
  final double netX;

  /// Whether to draw the net line — suppressed during calibration when the net
  /// position is still the un-inferred default.
  final bool showNet;

  /// The latest ball-speed reading (km/h), rendered as a live label beside the
  /// tracked ball. Null when no reading yet; only drawn while the ball is in
  /// view (a detector dropout leaves this stale, so we hide it with the ball).
  final double? currentSpeedKmh;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final f = frame;
    final ball = f?.ball;
    final ghost = ball == null ? predictedBall : null;
    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final h = constraints.maxHeight;
        return Stack(
          children: [
            if (showNet)
              Positioned(
                left: netX * w - 1,
                top: 0,
                bottom: 0,
                child: Container(width: 2, color: Colors.white38),
              ),
            for (final p in f?.people ?? const [])
              Positioned(
                left: p.box.left * w,
                top: p.box.top * h,
                width: p.box.width * w,
                height: p.box.height * h,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    border: Border.all(color: theme.colorScheme.tertiary),
                  ),
                ),
              ),
            if (ball != null) ...[
              Positioned(
                left: ball.box.centerX * w - 6,
                top: ball.box.centerY * h - 6,
                child: Container(
                  width: 12,
                  height: 12,
                  decoration: const BoxDecoration(
                    color: Color(0xFFFFEB3B),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
              if (currentSpeedKmh != null)
                Positioned(
                  left: ball.box.centerX * w + 10,
                  top: ball.box.centerY * h - 8,
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                    color: Colors.black54,
                    child: Text(
                      '${currentSpeedKmh!.round()} km/h',
                      style: const TextStyle(
                        color: Color(0xFFFFEB3B),
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ),
            ] else if (ghost != null)
              Positioned(
                left: ghost.x * w - 6,
                top: ghost.y * h - 6,
                child: Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: const Color(0x66FFEB3B),
                    shape: BoxShape.circle,
                    border: Border.all(color: const Color(0xAAFFEB3B)),
                  ),
                ),
              ),
          ],
        );
      },
    );
  }
}

/// Rolling referee-call feed overlaid on the camera preview.
class _LiveCallFeed extends StatelessWidget {
  const _LiveCallFeed({
    required this.calls,
    required this.matchOver,
    this.onManualPoint,
  });

  final List<PointDecision> calls;
  final bool matchOver;

  /// Called when the user hand-awards a point the vision missed. Null while
  /// calibrating (no scoring yet) or once the match is over.
  final void Function(Player)? onManualPoint;

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
      PointReason.manual => 'manual',
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
          // Manual score correction: if the pipeline misses a rally, the user
          // can hand the point to the right player so the score stays true.
          if (onManualPoint != null) ...[
            const SizedBox(height: 8),
            Row(
              children: [
                Text(
                  'Missed a point?',
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: Colors.white70),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: const BorderSide(color: Colors.white54),
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    minimumSize: const Size(0, 32),
                  ),
                  onPressed: () => onManualPoint!(Player.a),
                  child: const Text('+A'),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: const BorderSide(color: Colors.white54),
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    minimumSize: const Size(0, 32),
                  ),
                  onPressed: () => onManualPoint!(Player.b),
                  child: const Text('+B'),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

/// Live "reposition the phone" nudge shown mid-match when the trailing-window
/// detection health is poor, so the user can fix a bad phone placement while it
/// still matters rather than only discovering it in the end-of-match summary.
class _PlacementWarning extends StatelessWidget {
  const _PlacementWarning({required this.hint});

  final String hint;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.orange.shade900.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.warning_amber_rounded, color: Colors.white),
          const SizedBox(width: 8),
          Flexible(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Poor tracking',
                  style: theme.textTheme.titleSmall
                      ?.copyWith(color: Colors.white),
                ),
                const SizedBox(height: 2),
                Text(
                  hint,
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: Colors.white),
                ),
              ],
            ),
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
            // The prioritized "what to work on next" cue per player — parity
            // with the demo MatchScreen's summary so the live/production path
            // surfaces coaching, not just raw stats.
            if (MatchInsights(summary).hasData) ...[
              const SizedBox(height: 8),
              Text(
                'Coaching',
                style: theme.textTheme.labelLarge?.copyWith(color: Colors.white),
              ),
              if (MatchInsights(summary).decisiveDimension case final d?)
                Text(
                  'Match difference: ${_name(d.leader!)} won the '
                  '${d.name.toLowerCase()} battle '
                  '(${(d.scoreFor(d.leader!) * 100).round()}% vs '
                  '${(d.scoreFor(d.leader!.other) * 100).round()}%).',
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: Colors.white70),
                ),
              for (final player in Player.values)
                if (MatchInsights(summary).insightsFor(player) case final pi
                    when pi.focusTip != null)
                  Text(
                    '${_name(player)} (Grade ${pi.grade}): ${pi.focusTip}',
                    style: theme.textTheme.bodySmall
                        ?.copyWith(color: Colors.white70),
                  ),
            ],
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
