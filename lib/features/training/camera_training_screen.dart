import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../../core/analysis/ball_tracker.dart';
import '../../core/training/shot_analyzer.dart';
import '../../core/training/training_report_json.dart';
import '../../core/vision/detection.dart';
import '../../core/vision/vision_model_profile.dart';
import '../../core/vision/yolo_vision_service.dart';
import 'training_shot_map.dart';

/// The **live camera** training screen: runs on-device `ultralytics_yolo`
/// inference over the phone's camera and grades each practice stroke in real
/// time via the [ShotAnalyzer] pipeline.
///
/// This is the production counterpart to [TrainingScreen] (which replays a
/// scripted drill): here a [YOLOView] platform view produces detections, its
/// `onStreamingData` callback feeds a [YoloVisionService], and that service's
/// [FrameResult] stream drives a [ShotAnalyzer]. The camera preview fills the
/// screen and the session grade, target band and recent-shot feed are overlaid
/// on top — the "Ball AI"-style live training view.
///
/// Placed table-side facing the practice net, each outgoing stroke that crosses
/// the net and lands on the target half is segmented, graded on landing depth
/// and pace, and folded into a running session summary. Tap "Finish" to freeze
/// the session and read the end-of-session report.
///
/// The camera preview is injected via [cameraPreviewBuilder] and the frame
/// source via [visionService] so widget tests can drive the pipeline headlessly
/// without a platform view or camera.
class CameraTrainingScreen extends StatefulWidget {
  const CameraTrainingScreen({
    super.key,
    this.visionService,
    this.cameraPreviewBuilder,
    this.config = const TrainingConfig(),
    this.model = defaultVisionModel,
  });

  /// The camera-backed frame source. Defaults to one whose adapter decodes
  /// [model]'s output (its label set + confidence thresholds).
  final YoloVisionService? visionService;

  /// Builds the camera preview widget. Defaults to a real [YOLOView] wired to
  /// [visionService]. Injectable so tests can substitute a headless stand-in.
  final Widget Function(BuildContext, YoloVisionService)? cameraPreviewBuilder;

  /// What a "good" shot looks like (target depth, pace reference, player side).
  final TrainingConfig config;

  /// The on-device model to run. Defaults to the stock COCO detector
  /// ([cocoDetectProfile]), which labels `sports ball` — the ball the
  /// [ShotAnalyzer] tracks. Swap to [pingPongDetectProfile] once a fine-tuned
  /// model is bundled to improve recall (see docs/ARCHITECTURE.md); the profile
  /// carries both the model path and the matching decode config.
  final VisionModelProfile model;

  @override
  State<CameraTrainingScreen> createState() => _CameraTrainingScreenState();
}

class _CameraTrainingScreenState extends State<CameraTrainingScreen> {
  late final YoloVisionService _vision;
  late final ShotAnalyzer _analyzer;

  StreamSubscription<FrameResult>? _sub;
  FrameResult? _lastFrame;
  bool _finished = false;
  final List<Shot> _recentShots = [];

  @override
  void initState() {
    super.initState();
    _vision = widget.visionService ?? widget.model.createVisionService();
    _analyzer = ShotAnalyzer(config: widget.config);
    _startVision();
  }

  Future<void> _startVision() async {
    await _vision.load();
    _sub = _vision.frames.listen(_onFrame);
    await _vision.start();
  }

  void _onFrame(FrameResult frame) {
    final shot = _analyzer.onFrame(frame);
    if (!mounted) return;
    setState(() {
      _lastFrame = frame;
      if (shot != null) {
        _recentShots.add(shot);
        if (_recentShots.length > 5) {
          _recentShots.removeRange(0, _recentShots.length - 5);
        }
      }
    });
  }

  void _finish() {
    _vision.stop();
    setState(() => _finished = true);
  }

  void _restart() {
    _analyzer.reset();
    setState(() {
      _recentShots.clear();
      _finished = false;
      _lastFrame = null;
    });
    _vision.start();
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
    final summary = _analyzer.summary;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Training'),
        actions: [
          IconButton(
            icon: Icon(_finished ? Icons.play_arrow : Icons.stop),
            tooltip: _finished ? 'Restart drill' : 'Finish session',
            onPressed: _finished ? _restart : _finish,
          ),
        ],
      ),
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            _buildCameraPreview(context),
            _TargetOverlay(frame: _lastFrame, config: widget.config),
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: _LiveSummaryHeader(summary: summary),
            ),
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: _finished
                  ? _SessionReport(summary: summary, config: widget.config)
                  : _ShotFeed(shots: _recentShots),
            ),
          ],
        ),
      ),
    );
  }
}

/// Translucent session-grade header overlaid on the camera preview.
class _LiveSummaryHeader extends StatelessWidget {
  const _LiveSummaryHeader({required this.summary});

  final TrainingSummary summary;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      color: Colors.black54,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Column(
            children: [
              Text(
                'Grade',
                style: theme.textTheme.labelMedium
                    ?.copyWith(color: Colors.white70),
              ),
              Text(
                summary.overallGrade,
                style: theme.textTheme.displaySmall?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${summary.shotCount} shots',
                  style: theme.textTheme.titleMedium
                      ?.copyWith(color: Colors.white),
                ),
                Text(
                  'Avg depth: ${(summary.averageDepth * 100).round()}%   '
                  'Consistency: ${(summary.consistency * 100).round()}%',
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: Colors.white70),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Draws the net line, the target landing band and the tracked ball on top of
/// the camera preview so the player can see where shots should land.
class _TargetOverlay extends StatelessWidget {
  const _TargetOverlay({required this.frame, required this.config});

  final FrameResult? frame;
  final TrainingConfig config;

  @override
  Widget build(BuildContext context) {
    final ball = frame?.ball;
    // The target band spans [targetDepth ± tolerance] on the far half, mapped
    // back to normalized x. Depth d on the right half is x = netX + d·(1−netX);
    // on the left half it mirrors to x = netX − d·netX.
    final netX = config.geometry.netX;
    final onRight = config.targetSide == TableSide.right;
    double depthToX(double d) =>
        onRight ? netX + d * (1 - netX) : netX - d * netX;
    final near =
        depthToX((config.targetDepth - config.depthTolerance).clamp(0.0, 1.0));
    final far =
        depthToX((config.targetDepth + config.depthTolerance).clamp(0.0, 1.0));
    final bandLeft = near < far ? near : far;
    final bandRight = near < far ? far : near;

    return IgnorePointer(
      child: LayoutBuilder(
        builder: (context, constraints) {
          final w = constraints.maxWidth;
          final h = constraints.maxHeight;
          return Stack(
            children: [
              // Target landing band.
              Positioned(
                left: bandLeft * w,
                top: 0,
                width: (bandRight - bandLeft) * w,
                height: h,
                child: const DecoratedBox(
                  decoration: BoxDecoration(color: Color(0x2648C9B0)),
                ),
              ),
              // Net line.
              Align(
                alignment: Alignment(netX * 2 - 1, 0),
                child: Container(width: 2, color: Colors.white54),
              ),
              if (ball != null)
                Positioned(
                  left: ball.box.centerX * w - 7,
                  top: ball.box.centerY * h - 7,
                  child: Container(
                    width: 14,
                    height: 14,
                    decoration: BoxDecoration(
                      color: const Color(0xFFFFEB3B),
                      shape: BoxShape.circle,
                      border: Border.all(color: Colors.black54),
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

/// Rolling list of the most recent graded strokes overlaid on the camera.
class _ShotFeed extends StatelessWidget {
  const _ShotFeed({required this.shots});

  final List<Shot> shots;

  static String _describe(Shot s) {
    final depth = (s.depth * 100).round();
    final pct = (s.score * 100).round();
    return '${s.grade.name} — depth $depth%, $pct%';
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
            'Recent shots',
            style: theme.textTheme.titleSmall?.copyWith(color: Colors.white),
          ),
          const SizedBox(height: 4),
          if (shots.isEmpty)
            Text(
              'Waiting for the first shot…',
              style: theme.textTheme.bodySmall?.copyWith(color: Colors.white70),
            )
          else
            for (final s in shots.reversed)
              Text(
                '• ${_describe(s)}',
                style:
                    theme.textTheme.bodySmall?.copyWith(color: Colors.white70),
              ),
        ],
      ),
    );
  }
}

/// End-of-session report, shown once the player taps Finish.
class _SessionReport extends StatelessWidget {
  const _SessionReport({required this.summary, required this.config});

  final TrainingSummary summary;
  final TrainingConfig config;

  Future<void> _copyReport(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    await Clipboard.setData(ClipboardData(text: summary.report()));
    messenger.showSnackBar(
      const SnackBar(content: Text('Report copied to clipboard')),
    );
  }

  Future<void> _exportJson(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    await Clipboard.setData(
      ClipboardData(text: trainingReportJsonString(summary, config: config)),
    );
    messenger.showSnackBar(
      const SnackBar(content: Text('JSON report copied to clipboard')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      color: Colors.black87,
      padding: const EdgeInsets.all(16),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Session complete',
              style: theme.textTheme.titleMedium?.copyWith(color: Colors.white),
            ),
            const SizedBox(height: 8),
            Text(
              summary.report(),
              style:
                  theme.textTheme.bodyMedium?.copyWith(color: Colors.white70),
            ),
            if (summary.shots.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text(
                'Placement map',
                style:
                    theme.textTheme.titleSmall?.copyWith(color: Colors.white),
              ),
              const SizedBox(height: 6),
              TrainingShotMapView(shots: summary.shots, config: config),
            ],
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: Wrap(
                spacing: 8,
                children: [
                  OutlinedButton.icon(
                    icon: const Icon(Icons.copy, size: 18),
                    label: const Text('Copy report'),
                    onPressed: () => _copyReport(context),
                  ),
                  OutlinedButton.icon(
                    icon: const Icon(Icons.data_object, size: 18),
                    label: const Text('Export JSON'),
                    onPressed: () => _exportJson(context),
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
