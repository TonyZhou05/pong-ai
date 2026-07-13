import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/analysis/ball_tracker.dart';
import '../../core/analysis/tracking_quality.dart';
import '../../core/history/history_store_provider.dart';
import '../../core/history/session_history_store.dart';
import '../../core/share/report_share.dart';
import '../../core/training/shot_analyzer.dart';
import '../../core/training/training_feedback.dart';
import '../../core/training/training_report_json.dart';
import '../../core/vision/detection.dart';
import '../../core/vision/replay_vision_service.dart';
import '../../core/vision/synthetic_frames.dart';
import '../../core/vision/vision_service.dart';
import 'training_shot_map.dart';

/// Live training screen: streams vision frames through a [ShotAnalyzer] and
/// renders each graded stroke plus a running session summary (pace, placement
/// depth and consistency), mirroring how [MatchScreen] surfaces the match
/// pipeline.
///
/// The frame source is injectable so tests (and, later, the real
/// `ultralytics_yolo` camera runtime) can supply their own [VisionService]. It
/// defaults to a [ReplayVisionService] playing a scripted drill so the whole
/// shot-analysis path is visible in-app without a camera.
class TrainingScreen extends StatefulWidget {
  const TrainingScreen({
    super.key,
    this.visionServiceBuilder,
    this.config = const TrainingConfig(),
    this.historyStoreLoader = defaultSessionHistoryStore,
    this.shareReport = defaultShareReport,
  });

  /// Builds the frame source. Defaults to the scripted training replay.
  final VisionService Function()? visionServiceBuilder;

  /// What a "good" shot looks like (target depth, pace reference, etc.).
  final TrainingConfig config;

  /// Resolves the store the "Save to history" action writes to. Defaults to the
  /// on-device documents-directory store; tests inject a temp-dir store.
  final Future<SessionHistoryStore> Function() historyStoreLoader;

  /// Hands the composed text report to the OS share sheet. Defaults to the
  /// `share_plus`-backed sink; tests inject a fake that records the text.
  final ShareReportSink shareReport;

  @override
  State<TrainingScreen> createState() => _TrainingScreenState();
}

class _TrainingScreenState extends State<TrainingScreen> {
  late final VisionService _vision;
  late final ShotAnalyzer _analyzer;
  // Single-player detection-health for the phone-placement hint (training has
  // one player, so it scores on any-player visibility, not both ends).
  final TrackingQualityAnalyzer _quality =
      TrackingQualityAnalyzer(requireBothPlayers: false);

  StreamSubscription<FrameResult>? _sub;
  FrameResult? _lastFrame;
  bool _finished = false;
  final List<Shot> _recentShots = [];

  @override
  void initState() {
    super.initState();
    _analyzer = ShotAnalyzer(config: widget.config);
    _vision = widget.visionServiceBuilder?.call() ??
        ReplayVisionService(trainingSessionFrames());
    _startVision();
  }

  Future<void> _startVision() async {
    await _vision.load();
    _sub = _vision.frames.listen(_onFrame, onDone: _onDone);
    await _vision.start();
  }

  void _onFrame(FrameResult frame) {
    final shot = _analyzer.onFrame(frame);
    _quality.observe(frame);
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

  void _onDone() {
    if (mounted) setState(() => _finished = true);
  }

  void _restart() {
    setState(() {
      _analyzer.reset();
      _quality.reset();
      _recentShots.clear();
      _finished = false;
      _lastFrame = null;
    });
    _vision.start();
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
        title: const Text('Training'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Restart drill',
            onPressed: _restart,
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            _SummaryHeader(summary: summary),
            const SizedBox(height: 8),
            Expanded(
              child: _TargetView(frame: _lastFrame, config: widget.config),
            ),
            if (_finished)
              // Bounded so the report's internal scroll view fits (and scrolls)
              // instead of overflowing the column with the placement map.
              Flexible(
                child: _SessionReport(
                  summary: summary,
                  config: widget.config,
                  quality: _quality,
                  historyStoreLoader: widget.historyStoreLoader,
                  shareReport: widget.shareReport,
                ),
              )
            else
              _ShotFeed(shots: _recentShots),
          ],
        ),
      ),
    );
  }
}

/// Big at-a-glance readout: session grade, shot count and headline metrics.
class _SummaryHeader extends StatelessWidget {
  const _SummaryHeader({required this.summary});

  final TrainingSummary summary;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Column(
            children: [
              Text('Grade', style: theme.textTheme.labelLarge),
              Text(
                summary.overallGrade,
                style: theme.textTheme.displayLarge
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(width: 24),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${summary.shotCount} shots', style: theme.textTheme.titleMedium),
                Text('Avg depth: ${(summary.averageDepth * 100).round()}%'),
                Text('Consistency: ${(summary.consistency * 100).round()}%'),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Schematic top-down table with the net, the target landing band and the
/// currently-tracked ball — the "Ball AI"-style live overlay for training.
class _TargetView extends StatelessWidget {
  const _TargetView({required this.frame, required this.config});

  final FrameResult? frame;
  final TrainingConfig config;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ball = frame?.ball;
    // The target band spans [targetDepth ± tolerance] on the far half, mapped
    // back to normalized x. Depth d on the right half is x = netX + d·(1−netX).
    final netX = config.geometry.netX;
    final onRight = config.targetSide == TableSide.right;
    double depthToX(double d) =>
        onRight ? netX + d * (1 - netX) : netX - d * netX;
    final near = depthToX((config.targetDepth - config.depthTolerance).clamp(0.0, 1.0));
    final far = depthToX((config.targetDepth + config.depthTolerance).clamp(0.0, 1.0));
    final bandLeft = near < far ? near : far;
    final bandRight = near < far ? far : near;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: AspectRatio(
        aspectRatio: 16 / 9,
        child: LayoutBuilder(
          builder: (context, constraints) {
            final w = constraints.maxWidth;
            final h = constraints.maxHeight;
            return DecoratedBox(
              decoration: BoxDecoration(
                color: const Color(0xFF0D3B12),
                border: Border.all(color: Colors.white24),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Stack(
                children: [
                  // Target landing band.
                  Positioned(
                    left: bandLeft * w,
                    top: 0,
                    width: (bandRight - bandLeft) * w,
                    height: h,
                    child: const DecoratedBox(
                      decoration: BoxDecoration(color: Color(0x3348C9B0)),
                    ),
                  ),
                  // Net line.
                  Align(
                    alignment: Alignment(netX * 2 - 1, 0),
                    child: Container(width: 2, color: Colors.white38),
                  ),
                  if (ball != null)
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
                  Positioned(
                    left: 8,
                    bottom: 6,
                    child: Text(
                      ball == null ? 'tracking…' : 'ball locked',
                      style: theme.textTheme.labelSmall
                          ?.copyWith(color: Colors.white70),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

/// Rolling list of the most recent graded strokes.
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
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Recent shots', style: theme.textTheme.titleMedium),
          const SizedBox(height: 4),
          if (shots.isEmpty)
            Text('Waiting for the first shot…', style: theme.textTheme.bodyMedium)
          else
            for (final s in shots.reversed)
              Text('• ${_describe(s)}', style: theme.textTheme.bodyMedium),
        ],
      ),
    );
  }
}

/// End-of-session report, shown once the replay finishes.
class _SessionReport extends StatelessWidget {
  const _SessionReport({
    required this.summary,
    required this.config,
    required this.quality,
    required this.historyStoreLoader,
    required this.shareReport,
  });

  final TrainingSummary summary;
  final TrainingConfig config;
  final TrackingQualityAnalyzer quality;
  final Future<SessionHistoryStore> Function() historyStoreLoader;

  /// Hands the composed text report to the OS share sheet.
  final ShareReportSink shareReport;

  /// The full text report shared / copied — session summary plus coaching
  /// feedback and any tracking-quality note.
  String _reportText() {
    final feedback = TrainingFeedback(summary, config: config);
    return [
      summary.report(),
      if (feedback.hasData) feedback.report(),
      if (quality.hasData) quality.report(),
    ].join('\n\n');
  }

  Future<void> _saveToHistory(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    final store = await historyStoreLoader();
    await store.save(
      kind: SessionKind.training,
      report: buildTrainingReportJson(summary, config: config),
    );
    messenger.showSnackBar(
      const SnackBar(content: Text('Saved to history')),
    );
  }

  Future<void> _copyReport(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    await Clipboard.setData(ClipboardData(text: _reportText()));
    messenger.showSnackBar(
      const SnackBar(content: Text('Report copied to clipboard')),
    );
  }

  Future<void> _shareReport(BuildContext context) async {
    await shareReport(
      _reportText(),
      subject: 'Table tennis training summary',
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
    final feedback = TrainingFeedback(summary, config: config);
    return Container(
      width: double.infinity,
      color: theme.colorScheme.surfaceContainerHighest,
      padding: const EdgeInsets.all(16),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Session complete', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(summary.report(), style: theme.textTheme.bodyMedium),
            if (feedback.focusTip != null) ...[
              const SizedBox(height: 8),
              Text(
                'Focus next: ${feedback.focusTip}',
                style: theme.textTheme.bodyMedium
                    ?.copyWith(fontWeight: FontWeight.w600),
              ),
            ],
            if (quality.hasData) ...[
              const SizedBox(height: 8),
              Text(
                'Tracking quality: ${quality.grade} — ${quality.hint}',
                style: theme.textTheme.bodySmall,
              ),
            ],
            if (summary.shots.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text('Placement map', style: theme.textTheme.titleSmall),
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
                    icon: const Icon(Icons.save_alt, size: 18),
                    label: const Text('Save to history'),
                    onPressed: () => _saveToHistory(context),
                  ),
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
                  OutlinedButton.icon(
                    icon: const Icon(Icons.ios_share, size: 18),
                    label: const Text('Share'),
                    onPressed: () => _shareReport(context),
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
