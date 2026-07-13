import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/analysis/ball_tracker.dart';
import '../../core/analysis/bounce_placement.dart';
import '../../core/analysis/match_controller.dart';
import '../../core/analysis/match_insights.dart';
import '../../core/analysis/match_report.dart';
import '../../core/analysis/match_report_json.dart';
import '../../core/analysis/match_summary.dart';
import '../../core/analysis/player_movement.dart';
import '../../core/analysis/rally_analyzer.dart';
import '../../core/analysis/rally_referee.dart';
import '../../core/analysis/tracking_quality.dart';
import '../../core/history/history_store_provider.dart';
import '../../core/history/session_history_store.dart';
import '../../core/share/report_share.dart';
import '../../core/scoring/match_situation.dart';
import '../../core/scoring/scoring_engine.dart';
import '../../core/vision/detection.dart';
import '../../core/vision/replay_vision_service.dart';
import '../../core/vision/synthetic_frames.dart';
import '../../core/vision/vision_service.dart';
import '../summary/momentum_chart.dart';
import '../summary/player_map.dart';
import '../summary/shot_map.dart';

/// Live match screen: streams vision frames through the [MatchController] and
/// renders the running score, the tracked ball, and the referee's calls.
///
/// The frame source is injectable so tests (and, later, the real
/// `ultralytics_yolo` camera runtime) can supply their own [VisionService]. It
/// defaults to a [ReplayVisionService] playing a scripted demo match so the
/// whole pipeline is visible in-app without a camera.
class MatchScreen extends StatefulWidget {
  const MatchScreen({
    super.key,
    this.visionServiceBuilder,
    this.matchControllerBuilder,
    this.historyStoreLoader = defaultSessionHistoryStore,
    this.shareReport = defaultShareReport,
  });

  /// Builds the frame source. Defaults to the scripted demo replay.
  final VisionService Function()? visionServiceBuilder;

  /// Builds the scoring pipeline. Defaults to a plain [MatchController]; tests
  /// inject one with a short match config to reach the summary panel quickly.
  final MatchController Function()? matchControllerBuilder;

  /// Resolves the store the "Save to history" action writes to. Defaults to the
  /// on-device documents-directory store; tests inject an in-memory fake.
  final Future<SessionHistoryStore> Function() historyStoreLoader;

  /// Hands the composed text report to the OS share sheet ("send to a coach").
  /// Defaults to the `share_plus`-backed sink; tests inject a fake that records
  /// the shared text.
  final ShareReportSink shareReport;

  @override
  State<MatchScreen> createState() => _MatchScreenState();
}

class _MatchScreenState extends State<MatchScreen> {
  late final VisionService _vision;
  late final MatchController _controller;

  StreamSubscription<FrameResult>? _sub;
  FrameResult? _lastFrame;

  /// Kalman-extrapolated ball position for a frame whose detector lost the ball,
  /// so the overlay can keep drawing it through motion-blur dropouts. Null when
  /// the ball is visible or the trajectory has been dropped.
  ({double x, double y})? _predictedBall;
  final List<PointDecision> _recentCalls = [];

  @override
  void initState() {
    super.initState();
    _controller = widget.matchControllerBuilder?.call() ?? MatchController();
    _vision = widget.visionServiceBuilder?.call() ??
        ReplayVisionService(demoMatchFrames());
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
      if (_controller.score.isMatchOver) {
        _vision.stop();
      }
    });
  }

  void _resolve(PointDecision decision, Player winner) {
    setState(() => _controller.resolveUndetermined(decision, winner));
  }

  void _undo() {
    if (_controller.undo()) setState(() {});
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
        title: const Text('Match'),
        actions: [
          IconButton(
            icon: const Icon(Icons.undo),
            tooltip: 'Undo last point',
            onPressed: _undo,
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            _Scoreboard(state: state),
            const SizedBox(height: 8),
            Expanded(
              child: _TableView(frame: _lastFrame, predictedBall: _predictedBall),
            ),
            if (pending.isNotEmpty)
              _UndeterminedPrompt(
                decision: pending.first,
                onPick: (winner) => _resolve(pending.first, winner),
              )
            else if (state.isMatchOver)
              Flexible(
                child: _SummaryPanel(
                  summary: _controller.summary,
                  rallies: _controller.rallyStats,
                  movement: {
                    for (final p in Player.values) p: _controller.movementFor(p),
                  },
                  positions: {
                    for (final p in Player.values)
                      p: _controller.positionsFor(p),
                  },
                  netX: _controller.geometry.netX,
                  placement: {
                    for (final s in TableSide.values)
                      s: _controller.placementFor(s),
                  },
                  maxBallSpeedKmh: _controller.hasBallSpeedData
                      ? _controller.maxBallSpeedKmh
                      : null,
                  tracking: _controller.trackingQuality,
                  reportText: buildMatchReport(_controller),
                  reportJson: matchReportJsonString(_controller),
                  reportJsonMap: buildMatchReportJson(_controller),
                  historyStoreLoader: widget.historyStoreLoader,
                  shareReport: widget.shareReport,
                ),
              )
            else
              _CallFeed(calls: _recentCalls, matchOver: state.isMatchOver),
          ],
        ),
      ),
    );
  }
}

/// Big two-sided score readout with a serve indicator.
class _Scoreboard extends StatelessWidget {
  const _Scoreboard({required this.state});

  final MatchState state;

  @override
  Widget build(BuildContext context) {
    final situation = MatchSituation(state);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Column(
        children: [
          Row(
            children: [
              _PlayerScore(
                label: 'A',
                points: state.pointsA,
                games: state.gamesA,
                serving: state.server == Player.a && !state.isMatchOver,
              ),
              const _ScoreSeparator(),
              _PlayerScore(
                label: 'B',
                points: state.pointsB,
                games: state.gamesB,
                serving: state.server == Player.b && !state.isMatchOver,
              ),
            ],
          ),
          if (situation.bannerLabel case final banner?)
            _PointPressureBanner(
              label: banner,
              matchPoint: situation.isMatchPoint,
            ),
        ],
      ),
    );
  }
}

/// A "you're one away" cue shown below the scoreboard whenever a side reaches
/// game point or match point.
class _PointPressureBanner extends StatelessWidget {
  const _PointPressureBanner({required this.label, required this.matchPoint});

  final String label;
  final bool matchPoint;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = matchPoint
        ? theme.colorScheme.error
        : theme.colorScheme.tertiary;
    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          label.toUpperCase(),
          style: theme.textTheme.labelMedium?.copyWith(
            color: color,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.5,
          ),
        ),
      ),
    );
  }
}

class _PlayerScore extends StatelessWidget {
  const _PlayerScore({
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
    return Expanded(
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Player $label', style: theme.textTheme.labelLarge),
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
            style: theme.textTheme.displayLarge
                ?.copyWith(fontWeight: FontWeight.bold),
          ),
          Text('Games: $games', style: theme.textTheme.bodyMedium),
        ],
      ),
    );
  }
}

class _ScoreSeparator extends StatelessWidget {
  const _ScoreSeparator();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 24),
      child: Text('–', style: Theme.of(context).textTheme.displayMedium),
    );
  }
}

/// A schematic top-down table with the net and the currently-tracked ball,
/// plus any detected player boxes — the "Ball AI"-style live overlay.
class _TableView extends StatelessWidget {
  const _TableView({required this.frame, this.predictedBall});

  final FrameResult? frame;

  /// Kalman-extrapolated ball position shown (dimmed) when the detector lost the
  /// ball this frame, so tracking stays visually continuous through blur.
  final ({double x, double y})? predictedBall;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: AspectRatio(
        aspectRatio: 16 / 9,
        child: LayoutBuilder(
          builder: (context, constraints) {
            final w = constraints.maxWidth;
            final h = constraints.maxHeight;
            final ball = frame?.ball;
            final ghost = ball == null ? predictedBall : null;
            return DecoratedBox(
              decoration: BoxDecoration(
                color: const Color(0xFF0D3B12),
                border: Border.all(color: Colors.white24),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Stack(
                children: [
                  // Net line down the middle.
                  Align(
                    alignment: Alignment.center,
                    child: Container(width: 2, color: Colors.white38),
                  ),
                  for (final p in frame?.people ?? const [])
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
                    )
                  else if (ghost != null)
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
                  Positioned(
                    left: 8,
                    bottom: 6,
                    child: Text(
                      ball != null
                          ? 'ball locked'
                          : ghost != null
                              ? 'predicting…'
                              : 'tracking…',
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

/// Rolling list of the referee's recent point calls.
class _CallFeed extends StatelessWidget {
  const _CallFeed({required this.calls, required this.matchOver});

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
      PointReason.manual => 'manual',
    };
    return '$who — $reason';
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
          Text(
            matchOver ? 'Match over' : 'Recent calls',
            style: theme.textTheme.titleMedium,
          ),
          const SizedBox(height: 4),
          if (calls.isEmpty)
            Text(
              'Waiting for the first rally…',
              style: theme.textTheme.bodyMedium,
            )
          else
            for (final c in calls.reversed)
              Text('• ${_describe(c)}', style: theme.textTheme.bodyMedium),
        ],
      ),
    );
  }
}

/// Post-match performance breakdown, shown once the match is over.
class _SummaryPanel extends StatelessWidget {
  const _SummaryPanel({
    required this.summary,
    required this.rallies,
    required this.movement,
    required this.positions,
    required this.netX,
    required this.placement,
    required this.maxBallSpeedKmh,
    required this.tracking,
    required this.reportText,
    required this.reportJson,
    required this.reportJsonMap,
    required this.historyStoreLoader,
    required this.shareReport,
  });

  final MatchSummary summary;

  /// Rally-length analytics (avg/longest strokes) over the match.
  final RallyStats rallies;

  /// Per-player footwork/positioning metrics mined from the pose model.
  final Map<Player, PlayerMovementStats> movement;

  /// Per-player foot-position samples, the raw material for the coverage map.
  final Map<Player, List<FramePoint>> positions;

  /// The calibrated net line (frame x), used to re-centre the coverage map.
  final double netX;

  /// Per-side bounce-placement / shot-map analytics from the tracker's bounces.
  final Map<TableSide, SidePlacementStats> placement;

  /// Fastest estimated ball speed (km/h), or null when no speed was measured.
  final double? maxBallSpeedKmh;

  /// Detection-health analytics — how reliably the phone placement tracked the
  /// ball and both players over the match.
  final TrackingQualityAnalyzer tracking;

  /// The full, shareable text report composed from every analytics layer,
  /// copied to the clipboard by the "Copy report" action.
  final String reportText;

  /// The same analytics as a machine-readable JSON string, copied to the
  /// clipboard by the "Export JSON" action for storage / integration.
  final String reportJson;

  /// The structured JSON report as a map, persisted by "Save to history".
  final Map<String, Object?> reportJsonMap;

  /// Resolves the store the "Save to history" action writes to.
  final Future<SessionHistoryStore> Function() historyStoreLoader;

  /// Hands the composed text report to the OS share sheet.
  final ShareReportSink shareReport;

  static String _name(Player p) => p == Player.a ? 'Player A' : 'Player B';

  Future<void> _saveToHistory(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    final store = await historyStoreLoader();
    await store.save(kind: SessionKind.match, report: reportJsonMap);
    messenger.showSnackBar(
      const SnackBar(content: Text('Saved to history')),
    );
  }

  Future<void> _copyReport(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    await Clipboard.setData(ClipboardData(text: reportText));
    messenger.showSnackBar(
      const SnackBar(content: Text('Report copied to clipboard')),
    );
  }

  Future<void> _shareReport(BuildContext context) async {
    await shareReport(reportText, subject: 'Table tennis match summary');
  }

  Future<void> _copyJson(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    await Clipboard.setData(ClipboardData(text: reportJson));
    messenger.showSnackBar(
      const SnackBar(content: Text('JSON summary copied to clipboard')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final winner = summary.matchWinner;
    return Container(
      width: double.infinity,
      color: theme.colorScheme.surfaceContainerHighest,
      padding: const EdgeInsets.all(16),
      child: SingleChildScrollView(
        child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            winner == null
                ? 'Match summary'
                : '${_name(winner)} wins the match',
            style: theme.textTheme.titleMedium,
          ),
          const SizedBox(height: 4),
          Text(
            '${summary.totalPoints} points played',
            style: theme.textTheme.bodyMedium,
          ),
          if (summary.gameScores.isNotEmpty)
            Text(
              'Games: ${summary.gameScores.join(', ')}',
              style: theme.textTheme.bodyMedium,
            ),
          if (summary.totalPoints > 0) ...[
            const SizedBox(height: 8),
            Text('Momentum', style: theme.textTheme.labelLarge),
            const SizedBox(height: 4),
            MomentumChartView(points: summary.points),
            Text(
              'Lead changes: ${summary.leadChanges}',
              style: theme.textTheme.bodyMedium,
            ),
            if (summary.decisiveRally case final r?)
              Text(
                'Lead taken for good at rally $r/${summary.totalPoints}',
                style: theme.textTheme.bodyMedium,
              ),
            const SizedBox(height: 8),
          ],
          if (rallies.rallyCount > 0)
            Text(
              'Rallies: avg ${rallies.averageStrokes.toStringAsFixed(1)} '
              'strokes, longest ${rallies.longestStrokes}',
              style: theme.textTheme.bodyMedium,
            ),
          if (rallies.hasWinData)
            Text(
              'Long-rally wins: '
              'A ${rallies.ralliesWonByLength(Player.a, RallyLength.long)} / '
              'B ${rallies.ralliesWonByLength(Player.b, RallyLength.long)}',
              style: theme.textTheme.bodyMedium,
            ),
          if (maxBallSpeedKmh != null)
            Text(
              'Top ball speed: ${maxBallSpeedKmh!.toStringAsFixed(1)} km/h',
              style: theme.textTheme.bodyMedium,
            ),
          if (tracking.hasData)
            Text(
              'Tracking quality: grade ${tracking.grade} '
              '(ball ${(tracking.ballDetectionRate * 100).round()}%, '
              'both players ${(tracking.twoPlayerRate * 100).round()}%)',
              style: theme.textTheme.bodyMedium,
            ),
          for (final side in TableSide.values)
            if (placement[side]!.count > 0)
              Text(
                '${side == TableSide.left ? 'Left' : 'Right'} bounces: '
                '${placement[side]!.shortCount} short / '
                '${placement[side]!.middleCount} mid / '
                '${placement[side]!.deepCount} deep',
                style: theme.textTheme.bodyMedium,
              ),
          if (placement[TableSide.left]!.count +
                  placement[TableSide.right]!.count >
              0) ...[
            const SizedBox(height: 8),
            Text('Shot map', style: theme.textTheme.labelLarge),
            const SizedBox(height: 4),
            ShotMapView(
              left: placement[TableSide.left]!,
              right: placement[TableSide.right]!,
            ),
          ],
          if (positions.values.any((p) => p.isNotEmpty)) ...[
            const SizedBox(height: 8),
            Text('Player coverage', style: theme.textTheme.labelLarge),
            const SizedBox(height: 4),
            PlayerPositionMapView(positions: positions, netX: netX),
          ],
          const SizedBox(height: 8),
          Row(
            children: [
              for (final player in Player.values)
                Expanded(
                  child: _PlayerStatColumn(
                    summary: summary,
                    player: player,
                    movement: movement[player],
                  ),
                ),
            ],
          ),
          if (MatchInsights(summary).hasData) ...[
            const SizedBox(height: 8),
            Text('Coaching', style: theme.textTheme.labelLarge),
            if (MatchInsights(summary).decisiveDimension case final d?)
              Text(
                'Match difference: ${_name(d.leader!)} won the '
                '${d.name.toLowerCase()} battle '
                '(${(d.scoreFor(d.leader!) * 100).round()}% vs '
                '${(d.scoreFor(d.leader!.other) * 100).round()}%).',
                style: theme.textTheme.bodyMedium,
              ),
            for (final player in Player.values)
              if (MatchInsights(summary).insightsFor(player) case final pi
                  when pi.focusTip != null)
                Text(
                  '${_name(player)} (Grade ${pi.grade}): ${pi.focusTip}',
                  style: theme.textTheme.bodyMedium,
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
                  onPressed: () => _copyJson(context),
                ),
                OutlinedButton.icon(
                  icon: const Icon(Icons.copy, size: 18),
                  label: const Text('Copy report'),
                  onPressed: () => _copyReport(context),
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

class _PlayerStatColumn extends StatelessWidget {
  const _PlayerStatColumn({
    required this.summary,
    required this.player,
    this.movement,
  });

  final MatchSummary summary;
  final Player player;

  /// Footwork/positioning metrics for this player, if the pose model tracked
  /// them; null/absent when they were never detected.
  final PlayerMovementStats? movement;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final m = movement;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          _SummaryPanel._name(player),
          style: theme.textTheme.labelLarge,
        ),
        Text('${summary.pointsWonBy(player)} pts won'),
        Text('${summary.forcedErrorsWonBy(player)} forced errors'),
        Text('${summary.openPlayPointsWonBy(player)} open play'),
        Text('longest run: ${summary.longestStreakFor(player)}'),
        Text('biggest lead: ${summary.largestLeadBy(player)}'),
        if (summary.largestDeficitOvercomeBy(player) case final d when d > 0)
          Text('comeback from $d down'),
        if (summary.serveWinRateFor(player) case final rate?)
          Text(
            'serve won: ${summary.servePointsWonBy(player)}/'
            '${summary.servePointsPlayedBy(player)} '
            '(${(rate * 100).round()}%)',
          ),
        if (summary.hasPressureData)
          Text(
            'game pts: ${summary.gamePointsConvertedBy(player)}/'
            '${summary.gamePointsHeldBy(player)} conv, '
            '${summary.gamePointsSavedBy(player)}/'
            '${summary.gamePointsFacedBy(player)} saved',
          ),
        if (m != null && m.wasTracked) ...[
          Text('moved: ${m.distanceTravelled.toStringAsFixed(2)}'),
          Text('mobility: ${m.mobilityPerSecond.toStringAsFixed(2)}/s'),
        ],
      ],
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
      color: Theme.of(context).colorScheme.surfaceContainerHighest,
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Who won this point?'),
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
