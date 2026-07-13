import 'dart:async';

import 'package:flutter/material.dart';

import '../../core/analysis/match_controller.dart';
import '../../core/analysis/match_summary.dart';
import '../../core/analysis/rally_referee.dart';
import '../../core/scoring/scoring_engine.dart';
import '../../core/vision/detection.dart';
import '../../core/vision/replay_vision_service.dart';
import '../../core/vision/synthetic_frames.dart';
import '../../core/vision/vision_service.dart';

/// Live match screen: streams vision frames through the [MatchController] and
/// renders the running score, the tracked ball, and the referee's calls.
///
/// The frame source is injectable so tests (and, later, the real
/// `ultralytics_yolo` camera runtime) can supply their own [VisionService]. It
/// defaults to a [ReplayVisionService] playing a scripted demo match so the
/// whole pipeline is visible in-app without a camera.
class MatchScreen extends StatefulWidget {
  const MatchScreen({super.key, this.visionServiceBuilder});

  /// Builds the frame source. Defaults to the scripted demo replay.
  final VisionService Function()? visionServiceBuilder;

  @override
  State<MatchScreen> createState() => _MatchScreenState();
}

class _MatchScreenState extends State<MatchScreen> {
  late final VisionService _vision;
  final MatchController _controller = MatchController();

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
              _SummaryPanel(summary: _controller.summary)
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
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
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
  const _SummaryPanel({required this.summary});

  final MatchSummary summary;

  static String _name(Player p) => p == Player.a ? 'Player A' : 'Player B';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final winner = summary.matchWinner;
    return Container(
      width: double.infinity,
      color: theme.colorScheme.surfaceContainerHighest,
      padding: const EdgeInsets.all(16),
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
          const SizedBox(height: 8),
          Row(
            children: [
              for (final player in Player.values)
                Expanded(child: _PlayerStatColumn(summary: summary, player: player)),
            ],
          ),
        ],
      ),
    );
  }
}

class _PlayerStatColumn extends StatelessWidget {
  const _PlayerStatColumn({required this.summary, required this.player});

  final MatchSummary summary;
  final Player player;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
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
