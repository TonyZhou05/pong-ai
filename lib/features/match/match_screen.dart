import 'dart:async';

import 'package:flutter/material.dart';

import '../../core/analysis/match_controller.dart';
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
            Expanded(child: _TableView(frame: _lastFrame)),
            if (pending.isNotEmpty)
              _UndeterminedPrompt(
                decision: pending.first,
                onPick: (winner) => _resolve(pending.first, winner),
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
  const _TableView({required this.frame});

  final FrameResult? frame;

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
