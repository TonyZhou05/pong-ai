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
import '../../core/benchmark/clip_fixture.dart';
import '../../core/history/history_store_provider.dart';
import '../../core/history/session_history_store.dart';
import '../../core/share/report_share.dart';
import '../../core/scoring/match_situation.dart';
import '../../core/scoring/scoring_engine.dart';
import '../../core/vision/detection.dart';
import '../../core/vision/position_synced_vision_service.dart';
import '../../core/vision/replay_vision_service.dart';
import '../../core/vision/synthetic_frames.dart';
import '../../core/vision/vision_service.dart';
import '../summary/momentum_chart.dart';
import '../summary/player_map.dart';
import '../summary/shot_map.dart';
import 'footage_demo.dart';

/// Live match screen: streams vision frames through the [MatchController] and
/// renders the running score, the tracked ball, and the referee's calls.
///
/// The frame source is injectable so tests (and, later, the real
/// `ultralytics_yolo` camera runtime) can supply their own [VisionService]. It
/// defaults to a [ReplayVisionService] playing a scripted demo match so the
/// whole pipeline is visible in-app without a camera.
///
/// When a [footage] demo is supplied the screen instead plays a *real*
/// side-recorded match video and overlays the recorded per-frame player/ball
/// identification on the actual pixels, replaying the detections in lock-step
/// with the video's playback clock (a [PositionSyncedVisionService]) so the
/// overlays — and the scoring pipeline consuming the same frames — can never
/// drift from the footage.
class MatchScreen extends StatefulWidget {
  const MatchScreen({
    super.key,
    this.footage,
    this.footagePlayerBuilder,
    this.footageFixtureLoader = loadFootageFixture,
    this.visionServiceBuilder,
    this.matchControllerBuilder,
    this.historyStoreLoader = defaultSessionHistoryStore,
    this.shareReport = defaultShareReport,
  });

  /// When set, the screen shows this real recorded clip with detection
  /// overlays instead of the schematic synthetic-replay table.
  final FootageDemo? footage;

  /// Builds the video playback surface for [footage]. Defaults to the real
  /// `video_player`-backed [VideoFootagePlayer]; tests inject a fake whose
  /// position the test drives (no platform channel headlessly).
  final FootagePlayer Function()? footagePlayerBuilder;

  /// Loads the [footage] detection-track fixture. Defaults to the asset
  /// bundle; tests inject an in-memory fixture.
  final Future<ClipFixture> Function(String asset) footageFixtureLoader;

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
  late VisionService _vision;
  late MatchController _controller;

  /// Whether [_vision]/[_controller] are initialized. Immediate on the
  /// synthetic path; set once the async footage load completes on the footage
  /// path (dispose must not touch the late fields before then).
  bool _ready = false;

  /// Footage-mode state: the video playback seam and the loaded detection
  /// track (kept so Replay can rebuild a fresh scoring pipeline).
  FootagePlayer? _player;
  ClipFixture? _fixture;
  Object? _footageError;

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
    if (widget.footage != null) {
      _initFootage();
      return;
    }
    _controller = widget.matchControllerBuilder?.call() ?? MatchController();
    _vision = widget.visionServiceBuilder?.call() ??
        ReplayVisionService(demoMatchFrames());
    _ready = true;
    _startVision();
  }

  /// Builds the scoring pipeline configured for the recorded clip, tuned for a
  /// *sparse* recorded detection track (real footage tracks the ball in only a
  /// fraction of frames, unlike the dense synthetic clips):
  ///
  ///  * the fixture's full [ClipFixture.geometry] (net line + table-surface
  ///    band), so a direction change beyond the table's edge — a paddle hit,
  ///    the ball sailing out — can't register as a bounce and mis-award the
  ///    point;
  ///  * `maxGapFrames: 30` (~1 s): mid-rally detection gaps are routine in a
  ///    sparse track, and the default 6 turns each one into a phantom
  ///    rally-ending ball-loss (the score visibly incrementing mid-rally);
  ///  * default `minBounceSpeed`: with the table band gating off-table
  ///    reversals, the default threshold is right — raising it was measured
  ///    to *miss real bounces* whose 30 fps sampling lands near the apex
  ///    (flattening the incoming Δy), which then mis-arms the referee's
  ///    out-of-bounds inference on the next crossing pair;
  ///  * `netBounceExclusion: 0.03`: a y-reversal at the net plane is the net
  ///    interfering (a clip, or the sampled trajectory kinking as it crosses),
  ///    not a table landing — verified against the dataset's labeled bounces;
  ///  * no `maxJump` gate: after a long gap the Kalman prediction has drifted,
  ///    so the gate would reject the *real* ball on reappearance and starve
  ///    the tracker into a bogus ball-loss (the live camera path keeps it —
  ///    its detections arrive every frame, where the gate's assumption holds);
  ///  * `postPointCooldown: 15` (~0.5 s): after a point the ball keeps
  ///    bouncing/rolling; without a cool-down those leftovers seed a phantom
  ///    rally that fizzles into a spurious "who won?" prompt;
  ///  * `extendedGapFrames: 60` (~2 s): a lob arcing out the top of the frame
  ///    or a player stepping off-frame to play it must not be scored as a
  ///    rally-ending ball loss while the point is still live;
  ///  * `requireServe`: rally activity only counts once a serve visibly
  ///    initiates it, so players knocking the ball to each other between
  ///    points can't inflate the counters or fizzle into bogus decisions.
  MatchController _footageController(ClipFixture clip) => MatchController(
        tracker: BallTracker(
          geometry: clip.geometry,
          maxGapFrames: 30,
          netBounceExclusion: 0.03,
          netCrossHysteresis: 0.03,
          extendedGapFrames: 60,
        ),
        referee: RallyReferee(
          leftPlayer: clip.leftPlayer,
          requireServe: true,
          // Dense-track tuning: a half-volley pickup looks like a second
          // same-side bounce (grace window cancels it on the return
          // crossing), and a bounce/cross >1.2s after the previous event is
          // dead-ball motion, not the same exchange.
          doubleBounceGraceMs: 500,
          staleEventMs: 1200,
        ),
        engine: ScoringEngine(
          firstServer: clip.firstServer,
          pointsPerGame: clip.pointsPerGame,
          bestOf: clip.bestOf,
        ),
        postPointCooldown: 15,
      );

  Future<void> _initFootage() async {
    final demo = widget.footage!;
    FootagePlayer? player;
    try {
      final fixture = await widget.footageFixtureLoader(demo.fixtureAsset);
      player = widget.footagePlayerBuilder?.call() ??
          VideoFootagePlayer(demo.videoAsset);
      await player.initialize();
      if (!mounted) {
        await player.dispose();
        return;
      }
      final readyPlayer = player; // non-null from here on
      _fixture = fixture;
      _player = readyPlayer;
      _controller =
          widget.matchControllerBuilder?.call() ?? _footageController(fixture);
      _vision = PositionSyncedVisionService(
        fixture.frames,
        positionMs: () => readyPlayer.position.inMilliseconds,
      );
      setState(() => _ready = true);
      await _startVision();
      await readyPlayer.play();
    } catch (e) {
      // Initialization failed (e.g. no video runtime); release the player if
      // the screen never took ownership of it.
      if (!identical(player, _player)) await player?.dispose();
      if (mounted) setState(() => _footageError = e);
    }
  }

  Future<void> _togglePlayback() async {
    final player = _player;
    if (player == null) return;
    if (player.isPlaying) {
      await player.pause();
    } else {
      await player.play();
    }
    if (mounted) setState(() {});
  }

  /// Rewinds the footage and replays it through a *fresh* scoring pipeline, so
  /// a second viewing doesn't double-score the same rallies.
  Future<void> _replayFootage() async {
    final player = _player;
    final fixture = _fixture;
    if (player == null || fixture == null) return;
    await _vision.stop();
    await player.pause();
    await player.seekToStart();
    // On some platforms (web in particular) the seek is applied
    // asynchronously: the player keeps *reporting* the old end-of-clip
    // position for a few frames. Restarting the position-synced replay
    // against that stale clock would instantly flush every frame into the
    // fresh controller — an instant bogus score and a dead overlay for the
    // whole second viewing. Wait for the rewind to actually land first.
    for (var i = 0;
        i < 40 && player.position > const Duration(milliseconds: 100);
        i++) {
      await Future<void>.delayed(const Duration(milliseconds: 25));
    }
    if (!mounted) return;
    setState(() {
      _controller =
          widget.matchControllerBuilder?.call() ?? _footageController(fixture);
      _recentCalls.clear();
      _lastFrame = null;
      _predictedBall = null;
    });
    await _vision.start();
    await player.play();
  }

  Future<void> _startVision() async {
    await _vision.load();
    _sub = _vision.frames.listen(_onFrame);
    await _vision.start();
  }

  void _onFrame(FrameResult frame) {
    var decisions = _controller.onFrame(frame);
    // Capture the overlay's ghost estimate before any end-of-footage flush
    // resets the tracker, so the final frame still draws its prediction.
    final predictedBall = frame.ball == null
        ? _controller.tracker.estimateBallAt(frame.timestampMs)
        : null;
    // This is the footage's final detection frame: a rally still in flight
    // can never continue, so let the referee resolve it now instead of
    // leaving the clip's last rally forever unscored. (Compared against the
    // frame itself — not the replay service's cursor, which can already be
    // exhausted while earlier frames are still being delivered.)
    final lastT = _fixture?.frames.isEmpty ?? true
        ? null
        : _fixture!.frames.last.timestampMs;
    if (_player != null && lastT != null && frame.timestampMs >= lastT) {
      final flushed = _controller.finishPlay();
      if (flushed.isNotEmpty) {
        decisions = [...decisions, ...flushed];
      }
    }
    if (!mounted) return;
    setState(() {
      _lastFrame = frame;
      _predictedBall = predictedBall;
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
    setState(() {
      _controller.resolveUndetermined(decision, winner);
      // Reflect the award in the call feed too — the entry was logged as
      // "Undetermined" when the referee surfaced it.
      final i = _recentCalls.indexOf(decision);
      if (i != -1) {
        _recentCalls[i] = PointDecision(
          winner: winner,
          reason: decision.reason,
          timestampMs: decision.timestampMs,
        );
      }
    });
  }

  void _undo() {
    if (_controller.undo()) setState(() {});
  }

  @override
  void dispose() {
    _sub?.cancel();
    if (_ready) _vision.dispose();
    _player?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) {
      return Scaffold(
        appBar: AppBar(title: const Text('Match')),
        body: Center(
          child: _footageError == null
              ? const CircularProgressIndicator()
              : Padding(
                  padding: const EdgeInsets.all(24),
                  child: Text('Could not load footage: $_footageError'),
                ),
        ),
      );
    }
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
            _BounceCounter(
              rally: _controller.currentRallyBounces,
              total: _controller.bounceCount,
            ),
            const SizedBox(height: 8),
            Expanded(
              child: _player == null
                  ? _TableView(
                      frame: _lastFrame,
                      predictedBall: _predictedBall,
                    )
                  : _FootageView(
                      player: _player!,
                      frame: _lastFrame,
                      predictedBall: _predictedBall,
                      netX: _controller.geometry.netX,
                      onTogglePlay: _togglePlayback,
                      onReplay: _replayFootage,
                    ),
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

/// Live table-bounce readout: how many times the ball has bounced on the table
/// in the rally being played right now, and across the whole match — the
/// "rally count" companion to the scoreboard.
class _BounceCounter extends StatelessWidget {
  const _BounceCounter({required this.rally, required this.total});

  /// Bounces in the current (in-flight) rally.
  final int rally;

  /// Total table bounces this match.
  final int total;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.sports_baseball,
            size: 14,
            color: theme.colorScheme.onSurfaceVariant,
          ),
          const SizedBox(width: 4),
          Text(
            'Bounces — rally: $rally · match: $total',
            key: const ValueKey('bounceCounter'),
            style: theme.textTheme.labelMedium
                ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
          ),
        ],
      ),
    );
  }
}

/// Real recorded footage with the app's identification overlaid: each frame's
/// player boxes (labelled by side) with pose keypoints, the detected ball (or
/// the dimmed Kalman-predicted ghost through dropouts), and the calibrated net
/// line — drawn in the same normalized [0,1] space the detections use, so the
/// overlay sits exactly on the players/ball in the video.
class _FootageView extends StatelessWidget {
  const _FootageView({
    required this.player,
    required this.frame,
    required this.predictedBall,
    required this.netX,
    required this.onTogglePlay,
    required this.onReplay,
  });

  final FootagePlayer player;
  final FrameResult? frame;

  /// Kalman-extrapolated ball position shown (dimmed) when the detector lost
  /// the ball this frame.
  final ({double x, double y})? predictedBall;

  /// The clip's calibrated net line (normalized frame x).
  final double netX;

  final Future<void> Function() onTogglePlay;
  final Future<void> Function() onReplay;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Center(
        child: AspectRatio(
          aspectRatio: player.aspectRatio,
          child: LayoutBuilder(
            builder: (context, constraints) {
              final w = constraints.maxWidth;
              final h = constraints.maxHeight;
              final ball = frame?.ball;
              final ghost = ball == null ? predictedBall : null;
              final people = frame?.people ?? const <PersonPose>[];
              return ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    player.view,
                    // Calibrated net line (from the clip's net-event geometry).
                    Positioned(
                      left: netX * w - 1,
                      top: 0,
                      bottom: 0,
                      child: Container(width: 2, color: Colors.white24),
                    ),
                    for (final (i, p) in people.indexed) ...[
                      Positioned(
                        key: ValueKey('footagePerson$i'),
                        left: p.box.left * w,
                        top: p.box.top * h,
                        width: p.box.width * w,
                        height: p.box.height * h,
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: theme.colorScheme.tertiary,
                              width: 2,
                            ),
                          ),
                        ),
                      ),
                      Positioned(
                        left: p.box.left * w,
                        top: (p.box.top * h - 18).clamp(0.0, h),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 4,
                            vertical: 1,
                          ),
                          color: theme.colorScheme.tertiary,
                          child: Text(
                            'Player ${p.box.centerX < netX ? 'A' : 'B'}',
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: theme.colorScheme.onTertiary,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                      // Pose keypoints — the "identification" of the player,
                      // not just a box.
                      for (final k in p.keypoints)
                        if (k.confidence > 0.5)
                          Positioned(
                            left: k.x * w - 2,
                            top: k.y * h - 2,
                            child: Container(
                              width: 4,
                              height: 4,
                              decoration: BoxDecoration(
                                color: theme.colorScheme.tertiary
                                    .withValues(alpha: 0.9),
                                shape: BoxShape.circle,
                              ),
                            ),
                          ),
                    ],
                    if (ball != null)
                      Positioned(
                        key: const ValueKey('footageBall'),
                        left: ball.box.centerX * w - 7,
                        top: ball.box.centerY * h - 7,
                        child: Container(
                          width: 14,
                          height: 14,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: const Color(0xFFFFEB3B),
                              width: 2,
                            ),
                          ),
                        ),
                      )
                    else if (ghost != null)
                      Positioned(
                        key: const ValueKey('footageGhost'),
                        left: ghost.x * w - 7,
                        top: ghost.y * h - 7,
                        child: Container(
                          width: 14,
                          height: 14,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(color: const Color(0x88FFEB3B)),
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
                    Positioned(
                      right: 4,
                      bottom: 4,
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          IconButton(
                            tooltip: 'Replay footage',
                            icon: const Icon(Icons.replay),
                            color: Colors.white,
                            onPressed: onReplay,
                          ),
                          IconButton(
                            tooltip: player.isPlaying ? 'Pause' : 'Play',
                            icon: Icon(
                              player.isPlaying
                                  ? Icons.pause_circle
                                  : Icons.play_circle,
                            ),
                            color: Colors.white,
                            onPressed: onTogglePlay,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
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
      PointReason.outOfBounds => 'out of bounds',
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
