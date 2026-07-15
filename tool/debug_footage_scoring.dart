/// Replays the bundled footage fixture through the footage-mode scoring
/// pipeline and logs every point decision with its timestamp and reason, so
/// scoring behaviour on the real clip can be diagnosed offline.
///
/// Usage:
///   dart run tool/debug_footage_scoring.dart \
///     [maxGapFrames] [maxJump] [minBounceSpeed] [postPointCooldown] [fixture.json]
///
/// Any argument ending in `.json` selects the fixture (default: the bundled
/// main demo); the numeric arguments are positional.
library;

import 'dart:convert';
import 'dart:io';

import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/benchmark/clip_fixture.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';

void main(List<String> rawArgs) {
  final path = rawArgs.firstWhere(
    (a) => a.endsWith('.json'),
    orElse: () => 'assets/footage/openttgames_test2.json',
  );
  final args = rawArgs.where((a) => !a.endsWith('.json')).toList();
  final maxGapFrames = args.isNotEmpty ? int.parse(args[0]) : 6;
  final maxJump = args.length > 1 ? double.parse(args[1]) : null;
  final minBounceSpeed = args.length > 2 ? double.parse(args[2]) : 0.004;

  final fixture = ClipFixture.fromJson(
    jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>,
  );

  final controller = MatchController(
    tracker: BallTracker(
      geometry: fixture.geometry,
      maxGapFrames: maxGapFrames,
      maxJump: maxJump,
      minBounceSpeed: minBounceSpeed,
      netBounceExclusion: 0.03,
      netCrossHysteresis: 0.03,
      extendedGapFrames: 60,
    ),
    referee: RallyReferee(
      leftPlayer: fixture.leftPlayer,
      requireServe: true,
      doubleBounceGraceMs: 500,
      staleEventMs: 1200,
    ),
    engine: ScoringEngine(
      firstServer: fixture.firstServer,
      pointsPerGame: fixture.pointsPerGame,
      bestOf: fixture.bestOf,
    ),
    postPointCooldown: args.length > 3 ? int.parse(args[3]) : 0,
  );

  var covered = 0;
  void log(List<PointDecision> decisions, int timestampMs, {String tag = ''}) {
    for (final d in decisions) {
      final t = (timestampMs / 1000).toStringAsFixed(1);
      stdout.writeln(
        't=${t}s$tag  ${d.winner?.name.toUpperCase() ?? 'UNDETERMINED'} '
        '(${d.reason.name})  score now '
        '${controller.score.pointsA}-${controller.score.pointsB}',
      );
    }
  }

  for (final frame in fixture.frames) {
    if (frame.ball != null) covered++;
    log(controller.onFrame(frame), frame.timestampMs);
  }
  // The clip is over: resolve any rally still in flight, as the app does.
  log(
    controller.finishPlay(),
    fixture.frames.isEmpty ? 0 : fixture.frames.last.timestampMs,
    tag: ' (end flush)',
  );
  stdout
    ..writeln('---')
    ..writeln('ball frames: $covered/${fixture.frames.length}')
    ..writeln('final: ${controller.score.pointsA}-${controller.score.pointsB} '
        '(games ${controller.score.gamesA}-${controller.score.gamesB})')
    ..writeln('undetermined pending: ${controller.undetermined.length}');
}
