/// Generates the shipped **labeled** benchmark clip
/// (`benchmark/clips/synthetic_labeled.json`).
///
/// The original shipped fixture (`synthetic_demo.json`) carries only a scoring
/// ground truth, so running `dart run bin/benchmark.dart` on the corpus can only
/// exercise Stage 1 (scoring accuracy) — Stages 2 (perception) and 3
/// (event-detection) always print "No clips carry ground truth". This tool
/// produces a second fixture that additionally carries:
///
///   * `groundTruthFrames` — the *true* per-frame ball + player detections, so
///     [DetectionBenchmark] scores ball precision/recall and pose PCK against a
///     detector whose predicted `frames` intentionally drop one ball detection
///     (a realistic missed frame → ball recall < 100%).
///   * `groundTruthEvents` — the *true* net-cross and bounce timings, so
///     [EventDetectionBenchmark] scores the tracker's event timing.
///
/// The clip is three rallies (L→R, R→L, L→R), each a net-crossing arc that
/// bounces on the destination side then loses the ball — a `notReturned` point
/// against the receiver, scoring A-B = 2-1.
///
/// To keep the fixture *self-consistent*, this generator replays both the
/// predicted and the ground-truth frames through the real [BallTracker] /
/// [MatchController] and asserts the emitted events and the scored outcome match
/// the declared ground truth before writing the JSON. Re-run with
/// `dart run tool/gen_labeled_clip.dart` after changing the shape.
library;

import 'dart:convert';
import 'dart:io';

import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/benchmark/benchmark_runner.dart';
import 'package:pong_ai/core/benchmark/clip_fixture.dart';
import 'package:pong_ai/core/benchmark/event_metrics.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';

const int _stepMs = 33;

/// A tiny ball detection centred at ([x], [y]).
Detection _ball(double x, double y) => Detection(
      label: 'ball',
      confidence: 0.9,
      box: BBox(x - 0.01, y - 0.01, 0.02, 0.02),
    );

/// A static side-on player: a narrow, tall box with two ankle keypoints. The
/// left player sits near x=0.2, the right near x=0.8.
PersonPose _player(double centerX, int trackId) => PersonPose(
      box: BBox(centerX - 0.05, 0.30, 0.10, 0.45),
      trackId: trackId,
      keypoints: [
        Keypoint(centerX - 0.02, 0.73, 0.9), // left ankle
        Keypoint(centerX + 0.02, 0.73, 0.9), // right ankle
      ],
    );

final List<PersonPose> _players = [_player(0.20, 1), _player(0.80, 2)];

/// The per-frame samples of one net-crossing bounce rally, source → dest.
///
/// x sweeps across the net (crossing between the 2nd and 3rd sample), y traces a
/// down-up arc whose apex (highest y) is the bounce on the destination side.
/// Returns the ball (x,y) samples; the caller stamps timestamps and appends the
/// trailing ball-lost gap.
List<({double x, double y})> _rallyArc({required bool leftToRight}) {
  const ys = <double>[0.45, 0.52, 0.60, 0.66, 0.58, 0.50];
  const xsLtr = <double>[0.35, 0.45, 0.55, 0.65, 0.72, 0.75];
  final xs = leftToRight
      ? xsLtr
      : xsLtr.map((x) => 1.0 - x).toList(growable: false);
  return [for (var i = 0; i < ys.length; i++) (x: xs[i], y: ys[i])];
}

void main() {
  // Rally directions: L→R (bounce right → A), R→L (bounce left → B),
  // L→R (bounce right → A) ⇒ final A-B = 2-1, winners [A, B, A].
  const directions = <bool>[true, false, true];

  final truthFrames = <FrameResult>[];
  final predictedFrames = <FrameResult>[];
  final events = <GroundTruthEvent>[];

  var t = 0;
  for (final leftToRight in directions) {
    final arc = _rallyArc(leftToRight: leftToRight);
    final arcStart = t;
    for (var i = 0; i < arc.length; i++) {
      final ts = arcStart + i * _stepMs;
      final ball = _ball(arc[i].x, arc[i].y);
      truthFrames.add(
        FrameResult(timestampMs: ts, ball: ball, people: _players),
      );
      // Predicted frames match ground truth, except the detector *misses* the
      // trailing ball detection (i == 5) of every rally — a realistic dropped
      // frame after the events have fired, so ball recall < 100% while the
      // net-cross/bounce timings and the score are unaffected.
      final dropped = i == arc.length - 1;
      predictedFrames.add(
        FrameResult(
          timestampMs: ts,
          ball: dropped ? null : ball,
          people: _players,
        ),
      );
    }
    // True events for this rally: net-cross at the 3rd sample (first on the new
    // side) and the bounce at the apex (4th sample, highest y).
    events
      ..add(GroundTruthEvent(arcStart + 2 * _stepMs, TrackedEventType.netCross))
      ..add(GroundTruthEvent(arcStart + 3 * _stepMs, TrackedEventType.bounce));

    t = arcStart + arc.length * _stepMs;
    // Ball-lost gap between rallies (both streams): enough empties to exceed the
    // tracker's maxGapFrames so the trajectory resets before the next rally.
    for (var i = 0; i < 8; i++) {
      truthFrames.add(FrameResult(timestampMs: t, people: _players));
      predictedFrames.add(FrameResult(timestampMs: t, people: _players));
      t += _stepMs;
    }
  }

  const groundTruth = ClipGroundTruth(
    pointsA: 2,
    pointsB: 1,
    pointWinners: [Player.a, Player.b, Player.a],
  );

  final clip = ClipFixture(
    name: 'synthetic_labeled_2_1',
    source: 'synthetic',
    frames: predictedFrames,
    groundTruthFrames: truthFrames,
    groundTruthEvents: events,
    groundTruth: groundTruth,
  );

  _verify(clip);

  const encoder = JsonEncoder.withIndent('  ');
  final out = File('benchmark/clips/synthetic_labeled.json');
  out.writeAsStringSync('${encoder.convert(clip.toJson())}\n');
  stdout.writeln('Wrote ${out.path} '
      '(${predictedFrames.length} frames, ${events.length} events).');
}

/// Replays the clip through the real tracker/pipeline and asserts the declared
/// ground truth is self-consistent, so the shipped fixture can never drift from
/// what the code actually produces.
void _verify(ClipFixture clip) {
  // 1. Scoring: the predicted frames must reproduce the declared A-B score.
  final result = const BenchmarkRunner().run(clip);
  _check(
    result.finalScoreCorrect,
    'score mismatch: detected ${result.detectedPointsA}-'
    '${result.detectedPointsB}, truth ${clip.groundTruth.pointsA}-'
    '${clip.groundTruth.pointsB}',
  );
  _check(
    result.orderedAccuracy == 1.0,
    'ordered winners mismatch (${result.orderedMatches}/'
    '${result.orderedComparable})',
  );

  // 2. Events: the tracker must emit every declared ground-truth event on the
  //    predicted frames within tolerance (100% precision & recall).
  final eventResult = const EventDetectionBenchmark().evaluate(
    name: clip.name,
    frames: clip.frames,
    groundTruth: clip.groundTruthEvents!,
    geometry: TableGeometry(netX: clip.netX),
  );
  _check(
    eventResult.bounce.recall == 1.0 && eventResult.bounce.precision == 1.0,
    'bounce events imperfect: tp ${eventResult.bounce.truePositives} '
    'fp ${eventResult.bounce.falsePositives} '
    'fn ${eventResult.bounce.falseNegatives}',
  );
  _check(
    eventResult.netCross.recall == 1.0 && eventResult.netCross.precision == 1.0,
    'net-cross events imperfect',
  );

  stdout.writeln('Verified: score 2-1, '
      '${clip.groundTruthEvents!.length} events all matched.');
}

void _check(bool ok, String message) {
  if (!ok) {
    stderr.writeln('GENERATION FAILED: $message');
    exit(1);
  }
}
