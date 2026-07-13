/// Per-frame detection & pose accuracy metrics for the benchmark harness.
///
/// The scoring benchmark ([BenchmarkRunner]) answers "did we get the score
/// right?" — but the objective explicitly prioritises *finding a good model
/// that tracks the players (and ball) well*, which is a per-frame perception
/// question the score alone can't isolate. This module implements the two
/// perception rows of the metrics table in docs/ARCHITECTURE.md:
///
///   * **Ball detection** — precision / recall / F1 at a loose IoU threshold
///     (the 40 mm ball is a tiny, motion-blurred object, so IoU 0.3 is used),
///     plus mean IoU and centre error on matched frames.
///   * **Player pose** — person detection rate and, on matched people, PCK
///     (fraction of keypoints within a normalized distance) and mean keypoint
///     error.
///
/// It compares two aligned streams of [FrameResult]s: the pipeline's
/// *predicted* frames and the clip's per-frame *ground truth*. Both are the
/// same runtime-agnostic model the live plugin emits, so this scores real model
/// output the moment an annotated clip (with ground-truth frames) is dropped in
/// — no camera or device required.
library;

import 'dart:math' as math;

import '../vision/detection.dart';
import 'clip_fixture.dart';

/// Intersection-over-union of two normalized [BBox]es. 0 when disjoint or
/// degenerate.
double boxIoU(BBox a, BBox b) {
  final ax2 = a.left + a.width;
  final ay2 = a.top + a.height;
  final bx2 = b.left + b.width;
  final by2 = b.top + b.height;

  final ix1 = math.max(a.left, b.left);
  final iy1 = math.max(a.top, b.top);
  final ix2 = math.min(ax2, bx2);
  final iy2 = math.min(ay2, by2);

  final iw = ix2 - ix1;
  final ih = iy2 - iy1;
  if (iw <= 0 || ih <= 0) return 0;

  final intersection = iw * ih;
  final union = a.width * a.height + b.width * b.height - intersection;
  if (union <= 0) return 0;
  return intersection / union;
}

/// Euclidean distance between two normalized box centres.
double _centerDistance(BBox a, BBox b) {
  final dx = a.centerX - b.centerX;
  final dy = a.centerY - b.centerY;
  return math.sqrt(dx * dx + dy * dy);
}

/// Thresholds controlling how predictions are matched to ground truth.
class DetectionBenchmarkConfig {
  const DetectionBenchmarkConfig({
    this.ballIoUThreshold = 0.3,
    this.personIoUThreshold = 0.5,
    this.keypointThreshold = 0.05,
  });

  /// Minimum IoU for a predicted ball to count as a true positive. Loose (0.3)
  /// because the ball is tiny and blurred — a near-miss is still a good track.
  final double ballIoUThreshold;

  /// Minimum box IoU to match a predicted person to a ground-truth person.
  final double personIoUThreshold;

  /// Max normalized centre distance for a keypoint to count as correct (PCK).
  final double keypointThreshold;
}

/// Ball-detection accuracy accumulated over a clip.
class BallDetectionMetrics {
  const BallDetectionMetrics({
    required this.truePositives,
    required this.falsePositives,
    required this.falseNegatives,
    required this.iouSum,
    required this.centerErrorSum,
  });

  /// Frames where a ground-truth ball was matched by a prediction (IoU ≥ thr).
  final int truePositives;

  /// Frames where the pipeline reported a ball but ground truth had none (or no
  /// prediction cleared the IoU threshold).
  final int falsePositives;

  /// Frames where ground truth had a ball the pipeline missed.
  final int falseNegatives;

  /// Sum of matched-frame IoU / centre error, for the means below.
  final double iouSum;
  final double centerErrorSum;

  int get groundTruthCount => truePositives + falseNegatives;
  int get predictedCount => truePositives + falsePositives;

  double get precision =>
      predictedCount == 0 ? 1.0 : truePositives / predictedCount;

  double get recall =>
      groundTruthCount == 0 ? 1.0 : truePositives / groundTruthCount;

  double get f1 {
    final p = precision;
    final r = recall;
    return (p + r) == 0 ? 0.0 : 2 * p * r / (p + r);
  }

  double get meanIoU => truePositives == 0 ? 0.0 : iouSum / truePositives;

  double get meanCenterError =>
      truePositives == 0 ? 0.0 : centerErrorSum / truePositives;

  /// Fraction of ground-truth-ball frames the pipeline tracked (== recall);
  /// named for the "% frames tracked through blur" metric.
  double get trackedFraction => recall;
}

/// Player pose accuracy accumulated over a clip.
class PoseDetectionMetrics {
  const PoseDetectionMetrics({
    required this.matchedPeople,
    required this.missedPeople,
    required this.spuriousPeople,
    required this.boxIoUSum,
    required this.correctKeypoints,
    required this.evaluatedKeypoints,
    required this.keypointErrorSum,
  });

  /// Ground-truth people matched to a prediction.
  final int matchedPeople;

  /// Ground-truth people with no matching prediction (person-level miss).
  final int missedPeople;

  /// Predicted people that matched no ground-truth person (spurious box).
  final int spuriousPeople;

  final double boxIoUSum;

  /// Labeled (ground-truth-visible) keypoints on matched people that landed
  /// within [DetectionBenchmarkConfig.keypointThreshold].
  final int correctKeypoints;

  /// Labeled keypoints evaluated on matched people (the PCK denominator).
  final int evaluatedKeypoints;

  final double keypointErrorSum;

  int get groundTruthPeople => matchedPeople + missedPeople;

  /// Fraction of ground-truth people that were detected (matched).
  double get detectionRate =>
      groundTruthPeople == 0 ? 1.0 : matchedPeople / groundTruthPeople;

  double get meanBoxIoU =>
      matchedPeople == 0 ? 0.0 : boxIoUSum / matchedPeople;

  /// Percentage of Correct Keypoints on matched people.
  double get pck =>
      evaluatedKeypoints == 0 ? 0.0 : correctKeypoints / evaluatedKeypoints;

  double get meanKeypointError =>
      evaluatedKeypoints == 0 ? 0.0 : keypointErrorSum / evaluatedKeypoints;
}

/// Combined per-clip perception metrics.
class DetectionBenchmarkResult {
  const DetectionBenchmarkResult({
    required this.clipName,
    required this.frameCount,
    required this.ball,
    required this.pose,
  });

  final String clipName;
  final int frameCount;
  final BallDetectionMetrics ball;
  final PoseDetectionMetrics pose;

  String report() {
    final b = ball;
    final p = pose;
    return (StringBuffer()
          ..writeln('Perception: $clipName  ($frameCount frame(s))')
          ..writeln(
            '  Ball  P/R/F1: ${_pct(b.precision)}/${_pct(b.recall)}/'
            '${_pct(b.f1)}  meanIoU ${b.meanIoU.toStringAsFixed(2)}  '
            '(tp ${b.truePositives}, fp ${b.falsePositives}, '
            'fn ${b.falseNegatives})',
          )
          ..writeln(
            '  Pose  detRate ${_pct(p.detectionRate)}  PCK ${_pct(p.pck)}  '
            'meanKpErr ${p.meanKeypointError.toStringAsFixed(3)}  '
            '(matched ${p.matchedPeople}, missed ${p.missedPeople}, '
            'spurious ${p.spuriousPeople})',
          ))
        .toString();
  }

  static String _pct(double v) => '${(v * 100).toStringAsFixed(1)}%';
}

/// Scores predicted [FrameResult]s against per-frame ground-truth frames.
class DetectionBenchmark {
  const DetectionBenchmark({
    this.config = const DetectionBenchmarkConfig(),
  });

  final DetectionBenchmarkConfig config;

  /// Evaluate a clip that carries per-frame ground truth, scoring its replayed
  /// [ClipFixture.frames] (predictions) against [ClipFixture.groundTruthFrames].
  /// Returns null when the clip has no per-frame ground truth to score against.
  DetectionBenchmarkResult? evaluateClip(ClipFixture clip) {
    final gt = clip.groundTruthFrames;
    if (gt == null) return null;
    return evaluate(
      name: clip.name,
      predicted: clip.frames,
      groundTruth: gt,
    );
  }

  /// Evaluate index-aligned [predicted] vs [groundTruth] frame streams. Streams
  /// of unequal length are compared over the longer one, with the shorter side
  /// treated as empty frames (so extra predictions become false positives and
  /// missing predictions become misses).
  DetectionBenchmarkResult evaluate({
    required String name,
    required List<FrameResult> predicted,
    required List<FrameResult> groundTruth,
  }) {
    var ballTp = 0;
    var ballFp = 0;
    var ballFn = 0;
    var ballIoUSum = 0.0;
    var ballCenterErrSum = 0.0;

    var matchedPeople = 0;
    var missedPeople = 0;
    var spuriousPeople = 0;
    var boxIoUSum = 0.0;
    var correctKp = 0;
    var evaluatedKp = 0;
    var kpErrSum = 0.0;

    final n = math.max(predicted.length, groundTruth.length);
    for (var i = 0; i < n; i++) {
      final pred = i < predicted.length ? predicted[i] : null;
      final gt = i < groundTruth.length ? groundTruth[i] : null;

      // --- Ball ---
      final predBall = pred?.ball;
      final gtBall = gt?.ball;
      if (gtBall != null && predBall != null) {
        final iou = boxIoU(predBall.box, gtBall.box);
        if (iou >= config.ballIoUThreshold) {
          ballTp++;
          ballIoUSum += iou;
          ballCenterErrSum += _centerDistance(predBall.box, gtBall.box);
        } else {
          // A detection that doesn't overlap the true ball: a miss and a
          // spurious box both.
          ballFp++;
          ballFn++;
        }
      } else if (gtBall != null) {
        ballFn++;
      } else if (predBall != null) {
        ballFp++;
      }

      // --- Pose ---
      final predPeople = pred?.people ?? const <PersonPose>[];
      final gtPeople = gt?.people ?? const <PersonPose>[];
      final used = <int>{};
      for (final gtP in gtPeople) {
        var best = -1;
        var bestIoU = config.personIoUThreshold;
        for (var j = 0; j < predPeople.length; j++) {
          if (used.contains(j)) continue;
          final iou = boxIoU(gtP.box, predPeople[j].box);
          if (iou >= bestIoU) {
            bestIoU = iou;
            best = j;
          }
        }
        if (best < 0) {
          missedPeople++;
          continue;
        }
        used.add(best);
        matchedPeople++;
        boxIoUSum += boxIoU(gtP.box, predPeople[best].box);
        final kp = _scoreKeypoints(gtP, predPeople[best]);
        correctKp += kp.correct;
        evaluatedKp += kp.evaluated;
        kpErrSum += kp.errorSum;
      }
      spuriousPeople += predPeople.length - used.length;
    }

    return DetectionBenchmarkResult(
      clipName: name,
      frameCount: n,
      ball: BallDetectionMetrics(
        truePositives: ballTp,
        falsePositives: ballFp,
        falseNegatives: ballFn,
        iouSum: ballIoUSum,
        centerErrorSum: ballCenterErrSum,
      ),
      pose: PoseDetectionMetrics(
        matchedPeople: matchedPeople,
        missedPeople: missedPeople,
        spuriousPeople: spuriousPeople,
        boxIoUSum: boxIoUSum,
        correctKeypoints: correctKp,
        evaluatedKeypoints: evaluatedKp,
        keypointErrorSum: kpErrSum,
      ),
    );
  }

  /// Score only the ground-truth-visible keypoints (gt confidence > 0), the
  /// standard PCK convention.
  ({int correct, int evaluated, double errorSum}) _scoreKeypoints(
    PersonPose gt,
    PersonPose pred,
  ) {
    var correct = 0;
    var evaluated = 0;
    var errorSum = 0.0;
    final count = math.min(gt.keypoints.length, pred.keypoints.length);
    for (var k = 0; k < count; k++) {
      final gk = gt.keypoints[k];
      if (gk.confidence <= 0) continue; // not labeled / not visible
      final pk = pred.keypoints[k];
      final dx = gk.x - pk.x;
      final dy = gk.y - pk.y;
      final dist = math.sqrt(dx * dx + dy * dy);
      evaluated++;
      errorSum += dist;
      if (dist <= config.keypointThreshold) correct++;
    }
    return (correct: correct, evaluated: evaluated, errorSum: errorSum);
  }
}
