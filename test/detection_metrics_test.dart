import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/benchmark/clip_fixture.dart';
import 'package:pong_ai/core/benchmark/detection_metrics.dart';
import 'package:pong_ai/core/vision/detection.dart';

FrameResult _ballFrame(int t, BBox? box) => FrameResult(
      timestampMs: t,
      ball: box == null
          ? null
          : Detection(label: 'ball', confidence: 0.9, box: box),
    );

PersonPose _person(BBox box, List<Keypoint> kps) =>
    PersonPose(box: box, keypoints: kps);

void main() {
  group('boxIoU', () {
    test('identical boxes have IoU 1', () {
      final iou = boxIoU(
        const BBox(0.1, 0.1, 0.2, 0.2),
        const BBox(0.1, 0.1, 0.2, 0.2),
      );
      expect(iou, closeTo(1.0, 1e-9));
    });

    test('disjoint boxes have IoU 0', () {
      final iou = boxIoU(
        const BBox(0, 0, 0.1, 0.1),
        const BBox(0.5, 0.5, 0.1, 0.1),
      );
      expect(iou, 0);
    });

    test('half-shifted equal boxes have IoU 1/3', () {
      final iou = boxIoU(
        const BBox(0, 0, 0.2, 0.2),
        const BBox(0.1, 0, 0.2, 0.2),
      );
      expect(iou, closeTo(1 / 3, 1e-9));
    });
  });

  group('ball detection metrics', () {
    test('a perfectly tracked ball is a true positive with IoU 1', () {
      const box = BBox(0.5, 0.5, 0.02, 0.02);
      final result = const DetectionBenchmark().evaluate(
        name: 'perfect',
        predicted: [_ballFrame(0, box), _ballFrame(33, box)],
        groundTruth: [_ballFrame(0, box), _ballFrame(33, box)],
      );

      expect(result.ball.truePositives, 2);
      expect(result.ball.falsePositives, 0);
      expect(result.ball.falseNegatives, 0);
      expect(result.ball.precision, 1.0);
      expect(result.ball.recall, 1.0);
      expect(result.ball.f1, 1.0);
      expect(result.ball.meanIoU, closeTo(1.0, 1e-9));
      expect(result.ball.meanCenterError, closeTo(0.0, 1e-9));
    });

    test('a missed ground-truth ball is a false negative', () {
      const box = BBox(0.5, 0.5, 0.02, 0.02);
      final result = const DetectionBenchmark().evaluate(
        name: 'missed',
        predicted: [_ballFrame(0, null)],
        groundTruth: [_ballFrame(0, box)],
      );

      expect(result.ball.truePositives, 0);
      expect(result.ball.falseNegatives, 1);
      expect(result.ball.falsePositives, 0);
      expect(result.ball.recall, 0.0);
      expect(result.ball.precision, 1.0); // nothing predicted → no FP
    });

    test('a ghost detection with no true ball is a false positive', () {
      const box = BBox(0.5, 0.5, 0.02, 0.02);
      final result = const DetectionBenchmark().evaluate(
        name: 'ghost',
        predicted: [_ballFrame(0, box)],
        groundTruth: [_ballFrame(0, null)],
      );

      expect(result.ball.falsePositives, 1);
      expect(result.ball.truePositives, 0);
      expect(result.ball.precision, 0.0);
    });

    test('a detection too far from the true ball counts as FP and FN', () {
      final result = const DetectionBenchmark().evaluate(
        name: 'nearmiss',
        predicted: [_ballFrame(0, const BBox(0.1, 0.1, 0.02, 0.02))],
        groundTruth: [_ballFrame(0, const BBox(0.8, 0.8, 0.02, 0.02))],
      );

      expect(result.ball.truePositives, 0);
      expect(result.ball.falsePositives, 1);
      expect(result.ball.falseNegatives, 1);
    });

    test('loose IoU threshold accepts a partially overlapping ball', () {
      // IoU 1/3 ≥ default 0.3 threshold → still a true positive.
      final result = const DetectionBenchmark().evaluate(
        name: 'loose',
        predicted: [_ballFrame(0, const BBox(0.1, 0, 0.2, 0.2))],
        groundTruth: [_ballFrame(0, const BBox(0, 0, 0.2, 0.2))],
      );

      expect(result.ball.truePositives, 1);
      expect(result.ball.meanIoU, closeTo(1 / 3, 1e-9));
    });
  });

  group('pose metrics', () {
    final gtPerson = _person(
      const BBox(0.0, 0.1, 0.2, 0.6),
      const [
        Keypoint(0.1, 0.1, 1.0), // visible
        Keypoint(0.2, 0.2, 1.0), // visible
        Keypoint(0.3, 0.3, 0.0), // not labeled → skipped
      ],
    );

    test('matched person scores PCK over visible keypoints only', () {
      final predPerson = _person(
        const BBox(0.0, 0.1, 0.2, 0.6),
        const [
          Keypoint(0.11, 0.11, 0.9), // ~0.014 from gt → correct
          Keypoint(0.35, 0.35, 0.9), // far → incorrect
          Keypoint(0.9, 0.9, 0.9), // gt not labeled → not counted
        ],
      );
      final result = const DetectionBenchmark().evaluate(
        name: 'pose',
        predicted: [
          FrameResult(timestampMs: 0, people: [predPerson]),
        ],
        groundTruth: [
          FrameResult(timestampMs: 0, people: [gtPerson]),
        ],
      );

      expect(result.pose.matchedPeople, 1);
      expect(result.pose.missedPeople, 0);
      expect(result.pose.spuriousPeople, 0);
      expect(result.pose.evaluatedKeypoints, 2);
      expect(result.pose.correctKeypoints, 1);
      expect(result.pose.pck, closeTo(0.5, 1e-9));
      expect(result.pose.detectionRate, 1.0);
      expect(result.pose.meanBoxIoU, closeTo(1.0, 1e-9));
    });

    test('a ground-truth person with no prediction is missed', () {
      final result = const DetectionBenchmark().evaluate(
        name: 'missed_person',
        predicted: [const FrameResult(timestampMs: 0)],
        groundTruth: [
          FrameResult(timestampMs: 0, people: [gtPerson]),
        ],
      );

      expect(result.pose.matchedPeople, 0);
      expect(result.pose.missedPeople, 1);
      expect(result.pose.detectionRate, 0.0);
    });

    test('a predicted person matching no ground truth is spurious', () {
      final result = const DetectionBenchmark().evaluate(
        name: 'spurious_person',
        predicted: [
          FrameResult(timestampMs: 0, people: [gtPerson]),
        ],
        groundTruth: [const FrameResult(timestampMs: 0)],
      );

      expect(result.pose.matchedPeople, 0);
      expect(result.pose.spuriousPeople, 1);
    });

    test('a prediction whose box IoU is below threshold does not match', () {
      final farPerson = _person(
        const BBox(0.7, 0.1, 0.2, 0.6),
        const [Keypoint(0.75, 0.15, 0.9)],
      );
      final result = const DetectionBenchmark().evaluate(
        name: 'lowiou_person',
        predicted: [
          FrameResult(timestampMs: 0, people: [farPerson]),
        ],
        groundTruth: [
          FrameResult(timestampMs: 0, people: [gtPerson]),
        ],
      );

      expect(result.pose.matchedPeople, 0);
      expect(result.pose.missedPeople, 1);
      expect(result.pose.spuriousPeople, 1);
    });
  });

  group('DetectionBenchmark clip + edges', () {
    test('evaluateClip returns null without per-frame ground truth', () {
      const clip = ClipFixture(
        name: 'no_gt_frames',
        frames: [],
        groundTruth: ClipGroundTruth(pointsA: 0, pointsB: 0),
      );
      expect(const DetectionBenchmark().evaluateClip(clip), isNull);
    });

    test('evaluateClip scores frames against groundTruthFrames', () {
      const box = BBox(0.5, 0.5, 0.02, 0.02);
      final clip = ClipFixture(
        name: 'gt_frames',
        frames: [_ballFrame(0, box)],
        groundTruthFrames: [_ballFrame(0, box)],
        groundTruth: const ClipGroundTruth(pointsA: 0, pointsB: 0),
      );
      final result = const DetectionBenchmark().evaluateClip(clip)!;

      expect(result.frameCount, 1);
      expect(result.ball.truePositives, 1);
      expect(result.report(), contains('Perception: gt_frames'));
    });

    test('unequal-length streams are scored over the longer stream', () {
      const box = BBox(0.5, 0.5, 0.02, 0.02);
      final result = const DetectionBenchmark().evaluate(
        name: 'ragged',
        predicted: [_ballFrame(0, box)],
        groundTruth: [_ballFrame(0, box), _ballFrame(33, box)],
      );

      expect(result.frameCount, 2);
      expect(result.ball.truePositives, 1);
      expect(result.ball.falseNegatives, 1); // trailing gt frame unmatched
    });
  });

  group('groundTruthFrames JSON', () {
    test('round-trips per-frame ground truth', () {
      final clip = ClipFixture(
        name: 'gt_json',
        frames: [_ballFrame(0, const BBox(0.5, 0.5, 0.02, 0.02))],
        groundTruthFrames: [
          _ballFrame(0, const BBox(0.51, 0.5, 0.02, 0.02)),
        ],
        groundTruth: const ClipGroundTruth(pointsA: 0, pointsB: 0),
      );
      final decoded = ClipFixture.fromJson(
        jsonDecode(jsonEncode(clip.toJson())) as Map<String, dynamic>,
      );

      expect(decoded.groundTruthFrames, hasLength(1));
      expect(
        decoded.groundTruthFrames!.single.ball!.box.left,
        closeTo(0.51, 1e-9),
      );
      // And a fixture with no gt frames stays null through JSON.
      const bare = ClipFixture(
        name: 'bare',
        frames: [],
        groundTruth: ClipGroundTruth(pointsA: 0, pointsB: 0),
      );
      final bareDecoded = ClipFixture.fromJson(
        jsonDecode(jsonEncode(bare.toJson())) as Map<String, dynamic>,
      );
      expect(bareDecoded.groundTruthFrames, isNull);
    });
  });
}
