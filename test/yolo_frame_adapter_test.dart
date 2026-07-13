import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/vision/yolo_frame_adapter.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Build a plugin `YOLOResult` for a normalized box, optionally with
/// pixel-space pose keypoints.
YOLOResult _result(
  String className,
  double confidence,
  Rect normalizedBox, {
  List<Keypoint>? keypoints,
  List<double>? keypointConfidences,
}) {
  return YOLOResult(
    classIndex: 0,
    className: className,
    confidence: confidence,
    boundingBox: Rect.fromLTWH(
      normalizedBox.left * 1000,
      normalizedBox.top * 1000,
      normalizedBox.width * 1000,
      normalizedBox.height * 1000,
    ),
    normalizedBox: normalizedBox,
    keypoints: keypoints,
    keypointConfidences: keypointConfidences,
  );
}

void main() {
  const adapter = YoloFrameAdapter();

  group('YoloFrameAdapter.fromResults', () {
    test('maps a sports ball detection to the ball', () {
      final frame = adapter.fromResults(
        [_result('sports ball', 0.8, const Rect.fromLTWH(0.4, 0.5, 0.02, 0.02))],
        timestampMs: 100,
      );

      expect(frame.timestampMs, 100);
      expect(frame.ball, isNotNull);
      expect(frame.ball!.label, 'ball');
      expect(frame.ball!.confidence, closeTo(0.8, 1e-9));
      expect(frame.ball!.box.centerX, closeTo(0.41, 1e-9));
      expect(frame.people, isEmpty);
    });

    test('keeps only the highest-confidence ball', () {
      final frame = adapter.fromResults(
        [
          _result('sports ball', 0.4, const Rect.fromLTWH(0.1, 0.1, 0.02, 0.02)),
          _result('sports ball', 0.9, const Rect.fromLTWH(0.7, 0.7, 0.02, 0.02)),
        ],
        timestampMs: 0,
      );

      expect(frame.ball!.confidence, closeTo(0.9, 1e-9));
      expect(frame.ball!.box.left, closeTo(0.7, 1e-9));
    });

    test('exposes the other ball candidates best-first beyond the primary', () {
      final frame = adapter.fromResults(
        [
          _result('sports ball', 0.4, const Rect.fromLTWH(0.1, 0.1, 0.02, 0.02)),
          _result('sports ball', 0.9, const Rect.fromLTWH(0.7, 0.7, 0.02, 0.02)),
          _result('sports ball', 0.6, const Rect.fromLTWH(0.5, 0.5, 0.02, 0.02)),
        ],
        timestampMs: 0,
      );

      // Primary is the top confidence; the rest ride along, next-best first.
      expect(frame.ball!.confidence, closeTo(0.9, 1e-9));
      expect(
        frame.ballCandidates.map((c) => c.confidence),
        [closeTo(0.6, 1e-9), closeTo(0.4, 1e-9)],
      );
    });

    test('a single ball leaves the candidate list empty', () {
      final frame = adapter.fromResults(
        [_result('sports ball', 0.8, const Rect.fromLTWH(0.4, 0.5, 0.02, 0.02))],
        timestampMs: 0,
      );
      expect(frame.ball, isNotNull);
      expect(frame.ballCandidates, isEmpty);
    });

    test('drops sub-threshold ball detections', () {
      final frame = adapter.fromResults(
        [_result('sports ball', 0.05, const Rect.fromLTWH(0.4, 0.5, 0.02, 0.02))],
        timestampMs: 0,
      );
      expect(frame.ball, isNull);
    });

    test('maps person detections to poses with normalized keypoints', () {
      final frame = adapter.fromResults(
        [
          _result(
            'person',
            0.95,
            const Rect.fromLTWH(0.1, 0.2, 0.2, 0.6),
            // Pixel-space keypoints in a 640x480 frame.
            keypoints: [Keypoint(320, 240), Keypoint(64, 96)],
            keypointConfidences: [0.9, 0.8],
          ),
        ],
        timestampMs: 33,
        imageWidth: 640,
        imageHeight: 480,
      );

      expect(frame.people, hasLength(1));
      final pose = frame.people.single;
      expect(pose.box.left, closeTo(0.1, 1e-9));
      expect(pose.keypoints, hasLength(2));
      expect(pose.keypoints[0].x, closeTo(0.5, 1e-9));
      expect(pose.keypoints[0].y, closeTo(0.5, 1e-9));
      expect(pose.keypoints[0].confidence, closeTo(0.9, 1e-9));
      expect(pose.keypoints[1].x, closeTo(0.1, 1e-9));
      expect(pose.keypoints[1].y, closeTo(0.2, 1e-9));
    });

    test('drops keypoints when image dimensions are unknown', () {
      final frame = adapter.fromResults(
        [
          _result(
            'person',
            0.95,
            const Rect.fromLTWH(0.1, 0.2, 0.2, 0.6),
            keypoints: [Keypoint(320, 240)],
            keypointConfidences: [0.9],
          ),
        ],
        timestampMs: 0,
      );

      expect(frame.people, hasLength(1));
      expect(frame.people.single.keypoints, isEmpty);
      // Box is still preserved for tracking.
      expect(frame.people.single.box.width, closeTo(0.2, 1e-9));
    });

    test('ignores non-ball, non-person classes', () {
      final frame = adapter.fromResults(
        [
          _result('chair', 0.99, const Rect.fromLTWH(0.1, 0.1, 0.2, 0.2)),
          _result('bench', 0.99, const Rect.fromLTWH(0.3, 0.3, 0.2, 0.2)),
        ],
        timestampMs: 0,
      );
      expect(frame.ball, isNull);
      expect(frame.people, isEmpty);
    });

    test('caps players at maxPeople, keeping the largest boxes by area', () {
      final frame = adapter.fromResults(
        [
          _result('person', 0.9, const Rect.fromLTWH(0.0, 0.0, 0.10, 0.5)),
          _result('person', 0.9, const Rect.fromLTWH(0.2, 0.0, 0.30, 0.5)),
          _result('person', 0.9, const Rect.fromLTWH(0.6, 0.0, 0.05, 0.5)),
        ],
        timestampMs: 0,
      );
      expect(frame.people, hasLength(2));
      final widths = frame.people.map((p) => p.box.width).toList()..sort();
      expect(widths, [closeTo(0.10, 1e-9), closeTo(0.30, 1e-9)]);
    });

    test('keeps side-on players (tall, narrow) over a wide, short spectator', () {
      // Phone at the side of the table: the two players are seen side-on, so
      // their boxes are narrow but tall. A spectator facing the camera is wide
      // but short — a width-only cap would wrongly keep the spectator.
      final frame = adapter.fromResults(
        [
          // Player 1: narrow + tall -> area 0.15 * 0.6 = 0.090.
          _result('person', 0.9, const Rect.fromLTWH(0.10, 0.2, 0.15, 0.6)),
          // Player 2: narrow + tall -> area 0.13 * 0.6 = 0.078.
          _result('person', 0.9, const Rect.fromLTWH(0.60, 0.2, 0.13, 0.6)),
          // Spectator: wide + short -> area 0.30 * 0.20 = 0.060 (widest box).
          _result('person', 0.9, const Rect.fromLTWH(0.35, 0.7, 0.30, 0.20)),
        ],
        timestampMs: 0,
      );
      expect(frame.people, hasLength(2));
      final lefts = frame.people.map((p) => p.box.left).toList()..sort();
      // Both players kept; the wide spectator (left 0.35) dropped.
      expect(lefts, [closeTo(0.10, 1e-9), closeTo(0.60, 1e-9)]);
    });

    test('propagates fps', () {
      final frame = adapter.fromResults([], timestampMs: 0, fps: 28.5);
      expect(frame.fps, closeTo(28.5, 1e-9));
    });

    test('size gate rejects an implausibly large ball, keeping a real one', () {
      const gated = YoloFrameAdapter(
        config: YoloFrameConfig(maxBallRelativeSize: 0.25),
      );
      final frame = gated.fromResults(
        [
          // A high-confidence but huge "sports ball" (a mislabeled head/logo):
          // large in both axes -> rejected.
          _result('sports ball', 0.95, const Rect.fromLTWH(0.3, 0.3, 0.4, 0.4)),
          // A genuine tiny ping-pong ball at lower confidence -> kept.
          _result('sports ball', 0.6, const Rect.fromLTWH(0.5, 0.5, 0.02, 0.02)),
        ],
        timestampMs: 0,
      );
      expect(frame.ball, isNotNull);
      expect(frame.ball!.confidence, closeTo(0.6, 1e-9));
      expect(frame.ball!.box.width, closeTo(0.02, 1e-9));
    });

    test('size gate keeps a motion-blurred ball elongated along one axis', () {
      const gated = YoloFrameAdapter(
        config: YoloFrameConfig(maxBallRelativeSize: 0.25),
      );
      // A fast ball smears wide (0.4) but stays thin (0.03) — the smaller
      // dimension is under the cap, so it survives.
      final frame = gated.fromResults(
        [_result('sports ball', 0.8, const Rect.fromLTWH(0.3, 0.5, 0.4, 0.03))],
        timestampMs: 0,
      );
      expect(frame.ball, isNotNull);
      expect(frame.ball!.box.width, closeTo(0.4, 1e-9));
    });

    test('size gate is off by default (large ball still accepted)', () {
      final frame = adapter.fromResults(
        [_result('sports ball', 0.9, const Rect.fromLTWH(0.3, 0.3, 0.4, 0.4))],
        timestampMs: 0,
      );
      expect(frame.ball, isNotNull);
      expect(frame.ball!.box.width, closeTo(0.4, 1e-9));
    });
  });

  group('YoloFrameAdapter.fromStreamingData', () {
    test('parses a raw streaming payload end to end', () {
      final data = <String, dynamic>{
        'timestamp': 1234.0,
        'fps': 30.0,
        'imageWidth': 640,
        'imageHeight': 480,
        'detections': [
          _result('sports ball', 0.7, const Rect.fromLTWH(0.5, 0.5, 0.02, 0.02))
              .toMap(),
          _result(
            'person',
            0.9,
            const Rect.fromLTWH(0.1, 0.1, 0.2, 0.6),
            keypoints: [Keypoint(320, 240)],
            keypointConfidences: [0.88],
          ).toMap(),
        ],
      };

      final frame = adapter.fromStreamingData(data);

      expect(frame.timestampMs, 1234);
      expect(frame.fps, closeTo(30.0, 1e-9));
      expect(frame.ball, isNotNull);
      expect(frame.people, hasLength(1));
      expect(frame.people.single.keypoints.single.x, closeTo(0.5, 1e-9));
    });

    test('uses the fallback timestamp when none is provided', () {
      final frame = adapter.fromStreamingData(
        <String, dynamic>{'detections': const []},
        fallbackTimestampMs: 99,
      );
      expect(frame.timestampMs, 99);
      expect(frame.ball, isNull);
      expect(frame.people, isEmpty);
    });

    test('tolerates a missing detections key', () {
      final frame = adapter.fromStreamingData(<String, dynamic>{});
      expect(frame.ball, isNull);
      expect(frame.people, isEmpty);
    });
  });
}
