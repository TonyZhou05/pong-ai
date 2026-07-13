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

    test('caps players at maxPeople, keeping the largest boxes', () {
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

    test('propagates fps', () {
      final frame = adapter.fromResults([], timestampMs: 0, fps: 28.5);
      expect(frame.fps, closeTo(28.5, 1e-9));
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
