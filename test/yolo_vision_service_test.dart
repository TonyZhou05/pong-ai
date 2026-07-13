import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/yolo_vision_service.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart' as yolo;

/// Build a raw `onStreamingData`-shaped payload (as the plugin serializes it)
/// with a single ball detection.
Map<String, dynamic> ballPayload({
  required double timestamp,
  double confidence = 0.9,
  double? fps,
}) {
  final ball = yolo.YOLOResult(
    classIndex: 32,
    className: 'sports ball',
    confidence: confidence,
    boundingBox: const Rect.fromLTWH(100, 100, 20, 20),
    normalizedBox: const Rect.fromLTWH(0.1, 0.1, 0.02, 0.02),
  );
  return {
    'detections': [ball.toMap()],
    'imageWidth': 1920,
    'imageHeight': 1080,
    if (fps != null) 'fps': fps,
    'timestamp': timestamp,
  };
}

void main() {
  test('forwards converted frames only while running', () async {
    final service = YoloVisionService();
    final received = <FrameResult>[];
    service.frames.listen(received.add);

    // Before start(): dropped.
    service.onStreamingData(ballPayload(timestamp: 100));
    await service.load();
    await service.start();

    service.onStreamingData(ballPayload(timestamp: 200));
    service.onStreamingData(ballPayload(timestamp: 300));
    await Future<void>.delayed(Duration.zero);

    expect(received.length, 2);
    expect(received.first.ball, isNotNull);
    expect(received.first.ball!.label, 'ball');
    expect(service.frameCount, 2);

    await service.stop();
    service.onStreamingData(ballPayload(timestamp: 400));
    await Future<void>.delayed(Duration.zero);
    expect(received.length, 2, reason: 'frames after stop() are dropped');

    await service.dispose();
  });

  test('start() auto-loads when not yet loaded', () async {
    final service = YoloVisionService();
    final received = <FrameResult>[];
    service.frames.listen(received.add);

    await service.start(); // no explicit load()
    service.onStreamingData(ballPayload(timestamp: 50));
    await Future<void>.delayed(Duration.zero);

    expect(received, hasLength(1));
    expect(service.isRunning, isTrue);
    await service.dispose();
  });

  test('rewrites duplicate/decreasing device timestamps to a rising clock',
      () async {
    final service = YoloVisionService();
    final received = <FrameResult>[];
    service.frames.listen(received.add);
    await service.start();

    // Reported timestamps: 1000, 1000 (dup), 900 (backwards), 2000 (ok).
    service.onStreamingData(ballPayload(timestamp: 1000, fps: 50));
    service.onStreamingData(ballPayload(timestamp: 1000, fps: 50));
    service.onStreamingData(ballPayload(timestamp: 900, fps: 50));
    service.onStreamingData(ballPayload(timestamp: 2000, fps: 50));
    await Future<void>.delayed(Duration.zero);

    final ts = received.map((f) => f.timestampMs).toList();
    // Strictly increasing throughout.
    for (var i = 1; i < ts.length; i++) {
      expect(ts[i] > ts[i - 1], isTrue, reason: 'ts must rise: $ts');
    }
    // 50 fps -> 20ms synthetic step for the stalled frames.
    expect(ts[0], 1000);
    expect(ts[1], 1020);
    expect(ts[2], 1040);
    expect(ts[3], 2000);

    await service.dispose();
  });

  test('synthesizes a monotonic clock when payloads carry no timestamp',
      () async {
    final service =
        YoloVisionService(defaultFrameInterval: const Duration(milliseconds: 40));
    final received = <FrameResult>[];
    service.frames.listen(received.add);
    await service.start();

    for (var i = 0; i < 3; i++) {
      final p = ballPayload(timestamp: 0)..remove('timestamp');
      service.onStreamingData(p);
    }
    await Future<void>.delayed(Duration.zero);

    final ts = received.map((f) => f.timestampMs).toList();
    expect(ts, [40, 80, 120]);
    await service.dispose();
  });

  test('onFrame path is also monotonic-clamped and lifecycle-gated', () async {
    final service = YoloVisionService();
    final received = <FrameResult>[];
    service.frames.listen(received.add);

    service.onFrame(const FrameResult(timestampMs: 500)); // before start
    await service.start();
    service.onFrame(const FrameResult(timestampMs: 500));
    service.onFrame(const FrameResult(timestampMs: 500)); // dup -> bumped
    await Future<void>.delayed(Duration.zero);

    expect(received.map((f) => f.timestampMs).toList(), [500, 533]);
    await service.dispose();
  });

  test('dropped ball below confidence still forwards an empty frame', () async {
    final service = YoloVisionService();
    final received = <FrameResult>[];
    service.frames.listen(received.add);
    await service.start();

    service.onStreamingData(ballPayload(timestamp: 10, confidence: 0.05));
    await Future<void>.delayed(Duration.zero);

    expect(received, hasLength(1));
    expect(received.first.ball, isNull);
    await service.dispose();
  });

  test('dispose closes the stream and blocks further emits', () async {
    final service = YoloVisionService();
    await service.start();
    await service.dispose();

    // Must not throw despite the closed controller.
    service.onStreamingData(ballPayload(timestamp: 1));
    expect(service.isRunning, isFalse);
  });
}
