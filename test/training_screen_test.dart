import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/app.dart';
import 'package:pong_ai/core/training/shot_analyzer.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';
import 'package:pong_ai/core/vision/vision_service.dart';
import 'package:pong_ai/features/training/training_screen.dart';

/// A camera-free [VisionService] whose frames are pushed manually by the test,
/// so no periodic timers leak into the widget-test environment.
class FakeVisionService implements VisionService {
  final StreamController<FrameResult> _controller =
      StreamController<FrameResult>.broadcast();

  @override
  Stream<FrameResult> get frames => _controller.stream;

  @override
  Future<void> load() async {}

  @override
  Future<void> start() async {}

  @override
  Future<void> stop() async {}

  @override
  Future<void> dispose() async {
    if (!_controller.isClosed) await _controller.close();
  }

  void emit(FrameResult frame) => _controller.add(frame);
}

void main() {
  test('scripted training frames grade to the expected shot mix', () {
    final analyzer = ShotAnalyzer();
    for (final frame in trainingSessionFrames()) {
      analyzer.onFrame(frame);
    }

    final shots = analyzer.shots;
    expect(shots, hasLength(6));
    expect(
      shots.map((s) => s.grade).toList(),
      const [
        ShotGrade.excellent,
        ShotGrade.excellent,
        ShotGrade.good,
        ShotGrade.good,
        ShotGrade.fair,
        ShotGrade.excellent,
      ],
    );
    // Every stroke lands on the far (right) half, so depth is positive.
    expect(shots.every((s) => s.depth > 0), isTrue);
  });

  testWidgets('training screen renders and updates as shots complete',
      (tester) async {
    final fake = FakeVisionService();
    await tester.pumpWidget(
      MaterialApp(home: TrainingScreen(visionServiceBuilder: () => fake)),
    );
    await tester.pump(); // let start()/listen wiring settle

    expect(find.text('Recent shots'), findsOneWidget);
    expect(find.text('Waiting for the first shot…'), findsOneWidget);
    expect(find.text('0 shots'), findsOneWidget);

    // Feed the first scripted stroke (14 frames): one graded shot lands.
    for (final frame in trainingSessionFrames().take(14)) {
      fake.emit(frame);
      await tester.pump();
    }

    expect(find.text('1 shots'), findsOneWidget);
    expect(find.textContaining('excellent'), findsOneWidget);
  });

  testWidgets('Training card on the home screen opens the training screen',
      (tester) async {
    await tester.pumpWidget(const PongAiApp());

    await tester.tap(find.text('Training'));
    await tester.pump(); // start the push transition
    await tester.pump(const Duration(milliseconds: 400)); // settle it

    expect(find.text('Recent shots'), findsOneWidget);

    // Dispose the pushed screen so its replay timer is cancelled.
    await tester.pumpWidget(const SizedBox());
  });
}
