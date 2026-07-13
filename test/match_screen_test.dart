import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/app.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';
import 'package:pong_ai/core/vision/vision_service.dart';
import 'package:pong_ai/features/match/match_screen.dart';

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
  testWidgets('scoreboard renders and updates as the referee awards points',
      (tester) async {
    final fake = FakeVisionService();
    await tester.pumpWidget(
      MaterialApp(home: MatchScreen(visionServiceBuilder: () => fake)),
    );
    await tester.pump(); // let start()/listen wiring settle

    expect(find.text('Player A'), findsOneWidget);
    expect(find.text('Player B'), findsOneWidget);
    expect(find.text('Waiting for the first rally…'), findsOneWidget);

    // Feed the first scripted rally (13 frames): a right-side bounce then a
    // lost ball, which the referee scores as a point for Player A.
    for (final frame in demoMatchFrames().take(13)) {
      fake.emit(frame);
      await tester.pump();
    }

    expect(find.textContaining('Player A — not returned'), findsOneWidget);
  });

  testWidgets('Match card on the home screen opens the match screen',
      (tester) async {
    await tester.pumpWidget(const PongAiApp());

    await tester.tap(find.text('Match'));
    await tester.pump(); // start the push transition
    await tester.pump(const Duration(milliseconds: 400)); // settle it

    expect(find.text('Player A'), findsOneWidget);

    // Dispose the pushed screen so its replay timer is cancelled.
    await tester.pumpWidget(const SizedBox());
  });
}
