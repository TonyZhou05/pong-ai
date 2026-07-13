import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';
import 'package:pong_ai/core/vision/yolo_vision_service.dart';
import 'package:pong_ai/features/match/camera_match_screen.dart';

void main() {
  testWidgets(
    'live scoreboard renders over a headless camera preview and scores '
    'from YoloVisionService frames',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraMatchScreen(
            visionService: vision,
            // Headless stand-in for the real YOLOView platform view.
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
            // No calibrator so the demo frames score deterministically.
            matchControllerBuilder: MatchController.new,
          ),
        ),
      );
      await tester.pump(); // let load()/start()/listen wiring settle

      expect(find.text('Player A'), findsOneWidget);
      expect(find.text('Player B'), findsOneWidget);
      expect(find.text('Waiting for the first rally…'), findsOneWidget);

      // Feed the first scripted rally through the camera service's frame path.
      // The monotonic clock + adapter seam are exercised end to end.
      for (final frame in demoMatchFrames().take(13)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      expect(find.textContaining('Player A — not returned'), findsOneWidget);

      // Tear down so the broadcast stream controller closes cleanly.
      await tester.pumpWidget(const SizedBox());
    },
  );
}
