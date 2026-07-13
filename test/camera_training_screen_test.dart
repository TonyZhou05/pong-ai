import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';
import 'package:pong_ai/core/vision/yolo_vision_service.dart';
import 'package:pong_ai/features/training/camera_training_screen.dart';

void main() {
  testWidgets(
    'live training grades a stroke from YoloVisionService camera frames',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            // Headless stand-in for the real YOLOView platform view.
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump(); // let load()/start()/listen wiring settle

      expect(find.text('Grade'), findsOneWidget);
      expect(find.text('Waiting for the first shot…'), findsOneWidget);

      // Feed the first scripted stroke through the camera service's frame path.
      for (final frame in trainingSessionFrames().take(6)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      // The first stroke (depth 0.75) grades excellent and appears in the feed.
      expect(find.text('1 shots'), findsOneWidget);
      expect(find.textContaining('excellent'), findsOneWidget);

      // Finish freezes the session and shows the report.
      await tester.tap(find.byTooltip('Finish session'));
      await tester.pump();
      expect(find.text('Session complete'), findsOneWidget);

      // Tear down so the broadcast stream controller closes cleanly.
      await tester.pumpWidget(const SizedBox());
    },
  );
}
