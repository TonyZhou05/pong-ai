import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/history/session_history_store.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';
import 'package:pong_ai/core/vision/yolo_vision_service.dart';
import 'package:pong_ai/features/training/camera_training_screen.dart';

/// In-memory [SessionHistoryStore] so the widget test can pump/settle without
/// real `dart:io` (which a `testWidgets` body defers). Mirrors the fake used in
/// session_history_screen_test.dart; the on-disk store has its own unit tests.
class FakeHistoryStore extends SessionHistoryStore {
  FakeHistoryStore() : super(Directory.systemTemp);

  final List<StoredSession> saved = [];

  @override
  Future<StoredSession> save({
    required SessionKind kind,
    required Map<String, dynamic> report,
    DateTime? at,
  }) async {
    final session = StoredSession(
      id: '${kind.key}-${saved.length}',
      kind: kind,
      savedAt: at ?? DateTime.now(),
      report: report,
    );
    saved.add(session);
    return session;
  }
}

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

  testWidgets(
    'live training Save to history persists the graded session',
    (tester) async {
      final vision = YoloVisionService();
      final store = FakeHistoryStore();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            historyStoreLoader: () async => store,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();

      // Drive one graded stroke so the saved report carries real shot data.
      for (final frame in trainingSessionFrames().take(6)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      await tester.tap(find.byTooltip('Finish session'));
      await tester.pump();

      await tester.ensureVisible(find.text('Save to history'));
      await tester.tap(find.text('Save to history'));
      await tester.pump(); // resolve the save future + snackbar

      expect(store.saved, hasLength(1));
      expect(store.saved.single.kind, SessionKind.training);
      final session = store.saved.single.report['session'] as Map;
      expect(session['shotCount'], 1);
      expect(find.text('Saved to history'), findsOneWidget);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'live training overlay draws the player box, ball, then a ghost ball',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();

      final overlayFinder = find.byWidgetPredicate(
        (w) => w.runtimeType.toString() == '_TargetOverlay',
      );
      expect(overlayFinder, findsOneWidget);

      // Feed two ball frames (with a player) to establish a trajectory the
      // Kalman filter can extrapolate from.
      for (var i = 1; i <= 2; i++) {
        vision.onFrame(
          FrameResult(
            timestampMs: i * 33,
            ball: Detection(
              label: 'ball',
              confidence: 0.9,
              box: BBox(0.40 + i * 0.05, 0.50, 0.02, 0.02),
            ),
            people: const [
              PersonPose(box: BBox(0.10, 0.30, 0.12, 0.50), keypoints: []),
            ],
          ),
        );
        await tester.pump();
        await tester.pump();
      }

      // Target band + player box + real ball marker are DecoratedBoxes (the net
      // line is a plain ColoredBox), so three inside the overlay.
      final decorated = find.descendant(
        of: overlayFinder,
        matching: find.byType(DecoratedBox),
      );
      expect(decorated, findsNWidgets(3));

      // A frame with no ball detection should keep drawing a ghost ball
      // extrapolated from the trajectory (still three DecoratedBoxes).
      vision.onFrame(
        const FrameResult(
          timestampMs: 99,
          people: [PersonPose(box: BBox(0.10, 0.30, 0.12, 0.50), keypoints: [])],
        ),
      );
      await tester.pump();
      await tester.pump();
      expect(decorated, findsNWidgets(3));

      await tester.pumpWidget(const SizedBox());
    },
  );
}
