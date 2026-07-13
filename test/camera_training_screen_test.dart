import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/history/session_history_store.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';
import 'package:pong_ai/core/vision/yolo_vision_service.dart';
import 'package:pong_ai/features/training/camera_training_screen.dart';

/// A single stroke crossing the net right→left and bouncing at [bounceX] on the
/// left half — the mirror of the built-in scripted stroke — for exercising the
/// `playerSide == right` (target = left) path the picker selects.
List<FrameResult> _leftHalfStroke({double bounceX = 0.125}) {
  const xs = <double>[0.52, 0.48];
  const ys = <double>[0.45, 0.48, 0.55, 0.65, 0.55, 0.45];
  return [
    for (var i = 0; i < ys.length; i++)
      FrameResult(
        timestampMs: i * 33,
        ball: Detection(
          label: 'ball',
          confidence: 0.9,
          box: BBox(
            (i < xs.length ? xs[i] : bounceX) - 0.01,
            ys[i] - 0.01,
            0.02,
            0.02,
          ),
        ),
      ),
  ];
}

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
            // Grade the scripted stroke immediately (no calibration warm-up).
            autoCalibrate: false,
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
    'live training speaks + captions a coach call on each graded shot',
    (tester) async {
      final vision = YoloVisionService();
      final calls = <String>[];
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            autoCalibrate: false,
            onAnnounce: calls.add,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();

      // Feed the first scripted stroke (depth 0.75 → excellent).
      for (final frame in trainingSessionFrames().take(6)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      // The grade call was spoken through the injected sink and captioned.
      expect(calls, ['Excellent shot!']);
      expect(find.text('Excellent shot!'), findsOneWidget);

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
            // Grade the scripted stroke immediately (no calibration warm-up).
            autoCalibrate: false,
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
    'player-side picker is shown pre-session and hides after the first shot',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            // Grade the scripted stroke immediately (no calibration warm-up).
            autoCalibrate: false,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();

      // The picker is offered before any shot is graded.
      expect(find.text('I hit from:'), findsOneWidget);
      expect(find.widgetWithText(ChoiceChip, 'Left'), findsOneWidget);
      expect(find.widgetWithText(ChoiceChip, 'Right'), findsOneWidget);

      // Grade the first default (right-target) stroke.
      for (final frame in trainingSessionFrames().take(6)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      // Once a shot lands the pre-session picker is locked away.
      expect(find.text('1 shots'), findsOneWidget);
      expect(find.text('I hit from:'), findsNothing);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'selecting Right grades a stroke that lands on the left half',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            // Grade the scripted stroke immediately (no calibration warm-up).
            autoCalibrate: false,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();

      // Player is on the right, so the target half is the left.
      await tester.tap(find.widgetWithText(ChoiceChip, 'Right'));
      await tester.pump();

      for (final frame in _leftHalfStroke()) {
        vision.onFrame(frame);
        await tester.pump();
      }
      // One more empty frame so the apex bounce (reported one frame late) grades.
      vision.onFrame(const FrameResult(timestampMs: 300));
      await tester.pump();

      // The left-landing stroke is graded because the target side flipped.
      expect(find.text('1 shots'), findsOneWidget);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'pausing drops strokes; resuming grades again',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            // Grade the scripted stroke immediately (no calibration warm-up).
            autoCalibrate: false,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();

      // Pause the drill for a break in play.
      await tester.tap(find.byTooltip('Pause drill'));
      await tester.pump();
      expect(find.text('PAUSED'), findsOneWidget);

      // A full stroke fed during the break must NOT be graded.
      for (final frame in trainingSessionFrames().take(6)) {
        vision.onFrame(frame);
        await tester.pump();
      }
      expect(find.text('Waiting for the first shot…'), findsOneWidget);

      // Resume clears the banner and re-enables grading.
      await tester.tap(find.byTooltip('Resume drill'));
      await tester.pump();
      expect(find.text('PAUSED'), findsNothing);

      // The same stroke now grades — the pre-break tracker state was cleared, so
      // it's segmented as a fresh, complete stroke.
      for (final frame in trainingSessionFrames().take(6)) {
        vision.onFrame(frame);
        await tester.pump();
      }
      expect(find.text('1 shots'), findsOneWidget);

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
            // Grade the scripted stroke immediately (no calibration warm-up).
            autoCalibrate: false,
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

  testWidgets(
    'auto-calibration shows a hint and defers grading until the table is learned',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          // Default autoCalibrate: true — warm up on the live ball path first.
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

      // Before enough ball samples, the calibration hint is shown and nothing
      // is graded even when a full scripted stroke is fed.
      expect(find.textContaining('Calibrating…'), findsOneWidget);
      for (final frame in trainingSessionFrames().take(6)) {
        vision.onFrame(frame);
        await tester.pump();
      }
      expect(find.text('0 shots'), findsOneWidget);

      // A full warm-up drill (spanning the table with varied ball height) lets
      // the calibrator infer the geometry; the hint then clears.
      var t = 100000;
      for (var i = 0; i < 24; i++) {
        final x = 0.10 + (i % 6) * 0.16; // sweep 0.10 .. 0.90 across the table
        final y = 0.35 + (i % 4) * 0.06; // vary height so the band isn't a line
        vision.onFrame(
          FrameResult(
            timestampMs: t += 33,
            ball: Detection(
              label: 'ball',
              confidence: 0.9,
              box: BBox(x - 0.01, y - 0.01, 0.02, 0.02),
            ),
          ),
        );
        await tester.pump();
      }
      expect(find.textContaining('Calibrating…'), findsNothing);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'auto-calibration prompts a reposition once warm-up stalls',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            // Default autoCalibrate: true, with a tiny stall budget.
            calibrationStallFrames: 3,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();
      expect(find.textContaining('Calibrating…'), findsOneWidget);
      expect(find.textContaining('reposition'), findsNothing);

      // Ball-less frames never let the calibrator reach its threshold, so after
      // the budget the banner switches to the actionable reposition prompt.
      for (var i = 0; i < 3; i++) {
        vision.onFrame(FrameResult(timestampMs: (i + 1) * 33));
        await tester.pump();
      }
      expect(find.textContaining('Calibrating…'), findsNothing);
      expect(find.textContaining('reposition'), findsOneWidget);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'live training overlay flashes a km/h readout on the moving ball',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraTrainingScreen(
            visionService: vision,
            // Grade the scripted stroke immediately (no calibration warm-up).
            autoCalibrate: false,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
          ),
        ),
      );
      await tester.pump();

      // No reading before two frames establish a ball velocity.
      vision.onFrame(
        const FrameResult(
          timestampMs: 33,
          ball: Detection(
            label: 'ball',
            confidence: 0.9,
            box: BBox(0.40, 0.50, 0.02, 0.02),
          ),
        ),
      );
      await tester.pump();
      await tester.pump();
      expect(find.textContaining('km/h'), findsNothing);

      // A second moving-ball frame yields a live radar-gun label.
      vision.onFrame(
        const FrameResult(
          timestampMs: 66,
          ball: Detection(
            label: 'ball',
            confidence: 0.9,
            box: BBox(0.55, 0.50, 0.02, 0.02),
          ),
        ),
      );
      await tester.pump();
      await tester.pump();
      expect(find.textContaining('km/h'), findsOneWidget);

      // A ball dropout hides the (now stale) reading with the ball.
      vision.onFrame(const FrameResult(timestampMs: 99));
      await tester.pump();
      await tester.pump();
      expect(find.textContaining('km/h'), findsNothing);

      await tester.pumpWidget(const SizedBox());
    },
  );
}
