import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/history/session_history_store.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';
import 'package:pong_ai/core/vision/yolo_vision_service.dart';
import 'package:pong_ai/features/match/camera_match_screen.dart';
import 'package:pong_ai/features/summary/momentum_chart.dart';
import 'package:pong_ai/features/summary/shot_map.dart';

/// An in-memory [SessionHistoryStore] with no real file I/O, so the widget test
/// can pump/settle normally. The on-disk store is covered by its own unit tests.
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

  testWidgets(
    'live match end shows the summary panel and Save to history persists it',
    (tester) async {
      final vision = YoloVisionService();
      final store = FakeHistoryStore();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraMatchScreen(
            visionService: vision,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
            // A short best-of-one, 3-point game so the scripted demo rallies end
            // the match (the full replay never completes one) and surface the
            // end-of-match panel. No calibrator so scoring is immediate.
            matchControllerBuilder: () =>
                MatchController(engine: ScoringEngine(pointsPerGame: 3, bestOf: 1)),
            historyStoreLoader: () async => store,
          ),
        ),
      );
      await tester.pump();

      // The demo's first four rallies award A, B, A, A → Player A wins 3–1.
      for (final frame in demoMatchFrames()) {
        vision.onFrame(frame);
        await tester.pump();
      }

      expect(find.textContaining('wins the match'), findsOneWidget);
      // The live match-over panel now surfaces the headline visual analytics
      // (momentum timeline + shot map), matching the demo MatchScreen panel.
      expect(find.byType(MomentumChartView), findsOneWidget);
      expect(find.byType(ShotMapView), findsOneWidget);
      await tester.ensureVisible(find.text('Save to history'));
      await tester.tap(find.text('Save to history'));
      await tester.pumpAndSettle();

      expect(find.text('Saved to history'), findsOneWidget);
      expect(store.saved, hasLength(1));
      expect(store.saved.single.kind, SessionKind.match);
      expect(store.saved.single.report['score'], isA<Map>());

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'live tracking overlay draws player boxes and the ball over the preview',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraMatchScreen(
            visionService: vision,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
            matchControllerBuilder: MatchController.new,
          ),
        ),
      );
      await tester.pump();

      // Before any frame arrives the overlay draws nothing but the net line.
      final overlayFinder = find.byWidgetPredicate(
        (w) => w.runtimeType.toString() == '_LiveTrackingOverlay',
      );
      expect(overlayFinder, findsOneWidget);

      // A frame with two players and a ball should render two player boxes plus
      // the ball marker inside the overlay.
      vision.onFrame(
        const FrameResult(
          timestampMs: 33,
          ball: Detection(
            label: 'ball',
            confidence: 0.9,
            box: BBox(0.49, 0.49, 0.02, 0.02),
          ),
          people: [
            PersonPose(box: BBox(0.10, 0.30, 0.12, 0.50), keypoints: []),
            PersonPose(box: BBox(0.75, 0.30, 0.12, 0.50), keypoints: []),
          ],
        ),
      );
      // One pump delivers the stream frame to _onFrame (setState), a second
      // rebuilds the overlay with the new detections.
      await tester.pump();
      await tester.pump();

      // The overlay renders two player boxes plus the ball marker as
      // DecoratedBoxes (the net line is a plain ColoredBox), so three in total.
      final decorated = find.descendant(
        of: overlayFinder,
        matching: find.byType(DecoratedBox),
      );
      expect(decorated, findsNWidgets(3));

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'first-server picker sets who serves before the match starts',
    (tester) async {
      final vision = YoloVisionService();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraMatchScreen(
            visionService: vision,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
            matchControllerBuilder: MatchController.new,
          ),
        ),
      );
      await tester.pump();

      // The picker is shown before any point is scored.
      expect(find.text('First server:'), findsOneWidget);

      // Default: Player A serves (serve icon shown on A's side).
      await tester.tap(find.widgetWithText(ChoiceChip, 'B'));
      await tester.pump();

      // Feed the first scripted rally; Player B should now be recorded as the
      // first server of the match.
      for (final frame in demoMatchFrames().take(13)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      // A point was scored, so the picker disappears (first server locked in).
      expect(find.text('First server:'), findsNothing);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'best-of format picker sets the match length before it starts',
    (tester) async {
      final vision = YoloVisionService();
      final controller = MatchController();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraMatchScreen(
            visionService: vision,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
            matchControllerBuilder: () => controller,
          ),
        ),
      );
      await tester.pump();

      // The format picker is shown before any point is scored; default best-of-5.
      expect(find.text('Best of:'), findsOneWidget);
      expect(controller.score.bestOf, 5);

      // Pick best-of-3.
      await tester.tap(find.widgetWithText(ChoiceChip, '3'));
      await tester.pump();
      expect(controller.score.bestOf, 3);

      // Feed the first scripted rally to start the match.
      for (final frame in demoMatchFrames().take(13)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      // A point was scored, so the picker disappears (format locked in).
      expect(find.text('Best of:'), findsNothing);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'manual +point button hand-awards a missed rally to the score',
    (tester) async {
      final vision = YoloVisionService();
      final controller = MatchController();
      await tester.pumpWidget(
        MaterialApp(
          home: CameraMatchScreen(
            visionService: vision,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
            matchControllerBuilder: () => controller,
          ),
        ),
      );
      await tester.pump();

      // The call feed offers manual score-correction buttons once scoring is
      // live (no calibrator, so there is no warm-up).
      expect(find.text('Missed a point?'), findsOneWidget);
      expect(controller.score.pointsB, 0);

      // Tapping +B hands the point to Player B — the fix for a rally the vision
      // missed entirely.
      await tester.tap(find.widgetWithText(OutlinedButton, '+B'));
      await tester.pump();

      expect(controller.score.pointsB, 1);
      expect(controller.points.single.reason, PointReason.manual);

      await tester.pumpWidget(const SizedBox());
    },
  );
}
