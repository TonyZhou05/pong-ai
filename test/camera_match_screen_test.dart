import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/analysis/table_calibrator.dart';
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
      // The panel now also surfaces the prioritized coaching cue, matching the
      // demo MatchScreen summary (both players served, so insights have data).
      expect(find.text('Coaching'), findsOneWidget);
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
    'Play again resets the score and resumes scoring on the same screen',
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
            // Short best-of-one, 3-point game so the demo rallies end the match.
            matchControllerBuilder: () =>
                MatchController(engine: ScoringEngine(pointsPerGame: 3, bestOf: 1)),
          ),
        ),
      );
      await tester.pump();

      for (final frame in demoMatchFrames()) {
        vision.onFrame(frame);
        await tester.pump();
      }
      expect(find.textContaining('wins the match'), findsOneWidget);

      // Tap "Play again": the match-over panel is replaced by the live call
      // feed and the score returns to the pre-match waiting state.
      await tester.ensureVisible(find.text('Play again'));
      await tester.tap(find.text('Play again'));
      await tester.pump();
      expect(find.textContaining('wins the match'), findsNothing);
      expect(find.text('Waiting for the first rally…'), findsOneWidget);

      // The camera stream resumed, so the next scripted rally scores again.
      for (final frame in demoMatchFrames().take(13)) {
        vision.onFrame(frame);
        await tester.pump();
      }
      expect(find.textContaining('Player A — not returned'), findsOneWidget);

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
    'live tracking overlay shows a km/h readout beside the moving ball',
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
            // No calibrator so the ball-speed estimator measures immediately.
            matchControllerBuilder: MatchController.new,
          ),
        ),
      );
      await tester.pump();

      // A single ball frame has no displacement yet -> no live speed reading.
      vision.onFrame(
        const FrameResult(
          timestampMs: 33,
          ball: Detection(
            label: 'ball',
            confidence: 0.9,
            box: BBox(0.30, 0.50, 0.02, 0.02),
          ),
          people: [],
        ),
      );
      await tester.pump();
      await tester.pump();
      expect(find.textContaining('km/h'), findsNothing);

      // A second ball frame 33 ms later, moved along the table, yields a live
      // reading rendered beside the ball.
      vision.onFrame(
        const FrameResult(
          timestampMs: 66,
          ball: Detection(
            label: 'ball',
            confidence: 0.9,
            box: BBox(0.42, 0.50, 0.02, 0.02),
          ),
          people: [],
        ),
      );
      await tester.pump();
      await tester.pump();
      expect(find.textContaining('km/h'), findsOneWidget);

      // When the ball drops out, the (now stale) speed label is hidden with it.
      vision.onFrame(
        const FrameResult(timestampMs: 99, ball: null, people: []),
      );
      await tester.pump();
      await tester.pump();
      expect(find.textContaining('km/h'), findsNothing);

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
    'game-length picker sets the per-game point target before it starts',
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

      // The game-length picker is shown before any point; default 11-point.
      expect(find.text('Play to:'), findsOneWidget);
      expect(controller.score.pointsPerGame, 11);

      // Pick the classic 21-point game length.
      await tester.tap(find.widgetWithText(ChoiceChip, '21'));
      await tester.pump();
      expect(controller.score.pointsPerGame, 21);

      // Feed the first scripted rally to start the match.
      for (final frame in demoMatchFrames().take(13)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      // A point was scored, so the picker disappears (format locked in).
      expect(find.text('Play to:'), findsNothing);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'live placement warning appears on poor tracking and clears when fixed',
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
            // No calibrator so tracking-health accrues on every live frame.
            matchControllerBuilder: MatchController.new,
          ),
        ),
      );
      await tester.pump();

      // No warning before enough evidence has accumulated.
      expect(find.text('Poor tracking'), findsNothing);

      // Feed a run of empty frames (no ball, no players) — a badly-placed phone
      // that sees neither the table nor the ball. Once the trailing window is
      // full of poor frames the nudge appears.
      for (var t = 0; t < 12; t++) {
        vision.onFrame(FrameResult(timestampMs: 33 * (t + 1), people: const []));
        await tester.pump();
      }
      expect(find.text('Poor tracking'), findsOneWidget);
      expect(find.textContaining('out of frame'), findsOneWidget);

      // Now the phone is repositioned: healthy frames (both players + a clear
      // ball) slide the poor frames out of the window and the nudge clears.
      for (var t = 12; t < 45; t++) {
        vision.onFrame(
          FrameResult(
            timestampMs: 33 * (t + 1),
            ball: const Detection(
              label: 'ball',
              confidence: 0.95,
              box: BBox(0.20, 0.50, 0.02, 0.02),
            ),
            people: const [
              PersonPose(box: BBox(0.10, 0.30, 0.12, 0.50), keypoints: []),
              PersonPose(box: BBox(0.75, 0.30, 0.12, 0.50), keypoints: []),
            ],
          ),
        );
        await tester.pump();
      }
      expect(find.text('Poor tracking'), findsNothing);

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

  testWidgets(
    'a completed game prompts the players to change ends, then clears',
    (tester) async {
      final vision = YoloVisionService();
      // A 3-point best-of-3 with end switching on: game 1 completing swaps the
      // internal mapping, so the players must physically change ends.
      final controller = MatchController(
        engine: ScoringEngine(pointsPerGame: 3, bestOf: 3),
        switchEndsBetweenGames: true,
      );
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

      expect(find.text('CHANGE ENDS'), findsNothing);

      // Feed the demo rallies until game 1 completes (A reaches 3 points).
      final frames = demoMatchFrames().iterator;
      while (controller.score.gamesA == 0 && frames.moveNext()) {
        vision.onFrame(frames.current);
        await tester.pump();
      }
      expect(controller.score.gamesA, 1);
      await tester.pump(); // rebuild the scoreboard with the pending flag set

      // The prompt now tells the players to swap sides for game 2.
      expect(find.text('CHANGE ENDS'), findsOneWidget);

      // Playing on: the first scored point of game 2 clears the prompt.
      while (controller.score.pointsA + controller.score.pointsB == 0 &&
          frames.moveNext()) {
        vision.onFrame(frames.current);
        await tester.pump();
      }
      expect(controller.score.pointsA + controller.score.pointsB, greaterThan(0));
      await tester.pump(); // rebuild the scoreboard with the flag cleared
      expect(find.text('CHANGE ENDS'), findsNothing);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'a completed game speaks a change-ends cue exactly once, then re-arms',
    (tester) async {
      final vision = YoloVisionService();
      final spoken = <String>[];
      // A 3-point best-of-3 with end switching on: game 1 completing flips the
      // internal mapping, so the players must physically swap sides — and a
      // table-side player who can't read the banner should hear it.
      final controller = MatchController(
        engine: ScoringEngine(pointsPerGame: 3, bestOf: 3),
        switchEndsBetweenGames: true,
      );
      await tester.pumpWidget(
        MaterialApp(
          home: CameraMatchScreen(
            visionService: vision,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
            matchControllerBuilder: () => controller,
            onAnnounce: spoken.add,
          ),
        ),
      );
      await tester.pump();

      // Feed the demo rallies until game 1 completes (A reaches 3 points).
      final frames = demoMatchFrames().iterator;
      while (controller.score.gamesA == 0 && frames.moveNext()) {
        vision.onFrame(frames.current);
        await tester.pump();
      }
      expect(controller.score.gamesA, 1);
      await tester.pump();

      // The change-ends cue fired through the injected sink and is captioned.
      expect(spoken, contains('Change ends.'));
      expect(spoken.where((c) => c == 'Change ends.'), hasLength(1));
      expect(find.text('Change ends.'), findsOneWidget);

      // Playing on: the flag clears on the next scored point, and later points
      // don't re-speak the (now stale) cue.
      final spokenBefore = spoken.length;
      while (controller.score.pointsA + controller.score.pointsB == 0 &&
          frames.moveNext()) {
        vision.onFrame(frames.current);
        await tester.pump();
      }
      expect(
        controller.score.pointsA + controller.score.pointsB,
        greaterThan(0),
      );
      expect(spoken.skip(spokenBefore), isNot(contains('Change ends.')));

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'a stalled calibration prompts the user to reposition the phone',
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
            // A calibrator that never gets ball samples + a tiny stall budget.
            matchControllerBuilder: () => MatchController(
              calibrator: TableCalibrator(minBallSamples: 20),
              calibrationStallFrames: 3,
            ),
          ),
        ),
      );
      await tester.pump();

      // Warm-up banner shows a progress read-out before the stall budget runs
      // out (no ball has been seen, so 0%).
      expect(find.textContaining('Calibrating table'), findsOneWidget);
      expect(find.textContaining('0%'), findsOneWidget);

      // Feed ball-less frames: calibration can never complete, so once the
      // stall budget is exceeded the banner becomes an actionable prompt.
      for (var i = 0; i < 4; i++) {
        vision.onFrame(FrameResult(timestampMs: i * 33));
        await tester.pump();
      }

      expect(find.textContaining('reposition the phone'), findsOneWidget);
      expect(find.textContaining('Calibrating table'), findsNothing);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'a scored point speaks the umpire call and captions it under the score',
    (tester) async {
      final vision = YoloVisionService();
      final spoken = <String>[];
      await tester.pumpWidget(
        MaterialApp(
          home: CameraMatchScreen(
            visionService: vision,
            cameraPreviewBuilder: (_, __) => const ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(),
            ),
            // No calibrator so the first scripted rally scores immediately.
            matchControllerBuilder: MatchController.new,
            onAnnounce: spoken.add,
          ),
        ),
      );
      await tester.pump();

      // Nothing spoken before the first point.
      expect(spoken, isEmpty);

      // The first scripted rally awards Player A a point (1–0).
      for (final frame in demoMatchFrames().take(13)) {
        vision.onFrame(frame);
        await tester.pump();
      }

      // The umpire call fired through the injected sink and is captioned on
      // screen so the table-side player is told the score.
      expect(spoken, contains('Player A, 1–0.'));
      expect(find.text('Player A, 1–0.'), findsOneWidget);

      await tester.pumpWidget(const SizedBox());
    },
  );

  testWidgets(
    'pausing freezes scoring during a break and resuming restores it',
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

      // Pause play via the AppBar control.
      expect(find.byTooltip('Pause play'), findsOneWidget);
      await tester.tap(find.byTooltip('Pause play'));
      await tester.pump();
      expect(find.text('PAUSED'), findsOneWidget);
      expect(find.byTooltip('Resume play'), findsOneWidget);

      // A full rally arrives during the break — nothing is scored.
      for (final frame in demoMatchFrames().take(13)) {
        vision.onFrame(frame);
        await tester.pump();
      }
      expect(find.textContaining('not returned'), findsNothing);
      expect(find.text('Waiting for the first rally…'), findsOneWidget);

      // Resume and feed a fresh rally (the first scripted rally, which scores
      // A) — with a clean trajectory it now scores.
      await tester.tap(find.byTooltip('Resume play'));
      await tester.pump();
      expect(find.text('PAUSED'), findsNothing);
      for (final frame in demoMatchFrames().take(13)) {
        vision.onFrame(frame);
        await tester.pump();
      }
      expect(find.textContaining('Player A — not returned'), findsOneWidget);

      await tester.pumpWidget(const SizedBox());
    },
  );
}
