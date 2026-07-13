import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/app.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/history/session_history_store.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
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

  testWidgets('match summary Save to history persists a match session',
      (tester) async {
    final fake = FakeVisionService();
    final store = FakeHistoryStore();
    // A short best-of-one, 3-point game so a few scripted rallies end the match
    // and surface the summary panel (the full demo replay never completes one).
    await tester.pumpWidget(
      MaterialApp(
        home: MatchScreen(
          visionServiceBuilder: () => fake,
          matchControllerBuilder: () =>
              MatchController(engine: ScoringEngine(pointsPerGame: 3, bestOf: 1)),
          historyStoreLoader: () async => store,
        ),
      ),
    );
    await tester.pump();

    // The demo's first four rallies award A, B, A, A → Player A wins 3–1 (by
    // two), completing the one-game match and showing the summary panel.
    for (final frame in demoMatchFrames()) {
      fake.emit(frame);
      await tester.pump();
    }

    expect(find.textContaining('wins the match'), findsOneWidget);
    await tester.ensureVisible(find.text('Save to history'));
    await tester.tap(find.text('Save to history'));
    await tester.pumpAndSettle();

    expect(find.text('Saved to history'), findsOneWidget);
    expect(store.saved, hasLength(1));
    expect(store.saved.single.kind, SessionKind.match);
    expect(store.saved.single.report['score'], isA<Map>());

    await tester.pumpWidget(const SizedBox());
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
