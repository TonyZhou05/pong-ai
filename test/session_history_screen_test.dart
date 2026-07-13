import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/history/session_history_store.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';
import 'package:pong_ai/core/vision/vision_service.dart';
import 'package:pong_ai/features/history/progress_chart.dart';
import 'package:pong_ai/features/history/session_history_screen.dart';
import 'package:pong_ai/features/training/training_screen.dart';

/// An in-memory [SessionHistoryStore] with no real file I/O, so widget tests can
/// pump/settle normally (real `dart:io` from a `testWidgets` body only advances
/// under `runAsync`). The on-disk store is covered by session_history_store_test.
class FakeHistoryStore extends SessionHistoryStore {
  FakeHistoryStore() : super(Directory.systemTemp);

  final List<StoredSession> saved = [];

  @override
  Future<List<StoredSession>> list() async {
    final copy = List<StoredSession>.of(saved)
      ..sort((a, b) => b.savedAt.compareTo(a.savedAt));
    return copy;
  }

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

  @override
  Future<bool> delete(String id) async {
    final before = saved.length;
    saved.removeWhere((s) => s.id == id);
    return saved.length != before;
  }
}

/// A camera-free [VisionService] whose frames the test pushes manually.
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

  Future<void> finish() async {
    if (!_controller.isClosed) await _controller.close();
  }
}

StoredSession _matchSession({
  required String id,
  required DateTime at,
  int gamesA = 0,
  int gamesB = 0,
  int pointsA = 0,
  int pointsB = 0,
}) {
  return StoredSession(
    id: id,
    kind: SessionKind.match,
    savedAt: at,
    report: {
      'score': {
        'gamesA': gamesA,
        'gamesB': gamesB,
        'pointsA': pointsA,
        'pointsB': pointsB,
      },
    },
  );
}

void main() {
  group('sessionHeadline', () {
    test('match with games shows the games score', () {
      final headline = sessionHeadline(
        _matchSession(id: 'm', at: DateTime(2026), gamesA: 3, gamesB: 1),
      );
      expect(headline, 'Match · 3–1 games');
    });

    test('match without games falls back to the point score', () {
      final headline = sessionHeadline(
        _matchSession(id: 'm', at: DateTime(2026), pointsA: 7, pointsB: 5),
      );
      expect(headline, 'Match · 7–5');
    });

    test('training shows the session grade and shot count', () {
      final headline = sessionHeadline(
        StoredSession(
          id: 't',
          kind: SessionKind.training,
          savedAt: DateTime(2026),
          report: {
            'session': {'overallGrade': 'B', 'shotCount': 6},
          },
        ),
      );
      expect(headline, 'Training · grade B · 6 shots');
    });

    test('unrecognizable report degrades to the bare kind', () {
      expect(
        sessionHeadline(
          StoredSession(
            id: 'x',
            kind: SessionKind.match,
            savedAt: DateTime(2026),
            report: const {},
          ),
        ),
        'Match',
      );
    });
  });

  group('SessionHistoryScreen', () {
    testWidgets('empty store shows the empty-state hint', (tester) async {
      await tester.pumpWidget(
        MaterialApp(home: SessionHistoryScreen(store: FakeHistoryStore())),
      );
      await tester.pumpAndSettle();

      expect(find.textContaining('No saved sessions yet'), findsOneWidget);
    });

    testWidgets('lists saved sessions newest-first and opens a detail view',
        (tester) async {
      final store = FakeHistoryStore();
      await store.save(
        kind: SessionKind.match,
        report: {
          'score': {'gamesA': 3, 'gamesB': 2},
        },
        at: DateTime(2026, 1, 1, 9),
      );
      await store.save(
        kind: SessionKind.training,
        report: {
          'session': {'overallGrade': 'A', 'shotCount': 8},
        },
        at: DateTime(2026, 1, 2, 9),
      );

      await tester.pumpWidget(
        MaterialApp(home: SessionHistoryScreen(store: store)),
      );
      await tester.pumpAndSettle();

      final tiles = find.byType(ListTile);
      expect(tiles, findsNWidgets(2));
      // Newest (training, Jan 2) sits above the match (Jan 1).
      final headlines = tester
          .widgetList<ListTile>(tiles)
          .map((t) => (t.title! as Text).data)
          .toList();
      expect(headlines.first, 'Training · grade A · 8 shots');
      expect(headlines.last, 'Match · 3–2 games');

      await tester.tap(find.text('Match · 3–2 games'));
      await tester.pumpAndSettle();

      // The detail view renders the raw report JSON.
      expect(find.textContaining('"gamesA": 3'), findsOneWidget);
    });

    testWidgets('shows a training-progress header once two drills are saved',
        (tester) async {
      final store = FakeHistoryStore();
      await store.save(
        kind: SessionKind.training,
        report: {
          'session': {
            'overallGrade': 'C',
            'shotCount': 6,
            'averageScore': 0.50,
          },
        },
        at: DateTime(2026, 1, 1, 9),
      );
      await store.save(
        kind: SessionKind.training,
        report: {
          'session': {
            'overallGrade': 'A',
            'shotCount': 8,
            'averageScore': 0.80,
          },
        },
        at: DateTime(2026, 1, 5, 9),
      );

      await tester.pumpWidget(
        MaterialApp(home: SessionHistoryScreen(store: store)),
      );
      await tester.pumpAndSettle();

      expect(find.text('Training progress'), findsOneWidget);
      expect(find.text('50% → 80%'), findsOneWidget);
      expect(find.byIcon(Icons.trending_up), findsOneWidget);
      // The visual progression chart renders alongside the text summary.
      expect(find.byType(ProgressChartView), findsOneWidget);
    });

    testWidgets('surfaces a recurring coaching focus in the trends card',
        (tester) async {
      final store = FakeHistoryStore();
      for (var i = 0; i < 2; i++) {
        await store.save(
          kind: SessionKind.training,
          report: {
            'session': {
              'overallGrade': 'C',
              'shotCount': 6,
              'averageScore': 0.50 + i * 0.1,
            },
            'coaching': {'focus': 'Rhythm'},
          },
          at: DateTime(2026, 1, 1 + i, 9),
        );
      }

      await tester.pumpWidget(
        MaterialApp(home: SessionHistoryScreen(store: store)),
      );
      await tester.pumpAndSettle();

      expect(
        find.textContaining('Keep working on rhythm (2/2 drills)'),
        findsOneWidget,
      );
    });

    testWidgets('no trends header with a single saved drill', (tester) async {
      final store = FakeHistoryStore();
      await store.save(
        kind: SessionKind.training,
        report: {
          'session': {
            'overallGrade': 'B',
            'shotCount': 6,
            'averageScore': 0.65,
          },
        },
        at: DateTime(2026, 1, 1, 9),
      );

      await tester.pumpWidget(
        MaterialApp(home: SessionHistoryScreen(store: store)),
      );
      await tester.pumpAndSettle();

      expect(find.text('Training progress'), findsNothing);
      expect(find.byType(ListTile), findsOneWidget);
    });

    testWidgets('delete removes a session from the list', (tester) async {
      final store = FakeHistoryStore();
      await store.save(
        kind: SessionKind.match,
        report: {
          'score': {'gamesA': 3, 'gamesB': 0},
        },
        at: DateTime(2026, 1, 1, 9),
      );

      await tester.pumpWidget(
        MaterialApp(home: SessionHistoryScreen(store: store)),
      );
      await tester.pumpAndSettle();
      expect(find.byType(ListTile), findsOneWidget);

      await tester.tap(find.byIcon(Icons.delete_outline));
      await tester.pumpAndSettle();

      expect(find.byType(ListTile), findsNothing);
      expect(find.textContaining('No saved sessions yet'), findsOneWidget);
      expect(store.saved, isEmpty);
    });
  });

  testWidgets('training report Save to history persists a session',
      (tester) async {
    final store = FakeHistoryStore();
    final fake = FakeVisionService();
    await tester.pumpWidget(
      MaterialApp(
        home: TrainingScreen(
          visionServiceBuilder: () => fake,
          historyStoreLoader: () async => store,
        ),
      ),
    );
    await tester.pump();

    // One graded stroke, then end the session so the report shows.
    for (final frame in trainingSessionFrames().take(14)) {
      fake.emit(frame);
      await tester.pump();
    }
    await fake.finish();
    await tester.pump();

    expect(find.text('Session complete'), findsOneWidget);
    await tester.ensureVisible(find.text('Save to history'));
    await tester.tap(find.text('Save to history'));
    await tester.pumpAndSettle();

    expect(find.text('Saved to history'), findsOneWidget);
    expect(store.saved, hasLength(1));
    expect(store.saved.single.kind, SessionKind.training);
    expect(store.saved.single.report['session'], isA<Map>());

    await tester.pumpWidget(const SizedBox());
  });
}
