/// Widget tests for the rally-labeling workbench: manifest and label store
/// injected in-memory (asset/prefs I/O hangs `testWidgets`).
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/labels/rally_label_store.dart';
import 'package:pong_ai/features/labeling/labeling_screen.dart';
import 'package:pong_ai/features/match/footage_demo.dart';

List<FootageMatch> _corpus() => const [
      FootageMatch(
        id: 'test_9_r1',
        title: 'Test 9 — Rally 1',
        demo: FootageDemo(
          videoAsset: 'assets/footage/test_9_r1.mp4',
          fixtureAsset: 'assets/footage/test_9_r1.json',
        ),
        durationMs: 8000,
        bounces: 6,
        pointsA: 0,
        pointsB: 1,
        pointWinners: ['b'],
      ),
      FootageMatch(
        id: 'test_9_r2',
        title: 'Test 9 — Rally 2',
        demo: FootageDemo(
          videoAsset: 'assets/footage/test_9_r2.mp4',
          fixtureAsset: 'assets/footage/test_9_r2.json',
        ),
        durationMs: 5000,
        bounces: 3,
      ),
      // Combined sets are excluded from labeling.
      FootageMatch(
        id: 'test_9_full',
        title: 'Test 9 — Full set (2 rallies)',
        demo: FootageDemo(
          videoAsset: 'assets/footage/test_9_full.mp4',
          fixtureAsset: 'assets/footage/test_9_full.json',
        ),
        durationMs: 15000,
      ),
    ];

void main() {
  testWidgets('lists rally clips with the pipeline call; excludes full sets',
      (tester) async {
    final store = InMemoryRallyLabelStore();
    await tester.pumpWidget(
      MaterialApp(
        home: LabelingScreen(
          manifestLoader: () async => _corpus(),
          labelStore: store,
        ),
      ),
    );
    await tester.pump();

    expect(find.text('Test 9 — Rally 1'), findsOneWidget);
    expect(find.text('Test 9 — Rally 2'), findsOneWidget);
    expect(find.textContaining('Full set'), findsNothing);
    expect(
      find.textContaining('pipeline says: 0–1 (B)'),
      findsOneWidget,
      reason: 'the existing label shows so corrections are unambiguous',
    );
    expect(
      find.textContaining('no call / unlabeled'),
      findsOneWidget,
      reason: 'an unlabeled clip says so',
    );
    expect(find.text('Label rallies (0/2)'), findsOneWidget);
  });

  testWidgets('selecting winner and reason persists the label',
      (tester) async {
    final store = InMemoryRallyLabelStore();
    await tester.pumpWidget(
      MaterialApp(
        home: LabelingScreen(
          manifestLoader: () async => _corpus(),
          labelStore: store,
        ),
      ),
    );
    await tester.pump();

    // Winner: Player A on the first rally card.
    await tester.tap(find.text('Player A (left)').first);
    await tester.pump();

    // Reason: open the first dropdown and pick "out of bounds".
    await tester.tap(find.text('Why (what the loser did)').first);
    await tester.pumpAndSettle();
    await tester.tap(find.text('Hit it out — long or wide').last);
    await tester.pumpAndSettle();

    final labels = await store.load();
    expect(labels, hasLength(1));
    final label = labels['test_9_r1']!;
    expect(label.winner, 'a');
    expect(label.reason, RallyLabelReason.outOfBounds);
    expect(find.text('Label rallies (1/2)'), findsOneWidget);

    // Export contains the label, keyed for the training tooling.
    final json = await store.exportJson();
    expect(json, contains('"clipId": "test_9_r1"'));
    expect(json, contains('"winner": "a"'));
    expect(json, contains('"reason": "outOfBounds"'));
  });
}
