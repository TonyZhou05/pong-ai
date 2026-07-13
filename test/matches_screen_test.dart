/// Widget tests for the Matches screen: the real-footage match corpus list.
/// The manifest loader is injected in-memory (asset I/O hangs `testWidgets`).
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/features/match/footage_demo.dart';
import 'package:pong_ai/features/matches/matches_screen.dart';

List<FootageMatch> _corpus() => const [
      FootageMatch(
        id: 'test_2_r1',
        title: 'Test 2 — Rally 1',
        demo: FootageDemo(
          videoAsset: 'assets/footage/test_2_r1.mp4',
          fixtureAsset: 'assets/footage/test_2_r1.json',
        ),
        durationMs: 12000,
        bounces: 16,
        source: 'OpenTTGames',
      ),
      FootageMatch(
        id: 'test_3_r4',
        title: 'Test 3 — Rally 4',
        demo: FootageDemo(
          videoAsset: 'assets/footage/test_3_r4.mp4',
          fixtureAsset: 'assets/footage/test_3_r4.json',
        ),
        durationMs: 11000,
        bounces: 12,
        source: 'OpenTTGames',
      ),
    ];

void main() {
  testWidgets('lists every corpus entry with its duration and bounces',
      (tester) async {
    await tester.pumpWidget(
      MaterialApp(home: MatchesScreen(manifestLoader: () async => _corpus())),
    );
    await tester.pump();

    expect(find.text('Matches'), findsOneWidget);
    expect(find.text('Test 2 — Rally 1'), findsOneWidget);
    expect(find.text('Test 3 — Rally 4'), findsOneWidget);
    expect(find.text('12s · 16 bounces · OpenTTGames'), findsOneWidget);
    expect(find.text('11s · 12 bounces · OpenTTGames'), findsOneWidget);
  });

  testWidgets('tapping an entry opens the footage match screen',
      (tester) async {
    await tester.pumpWidget(
      MaterialApp(home: MatchesScreen(manifestLoader: () async => _corpus())),
    );
    await tester.pump();

    await tester.tap(find.text('Test 2 — Rally 1'));
    await tester.pump(); // start the push transition
    await tester.pump(const Duration(milliseconds: 400)); // settle it

    // Lands on the footage Match screen (loading state headlessly — there is
    // no video runtime; the footage pipeline itself is covered by
    // footage_match_screen_test.dart with injected fakes).
    expect(find.widgetWithText(AppBar, 'Match'), findsOneWidget);

    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('a manifest load failure surfaces an error message',
      (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: MatchesScreen(
          manifestLoader: () async => throw StateError('missing manifest'),
        ),
      ),
    );
    await tester.pump();

    expect(find.textContaining('Could not load the match list'), findsOneWidget);
  });
}
