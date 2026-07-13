/// Widget tests for the Match screen's real-footage demo mode: the video
/// playback surface with the recorded player/ball identification overlaid,
/// replayed in lock-step with the (test-driven) playback clock.
///
/// Follows the repo's injectable-seam pattern: an in-memory fixture loader (no
/// asset I/O — file-backed loads hang `testWidgets`) and a fake
/// [FootagePlayer] whose position the test advances by hand (no video plugin
/// channel headlessly).
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/benchmark/clip_fixture.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/features/match/footage_demo.dart';
import 'package:pong_ai/features/match/match_screen.dart';

/// A scripted stand-in for the video player: the test sets [positionMs].
class _FakeFootagePlayer implements FootagePlayer {
  int positionMs = 0;
  bool playing = false;
  bool disposed = false;
  int seeks = 0;

  /// When positive, [seekToStart] does *not* rewind [positionMs] immediately:
  /// the next [staleReadsAfterSeek] position reads still report the old value
  /// before it snaps to 0 — mimicking web video's asynchronous seek, where the
  /// player keeps reporting end-of-clip for a few frames after seekTo(0).
  int staleReadsAfterSeek = 0;
  int _staleReadsLeft = 0;
  int _stalePos = 0;

  @override
  Future<void> initialize() async {}

  @override
  Duration get position {
    if (_staleReadsLeft > 0) {
      _staleReadsLeft--;
      if (_staleReadsLeft == 0) positionMs = 0;
      return Duration(milliseconds: _stalePos);
    }
    return Duration(milliseconds: positionMs);
  }

  @override
  bool get isPlaying => playing;

  @override
  double get aspectRatio => 16 / 9;

  @override
  Future<void> play() async => playing = true;

  @override
  Future<void> pause() async => playing = false;

  @override
  Future<void> seekToStart() async {
    seeks++;
    if (staleReadsAfterSeek > 0) {
      _stalePos = positionMs;
      _staleReadsLeft = staleReadsAfterSeek;
    } else {
      positionMs = 0;
    }
  }

  @override
  Widget get view => const ColoredBox(
        key: ValueKey('fakeFootageVideo'),
        color: Colors.black,
      );

  @override
  Future<void> dispose() async => disposed = true;
}

PersonPose _person(double left) => PersonPose(
      box: BBox(left, 0.3, 0.12, 0.45),
      keypoints: const [Keypoint(0.5, 0.5, 0.9)],
    );

Detection _ball(double x) => Detection(
      label: 'sports ball',
      confidence: 0.9,
      box: BBox(x - 0.01, 0.49, 0.02, 0.02),
    );

/// Three recorded frames: ball + both players, ball + both players, then a
/// detector dropout (people only).
ClipFixture _fixture() => ClipFixture(
      name: 'test_footage',
      netX: 0.5,
      groundTruth: const ClipGroundTruth(pointsA: 0, pointsB: 0),
      frames: [
        FrameResult(
          timestampMs: 0,
          ball: _ball(0.30),
          people: [_person(0.05), _person(0.80)],
        ),
        FrameResult(
          timestampMs: 50,
          ball: _ball(0.40),
          people: [_person(0.05), _person(0.80)],
        ),
        FrameResult(
          timestampMs: 100,
          people: [_person(0.05), _person(0.80)],
        ),
      ],
    );

const _demo = FootageDemo(
  videoAsset: 'unused.mp4',
  fixtureAsset: 'unused.json',
);

void main() {
  Future<_FakeFootagePlayer> pumpFootageScreen(WidgetTester tester) async {
    final player = _FakeFootagePlayer();
    await tester.pumpWidget(
      MaterialApp(
        home: MatchScreen(
          footage: _demo,
          footagePlayerBuilder: () => player,
          footageFixtureLoader: (_) async => _fixture(),
        ),
      ),
    );
    // Let the async fixture load + player init complete.
    await tester.pump();
    await tester.pump();
    return player;
  }

  /// Advances the fake playback clock and pumps enough for the poll timer to
  /// emit and the resulting setState to land (stream frames rebuild on the
  /// *next* pump).
  Future<void> advanceTo(
    WidgetTester tester,
    _FakeFootagePlayer player,
    int positionMs,
  ) async {
    player.positionMs = positionMs;
    await tester.pump(const Duration(milliseconds: 20));
    await tester.pump();
  }

  testWidgets('shows the footage video with player and ball identification',
      (tester) async {
    final player = await pumpFootageScreen(tester);
    expect(find.byKey(const ValueKey('fakeFootageVideo')), findsOneWidget);
    expect(player.playing, isTrue, reason: 'footage auto-plays once ready');

    await advanceTo(tester, player, 10); // frame t=0 reached
    expect(find.byKey(const ValueKey('footageBall')), findsOneWidget);
    expect(find.byKey(const ValueKey('footagePerson0')), findsOneWidget);
    expect(find.byKey(const ValueKey('footagePerson1')), findsOneWidget);
    expect(find.text('Player A'), findsWidgets);
    expect(find.text('Player B'), findsWidgets);
    expect(find.text('ball locked'), findsOneWidget);
    expect(find.text('Bounces — rally: 0 · match: 0'), findsOneWidget);

    // Play out the remaining frames so the poll timer self-cancels.
    await advanceTo(tester, player, 200);
  });

  testWidgets('a detector dropout swaps the ball for the predicted ghost',
      (tester) async {
    final player = await pumpFootageScreen(tester);

    await advanceTo(tester, player, 60); // t=0 and t=50 (ball trajectory)
    expect(find.byKey(const ValueKey('footageBall')), findsOneWidget);

    await advanceTo(tester, player, 110); // t=100: no ball detected
    expect(find.byKey(const ValueKey('footageBall')), findsNothing);
    expect(
      find.byKey(const ValueKey('footageGhost')),
      findsOneWidget,
      reason: 'the Kalman prediction bridges the dropout',
    );
    expect(find.text('predicting…'), findsOneWidget);
  });

  testWidgets('replay tolerates a stale (asynchronously applied) seek',
      (tester) async {
    final player = await pumpFootageScreen(tester);
    // Play the whole clip out.
    await advanceTo(tester, player, 200);
    expect(find.byKey(const ValueKey('footageBall')), findsNothing);

    // Web video applies seekTo(0) asynchronously: the next few position reads
    // still report end-of-clip. Replay must wait for the rewind to land
    // instead of flushing every frame against the stale clock.
    player.staleReadsAfterSeek = 3;
    await tester.tap(find.byTooltip('Replay footage'));
    // Let the settle loop poll through the stale reads (25 ms apart).
    for (var i = 0; i < 6; i++) {
      await tester.pump(const Duration(milliseconds: 25));
    }
    expect(player.seeks, 1);
    expect(player.playing, isTrue);

    // The second viewing streams from the top: the t=0 frame's ball shows.
    await advanceTo(tester, player, 10);
    expect(find.byKey(const ValueKey('footageBall')), findsOneWidget);

    await advanceTo(tester, player, 200); // exhaust so no timer is pending
  });

  testWidgets('resolving the "who won?" prompt updates the call feed',
      (tester) async {
    final player = _FakeFootagePlayer();
    // A rally that is lost in flight: ball moves, then vanishes — the referee
    // cannot attribute it and queues an undetermined decision. The default
    // controller (maxGapFrames: 6) makes the loss fire within a few frames.
    final fixture = ClipFixture(
      name: 'undetermined_footage',
      netX: 0.5,
      groundTruth: const ClipGroundTruth(pointsA: 0, pointsB: 0),
      frames: [
        FrameResult(timestampMs: 0, ball: _ball(0.30)),
        FrameResult(timestampMs: 33, ball: _ball(0.35)),
        for (var i = 2; i < 12; i++) FrameResult(timestampMs: i * 33),
      ],
    );
    await tester.pumpWidget(
      MaterialApp(
        home: MatchScreen(
          footage: _demo,
          footagePlayerBuilder: () => player,
          footageFixtureLoader: (_) async => fixture,
          matchControllerBuilder: MatchController.new,
        ),
      ),
    );
    await tester.pump();
    await tester.pump();

    await advanceTo(tester, player, 400); // play past the ball loss
    expect(find.text('Who won this point?'), findsOneWidget);

    await tester.tap(find.widgetWithText(FilledButton, 'Player A'));
    await tester.pump();

    // The prompt clears, the point is awarded, and the feed entry flips from
    // "Undetermined" to the awarded player.
    expect(find.text('Who won this point?'), findsNothing);
    expect(find.text('• Player A — out of play'), findsOneWidget);
    expect(find.textContaining('Undetermined'), findsNothing);
  });

  testWidgets('a rally still in flight when the clip ends is resolved',
      (tester) async {
    final player = _FakeFootagePlayer();
    // A bounce on the right, ball still tracked on the clip's final frame:
    // the feed ends with the rally unresolved — the end-of-footage flush must
    // let the referee score it (right side never returned -> Player A).
    final fixture = ClipFixture(
      name: 'ends_mid_rally',
      netX: 0.5,
      groundTruth: const ClipGroundTruth(pointsA: 1, pointsB: 0),
      frames: [
        FrameResult(timestampMs: 0, ball: _ball(0.75)),
        const FrameResult(
          timestampMs: 33,
          ball: Detection(
            label: 'sports ball',
            confidence: 0.9,
            box: BBox(0.74, 0.69, 0.02, 0.02),
          ),
        ),
        FrameResult(timestampMs: 66, ball: _ball(0.75)), // bounce reported
      ],
    );
    await tester.pumpWidget(
      MaterialApp(
        home: MatchScreen(
          footage: _demo,
          footagePlayerBuilder: () => player,
          footageFixtureLoader: (_) async => fixture,
          matchControllerBuilder: MatchController.new,
        ),
      ),
    );
    await tester.pump();
    await tester.pump();

    await advanceTo(tester, player, 100); // plays out the whole clip
    expect(find.text('• Player A — not returned'), findsOneWidget);
  });

  testWidgets('pausing freezes the replay and replay restarts from the top',
      (tester) async {
    final player = await pumpFootageScreen(tester);
    await advanceTo(tester, player, 60);
    expect(find.byKey(const ValueKey('footageBall')), findsOneWidget);

    // Pause: the playback clock stalls, so no further frames flow.
    await tester.tap(find.byTooltip('Pause'));
    await tester.pump();
    expect(player.playing, isFalse);
    expect(find.byTooltip('Play'), findsOneWidget);

    // Replay rewinds the video and re-runs the detections from the start.
    await tester.tap(find.byTooltip('Replay footage'));
    await tester.pump();
    expect(player.seeks, 1);
    expect(player.playing, isTrue);

    await advanceTo(tester, player, 10);
    expect(find.byKey(const ValueKey('footageBall')), findsOneWidget);

    // Exhaust the frames so no poll timer is left pending.
    await advanceTo(tester, player, 200);
  });
}
