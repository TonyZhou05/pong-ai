import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/tracking_quality.dart';
import 'package:pong_ai/core/vision/detection.dart';

PersonPose _person(double left) =>
    PersonPose(box: BBox(left, 0.3, 0.1, 0.4), keypoints: const []);

FrameResult _frame(
  int t, {
  double? ballConf,
  int players = 0,
}) =>
    FrameResult(
      timestampMs: t,
      ball: ballConf == null
          ? null
          : Detection(
              label: 'ball',
              confidence: ballConf,
              box: const BBox(0.5, 0.5, 0, 0),
            ),
      people: [for (var i = 0; i < players; i++) _person(0.1 + i * 0.6)],
    );

void main() {
  test('no data before any frame', () {
    final a = TrackingQualityAnalyzer();
    expect(a.hasData, isFalse);
    expect(a.qualityScore, 0);
    expect(a.grade, 'N/A');
    expect(a.hint, contains('No frames'));
  });

  test('perfect tracking scores an A and reports healthy', () {
    final a = TrackingQualityAnalyzer();
    for (var t = 0; t < 10; t++) {
      a.observe(_frame(t, ballConf: 0.95, players: 2));
    }
    expect(a.frameCount, 10);
    expect(a.ballDetectionRate, 1.0);
    expect(a.averageBallConfidence, closeTo(0.95, 1e-9));
    expect(a.twoPlayerRate, 1.0);
    // 0.4*1 + 0.2*0.95 + 0.4*1 == 0.99
    expect(a.qualityScore, closeTo(0.99, 1e-9));
    expect(a.grade, 'A');
    expect(a.hint, contains('healthy'));
  });

  test('component rates are fractions of observed frames', () {
    final a = TrackingQualityAnalyzer();
    a.observe(_frame(0, ballConf: 0.8, players: 2)); // ball + both
    a.observe(_frame(1, ballConf: 0.6, players: 1)); // ball + one
    a.observe(_frame(2, players: 0)); // nothing
    a.observe(_frame(3, players: 2)); // both, no ball

    expect(a.frameCount, 4);
    expect(a.ballDetectionRate, closeTo(0.5, 1e-9)); // 2/4
    expect(a.averageBallConfidence, closeTo(0.7, 1e-9)); // (0.8+0.6)/2
    expect(a.anyPlayerRate, closeTo(0.75, 1e-9)); // 3/4
    expect(a.twoPlayerRate, closeTo(0.5, 1e-9)); // 2/4
  });

  test('missing second player drives the hint and lowers the grade', () {
    final a = TrackingQualityAnalyzer();
    // Ball always seen with high confidence, but only ever one player visible.
    for (var t = 0; t < 10; t++) {
      a.observe(_frame(t, ballConf: 0.9, players: 1));
    }
    // 0.4*1 + 0.2*0.9 + 0.4*0 == 0.58 -> grade C
    expect(a.qualityScore, closeTo(0.58, 1e-9));
    expect(a.twoPlayerRate, 0.0);
    expect(a.hint, contains('Both players'));
  });

  test('frequent ball loss drives the hint when players are fine', () {
    final a = TrackingQualityAnalyzer();
    for (var t = 0; t < 10; t++) {
      // Both players always visible; ball seen only 2/10 frames.
      a.observe(_frame(t, ballConf: t < 2 ? 0.9 : null, players: 2));
    }
    expect(a.twoPlayerRate, 1.0);
    expect(a.ballDetectionRate, closeTo(0.2, 1e-9));
    expect(a.hint, contains('ball is frequently lost'));
  });

  test('training mode scores on any-player visibility, not both ends', () {
    final match = TrackingQualityAnalyzer();
    final training = TrackingQualityAnalyzer(requireBothPlayers: false);
    // A lone practising player: ball always seen, only ever one person in view.
    for (var t = 0; t < 10; t++) {
      match.observe(_frame(t, ballConf: 0.9, players: 1));
      training.observe(_frame(t, ballConf: 0.9, players: 1));
    }
    expect(training.playerVisibilityRate, 1.0); // any-player
    expect(match.playerVisibilityRate, 0.0); // both-players
    // 0.4*1 + 0.2*0.9 + 0.4*1 == 0.98 -> A for training, C for the match view.
    expect(training.qualityScore, closeTo(0.98, 1e-9));
    expect(training.grade, 'A');
    expect(match.grade, 'C');
    expect(training.hint, contains('healthy'));
  });

  test('training-mode hint and report speak to a single player', () {
    final a = TrackingQualityAnalyzer(requireBothPlayers: false);
    // Player frequently out of frame drives the single-player hint.
    for (var t = 0; t < 10; t++) {
      a.observe(_frame(t, ballConf: 0.9, players: t < 3 ? 1 : 0));
    }
    expect(a.playerVisibilityRate, closeTo(0.3, 1e-9));
    expect(a.hint, contains('You are often out of frame'));
    expect(a.report(), contains('player visible in 30% of frames'));
  });

  group('live trailing-window placement signals', () {
    test('no recent data until recentMinFrames observed', () {
      final a = TrackingQualityAnalyzer(recentMinFrames: 5);
      for (var t = 0; t < 4; t++) {
        a.observe(_frame(t, players: 0)); // poor frames, but too few
      }
      expect(a.hasRecentData, isFalse);
      expect(a.isPlacementPoor, isFalse); // not enough evidence yet
      a.observe(_frame(4, players: 0));
      expect(a.hasRecentData, isTrue);
      expect(a.isPlacementPoor, isTrue);
    });

    test('recent window flags poor placement, then recovers when fixed', () {
      final a = TrackingQualityAnalyzer(recentWindow: 10, recentMinFrames: 5);
      // 10 poor frames: no players, no ball -> trailing score 0.
      for (var t = 0; t < 10; t++) {
        a.observe(_frame(t, players: 0));
      }
      expect(a.isPlacementPoor, isTrue);
      expect(a.recentGrade, 'F');
      expect(a.recentHint, contains('Both players'));

      // 10 good frames slide the poor ones out of the window entirely.
      for (var t = 10; t < 20; t++) {
        a.observe(_frame(t, ballConf: 0.95, players: 2));
      }
      expect(a.recentBallDetectionRate, 1.0);
      expect(a.recentPlayerVisibilityRate, 1.0);
      expect(a.recentQualityScore, closeTo(0.99, 1e-9));
      expect(a.isPlacementPoor, isFalse);
      expect(a.recentGrade, 'A');

      // The cumulative (whole-session) score is still dragged down by the bad
      // first half, so it lags well behind the recovered live window — which is
      // exactly why the live nudge uses a trailing window rather than the total.
      expect(a.qualityScore, lessThan(a.recentQualityScore));
      expect(a.qualityScore, closeTo(0.59, 1e-9)); // 0.4*.5 + 0.2*.95 + 0.4*.5
    });

    test('recent window is bounded and reset clears it', () {
      final a = TrackingQualityAnalyzer(recentWindow: 3, recentMinFrames: 1);
      for (var t = 0; t < 10; t++) {
        a.observe(_frame(t, ballConf: 0.9, players: 2));
      }
      // Only the last 3 frames feed the recent rate even after 10 observed.
      expect(a.recentBallDetectionRate, 1.0);
      expect(a.frameCount, 10);

      a.reset();
      expect(a.hasRecentData, isFalse);
      expect(a.recentQualityScore, 0);
      expect(a.recentGrade, 'N/A');
      expect(a.isPlacementPoor, isFalse);
    });
  });

  test('report and reset', () {
    final a = TrackingQualityAnalyzer();
    a.observe(_frame(0, ballConf: 0.9, players: 2));
    final report = a.report();
    expect(report, contains('Tracking quality'));
    expect(report, contains('both players visible in 100% of frames'));

    a.reset();
    expect(a.hasData, isFalse);
    expect(a.frameCount, 0);
    expect(a.report(), contains('no frames analysed'));
  });
}
