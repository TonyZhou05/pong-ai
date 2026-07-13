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
