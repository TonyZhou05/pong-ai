import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/table_calibrator.dart';
import 'package:pong_ai/core/vision/detection.dart';

/// A frame carrying a ball at ([x], [y]) and optional player centroids.
FrameResult _frame(int t, double x, double y, {List<double> players = const []}) {
  return FrameResult(
    timestampMs: t,
    ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, y, 0, 0)),
    people: [
      for (final px in players)
        PersonPose(box: BBox(px, 0.5, 0, 0), keypoints: const []),
    ],
  );
}

void main() {
  group('TableCalibrator', () {
    test('returns null until enough ball samples are seen', () {
      final cal = TableCalibrator(minBallSamples: 5);
      for (var i = 0; i < 4; i++) {
        cal.addBall(0.1 + i * 0.05, 0.4 + i * 0.02);
      }
      expect(cal.ballSampleCount, 4);
      expect(cal.isReady, isFalse);
      expect(cal.calibrate(), isNull);
    });

    test('estimates a surface band from the spread of ball positions', () {
      final cal = TableCalibrator(minBallSamples: 10, trim: 0.0);
      // Ball travels across x in [0.2, 0.8] and y in [0.4, 0.6].
      for (var i = 0; i <= 10; i++) {
        cal.addBall(0.2 + 0.06 * i, 0.4 + 0.02 * (i % 11));
      }
      final g = cal.calibrate();
      expect(g, isNotNull);
      expect(g!.left, closeTo(0.2, 1e-9));
      expect(g.right, closeTo(0.8, 1e-9));
      expect(g.top, closeTo(0.4, 1e-9));
      expect(g.bottom, closeTo(0.6, 1e-9));
    });

    test('trims outlier (off-table) samples out of the bounds', () {
      final cal = TableCalibrator(minBallSamples: 20, trim: 0.05);
      // 20 in-band samples plus one wild floor bounce that min/max would catch.
      for (var i = 0; i < 20; i++) {
        cal.addBall(0.3 + 0.02 * i, 0.45 + 0.005 * i);
      }
      cal.addBall(0.99, 0.99); // outlier
      final g = cal.calibrate()!;
      // The trimmed right/bottom edge stays well below the outlier.
      expect(g.right, lessThan(0.9));
      expect(g.bottom, lessThan(0.9));
    });

    test('places the net between the two players when they are visible', () {
      final cal = TableCalibrator(minBallSamples: 6);
      for (var i = 0; i < 6; i++) {
        // Players consistently at x=0.15 (left end) and x=0.75 (right end).
        cal.observe(_frame(i, 0.3 + 0.05 * i, 0.4 + 0.03 * i, players: [0.15, 0.75]));
      }
      final g = cal.calibrate()!;
      expect(g.netX, closeTo(0.45, 1e-6));
    });

    test('falls back to the ball-travel midpoint when no players are seen', () {
      final cal = TableCalibrator(minBallSamples: 6, trim: 0.0);
      for (var i = 0; i <= 6; i++) {
        cal.addBall(0.2 + 0.1 * i, 0.4 + 0.03 * i); // x in [0.2, 0.8]
      }
      final g = cal.calibrate()!;
      expect(g.netX, closeTo(0.5, 1e-9));
    });

    test('keeps the estimated net strictly inside the table edges', () {
      final cal = TableCalibrator(minBallSamples: 6, trim: 0.0);
      for (var i = 0; i <= 6; i++) {
        cal.addBall(0.4 + 0.02 * i, 0.4 + 0.03 * i); // narrow x band [0.4, 0.52]
      }
      // Both players read on the far left, which would push the net off-table.
      for (var i = 0; i < 3; i++) {
        cal.addPlayer(0.05);
        cal.addPlayer(0.1);
      }
      final g = cal.calibrate()!;
      expect(g.netX, greaterThan(g.left));
      expect(g.netX, lessThan(g.right));
    });

    test('returns null when the ball never really moved (degenerate band)', () {
      final cal = TableCalibrator(minBallSamples: 5);
      for (var i = 0; i < 5; i++) {
        cal.addBall(0.5, 0.5);
      }
      expect(cal.calibrate(), isNull);
    });

    test('the calibrated geometry accepts on-table and rejects off-table', () {
      final cal = TableCalibrator(minBallSamples: 10, trim: 0.0);
      for (var i = 0; i <= 10; i++) {
        cal.addBall(0.2 + 0.06 * i, 0.4 + 0.02 * (i % 11));
      }
      final g = cal.calibrate()!;
      expect(g.containsSurface(0.5, 0.5), isTrue);
      expect(g.containsSurface(0.5, 0.95), isFalse); // floor, below the table
    });

    test('reset clears accumulated samples', () {
      final cal = TableCalibrator(minBallSamples: 2);
      cal.addBall(0.3, 0.5);
      cal.addBall(0.6, 0.5);
      expect(cal.isReady, isTrue);
      cal.reset();
      expect(cal.ballSampleCount, 0);
      expect(cal.isReady, isFalse);
    });
  });
}
