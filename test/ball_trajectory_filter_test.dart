import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_trajectory_filter.dart';

void main() {
  group('BallTrajectoryFilter', () {
    test('has no estimate before any detection', () {
      final f = BallTrajectoryFilter();
      expect(f.hasEstimate, isFalse);
      expect(f.position, isNull);
      expect(f.velocity, isNull);
      expect(f.estimateAt(100), isNull);
      expect(f.lastTimestampMs, isNull);
    });

    test('first detection seeds the position with zero velocity', () {
      final f = BallTrajectoryFilter();
      f.observe(0, 0.3, 0.5);
      expect(f.hasEstimate, isTrue);
      expect(f.position!.x, closeTo(0.3, 1e-9));
      expect(f.position!.y, closeTo(0.5, 1e-9));
      // A single detection tells us nothing about velocity yet.
      expect(f.velocity!.vx, closeTo(0, 1e-9));
      expect(f.velocity!.vy, closeTo(0, 1e-9));
      expect(f.lastTimestampMs, 0);
    });

    test('locks onto a constant-velocity track and extrapolates forward', () {
      final f = BallTrajectoryFilter();
      // Ball moving +0.03 x and +0.02 y every 33 ms (~0.9 / 0.6 units/sec).
      var x = 0.1;
      var y = 0.2;
      for (var t = 0; t <= 33 * 8; t += 33) {
        f.observe(t, x, y);
        x += 0.03;
        y += 0.02;
      }
      // Velocity should have converged near the true per-second rate.
      expect(f.velocity!.vx, closeTo(0.03 / 0.033, 0.15));
      expect(f.velocity!.vy, closeTo(0.02 / 0.033, 0.15));

      // Extrapolating one frame past the last observation lands near where the
      // next real detection would be.
      const lastT = 33 * 8;
      final est = f.estimateAt(lastT + 33)!;
      expect(est.x, closeTo(x, 0.02));
      expect(est.y, closeTo(y, 0.02));
    });

    test('estimateAt for a past/equal timestamp returns the smoothed position',
        () {
      final f = BallTrajectoryFilter();
      f.observe(0, 0.4, 0.4);
      f.observe(33, 0.45, 0.42);
      final now = f.position!;
      final past = f.estimateAt(0)!; // does not extrapolate backwards
      expect(past.x, closeTo(now.x, 1e-9));
      expect(past.y, closeTo(now.y, 1e-9));
    });

    test('smooths a noisy measurement toward the predicted track', () {
      final f = BallTrajectoryFilter();
      // Establish a clean rightward track.
      for (var i = 0; i < 5; i++) {
        f.observe(i * 33, 0.2 + i * 0.03, 0.5);
      }
      final predicted = f.estimateAt(5 * 33)!.x;
      // Feed an outlier jump; the filtered position should sit between the
      // prediction and the noisy measurement, not snap to the outlier.
      f.observe(5 * 33, 0.9, 0.5);
      final filtered = f.position!.x;
      expect(filtered, greaterThan(predicted));
      expect(filtered, lessThan(0.9));
    });

    test('ignores out-of-order or duplicate timestamps', () {
      final f = BallTrajectoryFilter();
      f.observe(100, 0.5, 0.5);
      f.observe(200, 0.6, 0.5);
      final before = f.position!;
      f.observe(200, 0.9, 0.9); // duplicate ts
      f.observe(150, 0.1, 0.1); // out of order
      expect(f.position!.x, closeTo(before.x, 1e-9));
      expect(f.position!.y, closeTo(before.y, 1e-9));
      expect(f.lastTimestampMs, 200);
    });

    test('reset clears all state', () {
      final f = BallTrajectoryFilter();
      f.observe(0, 0.5, 0.5);
      f.observe(33, 0.55, 0.5);
      f.reset();
      expect(f.hasEstimate, isFalse);
      expect(f.position, isNull);
      expect(f.estimateAt(66), isNull);
      expect(f.lastTimestampMs, isNull);
    });
  });
}
