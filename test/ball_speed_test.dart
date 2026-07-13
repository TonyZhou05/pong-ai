import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_speed.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/vision/detection.dart';

FrameResult _ball(int t, double x) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, 0.5, 0, 0)),
      people: const [],
    );

FrameResult _noBall(int t) =>
    FrameResult(timestampMs: t, ball: null, people: const []);

void main() {
  test('no data before two detections', () {
    final e = BallSpeedEstimator();
    expect(e.hasData, isFalse);
    expect(e.maxKmh, 0);
    expect(e.averageKmh, 0);

    e.observe(_ball(0, 0.2));
    // A single detection has no displacement to measure yet.
    expect(e.hasData, isFalse);
  });

  test('estimates km/h from along-table displacement (full-frame ruler)', () {
    // Full-frame geometry -> 1 x-unit == 2.74 m. dx=0.1 over 33 ms:
    // 0.274 m / 0.033 s * 3.6 == ~29.9 km/h.
    final e = BallSpeedEstimator();
    e.observe(_ball(0, 0.2));
    e.observe(_ball(33, 0.3));

    expect(e.hasData, isTrue);
    expect(e.sampleCount, 1);
    const expected = 0.1 * 2.74 / (33 / 1000.0) * 3.6;
    expect(e.maxKmh, closeTo(expected, 1e-6));
    expect(e.averageKmh, closeTo(expected, 1e-6));
  });

  test('tracks the max and averages across readings; direction-agnostic', () {
    final e = BallSpeedEstimator();
    e.observe(_ball(0, 0.20));
    e.observe(_ball(33, 0.30)); // +0.10 -> ~29.9
    e.observe(_ball(66, 0.10)); // -0.20 (moving back) -> ~59.8
    e.observe(_ball(99, 0.15)); // +0.05 -> ~14.9

    expect(e.sampleCount, 3);
    const slow = 0.05 * 2.74 / (33 / 1000.0) * 3.6;
    const fast = 0.20 * 2.74 / (33 / 1000.0) * 3.6;
    expect(e.maxKmh, closeTo(fast, 1e-6));
    expect(e.averageKmh, greaterThan(slow));
    expect(e.averageKmh, lessThan(fast));
  });

  test('calibrated narrower table span scales the ruler up', () {
    // The table occupies only the middle 50% of the frame, so each x-unit maps
    // to a smaller real distance -> the same frame displacement is slower.
    const geometry = TableGeometry(netX: 0.5, left: 0.25, right: 0.75);
    final e = BallSpeedEstimator(geometry: geometry);
    e.observe(_ball(0, 0.30));
    e.observe(_ball(33, 0.40));

    const expected = 0.1 * (2.74 / 0.5) * 3.6 / (33 / 1000.0);
    // metersPerUnitX = 2.74 / 0.5 = 5.48.
    expect(e.metersPerUnitX, closeTo(5.48, 1e-9));
    expect(e.maxKmh, closeTo(expected, 1e-6));
  });

  test('does not measure across a ball-loss gap', () {
    final e = BallSpeedEstimator(maxGapMs: 100);
    e.observe(_ball(0, 0.20));
    for (var t = 33; t <= 200; t += 33) {
      e.observe(_noBall(t));
    }
    // Ball reappears far away after >100 ms of loss: no bogus reading.
    e.observe(_ball(231, 0.80));
    expect(e.hasData, isFalse);

    // The next in-window frame measures normally.
    e.observe(_ball(264, 0.82));
    expect(e.sampleCount, 1);
  });

  test('ignores out-of-order / duplicate timestamps', () {
    final e = BallSpeedEstimator();
    e.observe(_ball(100, 0.20));
    e.observe(_ball(100, 0.40)); // duplicate ts -> no reading
    e.observe(_ball(50, 0.60)); // earlier ts -> no reading
    expect(e.hasData, isFalse);
  });

  test('rejects physically-impossible teleports as spurious', () {
    final e = BallSpeedEstimator(maxPlausibleKmh: 250);
    e.observe(_ball(0, 0.05));
    // Full-frame jump in one 33 ms frame => ~0.95*2.74/0.033*3.6 ~ 284 km/h.
    e.observe(_ball(33, 1.0));
    expect(e.hasData, isFalse);
  });

  test('lastKmh reports the most recent reading for a live readout', () {
    final e = BallSpeedEstimator();
    expect(e.lastKmh, isNull);

    e.observe(_ball(0, 0.20));
    // Still no displacement measured, so no live reading yet.
    expect(e.lastKmh, isNull);

    e.observe(_ball(33, 0.30)); // +0.10 -> ~29.9
    const slow = 0.10 * 2.74 / (33 / 1000.0) * 3.6;
    expect(e.lastKmh, closeTo(slow, 1e-6));

    e.observe(_ball(66, 0.06)); // -0.24 -> faster
    const fast = 0.24 * 2.74 / (33 / 1000.0) * 3.6;
    // lastKmh follows the newest reading, not the max.
    expect(e.lastKmh, closeTo(fast, 1e-6));
    expect(e.maxKmh, closeTo(fast, 1e-6));

    e.reset();
    expect(e.lastKmh, isNull);
  });

  test('reset clears all state', () {
    final e = BallSpeedEstimator();
    e.observe(_ball(0, 0.2));
    e.observe(_ball(33, 0.4));
    expect(e.hasData, isTrue);
    e.reset();
    expect(e.hasData, isFalse);
    expect(e.maxKmh, 0);
    expect(e.sampleCount, 0);
  });
}
