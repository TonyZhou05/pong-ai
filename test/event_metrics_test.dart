import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/benchmark/event_metrics.dart';
import 'package:pong_ai/core/benchmark/openttgames_converter.dart';
import 'package:pong_ai/core/vision/detection.dart';

FrameResult _frame(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 1, box: BBox(x, y, 0, 0)),
    );

FrameResult _empty(int t) => FrameResult(timestampMs: t);

/// A downward-then-upward arc at x whose apex (bounce) lands on the middle
/// frame's timestamp. Reported bounce time == the apex sample timestamp.
List<FrameResult> _bounceArc(int t0, double x, {int step = 33}) => [
      _frame(t0, x, 0.40),
      _frame(t0 + step, x, 0.60), // descending apex
      _frame(t0 + 2 * step, x, 0.55), // ascending -> bounce at t0+step
    ];

void main() {
  group('EventDetectionBenchmark bounce matching', () {
    test('a clean bounce arc is a true positive with ~0 temporal error', () {
      const bench = EventDetectionBenchmark();
      final result = bench.evaluate(
        name: 'clean',
        frames: _bounceArc(0, 0.30),
        groundTruth: const [GroundTruthEvent(33, TrackedEventType.bounce)],
      );
      expect(result.bounce.truePositives, 1);
      expect(result.bounce.falsePositives, 0);
      expect(result.bounce.falseNegatives, 0);
      expect(result.bounce.precision, 1.0);
      expect(result.bounce.recall, 1.0);
      expect(result.bounce.meanTemporalErrorMs, 0.0);
    });

    test('matches within tolerance and records the timing error', () {
      const bench = EventDetectionBenchmark(
        config: EventBenchmarkConfig(toleranceMs: 100),
      );
      final result = bench.evaluate(
        name: 'jitter',
        frames: _bounceArc(0, 0.30),
        // Ground truth labeled 40 ms off the tracker's apex (33) -> still in window.
        groundTruth: const [GroundTruthEvent(73, TrackedEventType.bounce)],
      );
      expect(result.bounce.truePositives, 1);
      expect(result.bounce.meanTemporalErrorMs, 40.0);
    });

    test('a ground-truth bounce with no emission is a false negative', () {
      const bench = EventDetectionBenchmark();
      final result = bench.evaluate(
        name: 'miss',
        frames: [_frame(0, 0.30, 0.50), _frame(33, 0.31, 0.50)], // flat, no bounce
        groundTruth: const [GroundTruthEvent(33, TrackedEventType.bounce)],
      );
      expect(result.bounce.truePositives, 0);
      expect(result.bounce.falseNegatives, 1);
      expect(result.bounce.recall, 0.0);
      // Vacuous precision when nothing was emitted.
      expect(result.bounce.precision, 1.0);
    });

    test('an emitted bounce with no ground truth is a false positive', () {
      const bench = EventDetectionBenchmark();
      final result = bench.evaluate(
        name: 'spurious',
        frames: _bounceArc(0, 0.30),
        groundTruth: const [],
      );
      expect(result.bounce.truePositives, 0);
      expect(result.bounce.falsePositives, 1);
      expect(result.bounce.precision, 0.0);
      // Vacuous recall when there were no ground-truth bounces.
      expect(result.bounce.recall, 1.0);
    });

    test('a bounce just outside the tolerance is fp + fn, not a match', () {
      const bench = EventDetectionBenchmark(
        config: EventBenchmarkConfig(toleranceMs: 50),
      );
      final result = bench.evaluate(
        name: 'toolate',
        frames: _bounceArc(0, 0.30), // apex/bounce at t=33
        groundTruth: const [GroundTruthEvent(200, TrackedEventType.bounce)],
      );
      expect(result.bounce.truePositives, 0);
      expect(result.bounce.falsePositives, 1);
      expect(result.bounce.falseNegatives, 1);
      expect(result.bounce.f1, 0.0);
    });

    test('greedy matching pairs each emission to its own nearest truth', () {
      const bench = EventDetectionBenchmark();
      final frames = <FrameResult>[
        ..._bounceArc(0, 0.30), // bounce at t=33
        _empty(120),
        _empty(150),
        _empty(180),
        _empty(210),
        _empty(240),
        _empty(270), // > maxGapFrames -> BallLost, resets trajectory
        ..._bounceArc(300, 0.30), // bounce at t=333
      ];
      final result = bench.evaluate(
        name: 'twobounces',
        frames: frames,
        groundTruth: const [
          GroundTruthEvent(33, TrackedEventType.bounce),
          GroundTruthEvent(333, TrackedEventType.bounce),
        ],
      );
      expect(result.bounce.truePositives, 2);
      expect(result.bounce.falsePositives, 0);
      expect(result.bounce.falseNegatives, 0);
    });
  });

  group('EventDetectionBenchmark net-cross matching', () {
    test('a left->right crossing is matched as a net-cross event', () {
      const bench = EventDetectionBenchmark();
      final result = bench.evaluate(
        name: 'cross',
        frames: [
          _frame(0, 0.30, 0.50),
          _frame(33, 0.45, 0.50),
          _frame(66, 0.60, 0.50), // crosses net (0.5) at t=66
        ],
        groundTruth: const [GroundTruthEvent(66, TrackedEventType.netCross)],
      );
      expect(result.netCross.truePositives, 1);
      expect(result.netCross.recall, 1.0);
      // A net-cross ground truth must not be satisfied by a bounce and vice versa.
      expect(result.bounce.groundTruthCount, 0);
    });
  });

  test('report() names both event types with P/R/F1', () {
    const bench = EventDetectionBenchmark();
    final result = bench.evaluate(
      name: 'clip1',
      frames: _bounceArc(0, 0.30),
      groundTruth: const [GroundTruthEvent(33, TrackedEventType.bounce)],
    );
    final text = result.report();
    expect(text, contains('Events: clip1'));
    expect(text, contains('Bounce'));
    expect(text, contains('NetCross'));
    expect(text, contains('P/R/F1'));
  });

  group('openTtGamesBounceEvents converter', () {
    test('extracts only bounce frames, fps-scaled and sorted', () {
      final events = openTtGamesBounceEvents(
        eventsMarkup: {
          '60': 'bounce',
          '30': 'net', // ball hits net -> not a tracker event, skipped
          '90': 'empty', // non-event marker, skipped
          '15': 'bounce',
          'meta': 'bounce', // non-integer key, skipped
        },
        fps: 30,
      );
      expect(events.length, 2);
      // Sorted by time: frame 15 -> 500 ms, frame 60 -> 2000 ms.
      expect(events[0].timestampMs, 500);
      expect(events[1].timestampMs, 2000);
      expect(events.every((e) => e.type == TrackedEventType.bounce), isTrue);
    });

    test('converted bounces score against the pipeline that produced them', () {
      // Ground-truth bounce labeled at frame 1 (33 ms at 30 fps) — the apex of
      // the arc — so a perfect tracker run recalls it.
      final gt = openTtGamesBounceEvents(
        eventsMarkup: {'1': 'bounce'},
        fps: 30,
      );
      expect(gt.single.timestampMs, 33);
      const bench = EventDetectionBenchmark();
      final result = bench.evaluate(
        name: 'ott',
        frames: _bounceArc(0, 0.30),
        groundTruth: gt,
      );
      expect(result.bounce.truePositives, 1);
      expect(result.bounce.recall, 1.0);
    });
  });
}
