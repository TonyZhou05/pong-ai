import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/vision/detection.dart';

/// Build a single-frame result whose only content is the ball at ([x], [y]).
FrameResult _frame(int t, double x, double y) {
  // A zero-size box centered exactly on (x, y).
  return FrameResult(
    timestampMs: t,
    ball: Detection(
      label: 'ball',
      confidence: 0.9,
      box: BBox(x, y, 0, 0),
    ),
  );
}

/// A frame with no ball detection.
FrameResult _empty(int t) => FrameResult(timestampMs: t);

/// Feed a whole trajectory and collect every event, in order.
List<TrackerEvent> _run(BallTracker tracker, List<FrameResult> frames) {
  final all = <TrackerEvent>[];
  for (final f in frames) {
    all.addAll(tracker.update(f));
  }
  return all;
}

void main() {
  group('TableGeometry', () {
    test('splits the frame at the net line', () {
      const g = TableGeometry(netX: 0.5);
      expect(g.sideOf(0.2), TableSide.left);
      expect(g.sideOf(0.8), TableSide.right);
      // Exactly on/after the net counts as the right side.
      expect(g.sideOf(0.5), TableSide.right);
    });
  });

  group('BallTracker — net crossing', () {
    test('emits a single crossing when the ball passes the net', () {
      final tracker = BallTracker();
      // Constant y so no bounce fires; x sweeps left -> right across net=0.5.
      final events = _run(tracker, [
        _frame(0, 0.30, 0.5),
        _frame(33, 0.45, 0.5),
        _frame(66, 0.60, 0.5), // crosses here
        _frame(99, 0.75, 0.5),
      ]);
      final crosses = events.whereType<NetCrossEvent>().toList();
      expect(crosses, hasLength(1));
      expect(crosses.single.from, TableSide.left);
      expect(crosses.single.to, TableSide.right);
      expect(crosses.single.timestampMs, 66);
    });

    test('emits a crossing for each back-and-forth over the net', () {
      final tracker = BallTracker();
      final events = _run(tracker, [
        _frame(0, 0.30, 0.5),
        _frame(33, 0.70, 0.5), // L -> R
        _frame(66, 0.30, 0.5), // R -> L
        _frame(99, 0.70, 0.5), // L -> R
      ]);
      expect(events.whereType<NetCrossEvent>(), hasLength(3));
    });
  });

  group('BallTracker — bounce detection', () {
    test('reports a bounce at the apex of a down-then-up arc', () {
      final tracker = BallTracker();
      // x fixed on the left side; y descends (increases) then ascends.
      final events = _run(tracker, [
        _frame(0, 0.30, 0.40),
        _frame(33, 0.30, 0.60), // descending, vy = +0.20
        _frame(66, 0.30, 0.55), // ascending, vy = -0.05 -> bounce at t=33
      ]);
      final bounces = events.whereType<BounceEvent>().toList();
      expect(bounces, hasLength(1));
      expect(bounces.single.timestampMs, 33);
      expect(bounces.single.y, closeTo(0.60, 1e-9));
      expect(bounces.single.side, TableSide.left);
    });

    test('ignores sub-threshold jitter as a bounce', () {
      final tracker = BallTracker(minBounceSpeed: 0.01);
      // Tiny oscillation below the threshold should not register.
      final events = _run(tracker, [
        _frame(0, 0.30, 0.500),
        _frame(33, 0.30, 0.505), // vy = +0.005 (< 0.01)
        _frame(66, 0.30, 0.500), // vy = -0.005
      ]);
      expect(events.whereType<BounceEvent>(), isEmpty);
    });

    test('a bounce on the right side is attributed to the right side', () {
      final tracker = BallTracker();
      final events = _run(tracker, [
        _frame(0, 0.80, 0.40),
        _frame(33, 0.80, 0.60),
        _frame(66, 0.80, 0.55),
      ]);
      expect(events.whereType<BounceEvent>().single.side, TableSide.right);
    });
  });

  group('BallTracker — missed detections', () {
    test('tolerates short gaps without losing the trajectory', () {
      final tracker = BallTracker(maxGapFrames: 3);
      final events = _run(tracker, [
        _frame(0, 0.30, 0.5),
        _empty(33),
        _empty(66),
        _frame(99, 0.70, 0.5), // still tracks -> crossing survives the gap
      ]);
      expect(events.whereType<BallLostEvent>(), isEmpty);
      expect(events.whereType<NetCrossEvent>(), hasLength(1));
    });

    test('emits BallLost after exceeding the allowed gap', () {
      final tracker = BallTracker(maxGapFrames: 2);
      final events = _run(tracker, [
        _frame(0, 0.30, 0.5),
        _empty(33),
        _empty(66),
        _empty(99), // 3rd miss > maxGapFrames=2
      ]);
      final lost = events.whereType<BallLostEvent>().toList();
      expect(lost, hasLength(1));
      expect(lost.single.timestampMs, 99);
      expect(tracker.lastSample, isNull);
    });

    test('no events before the first detection ever arrives', () {
      final tracker = BallTracker();
      final events = _run(tracker, [_empty(0), _empty(33)]);
      expect(events, isEmpty);
      expect(tracker.currentSide, isNull);
    });
  });

  group('BallTracker — state helpers', () {
    test('currentSide reflects the latest accepted sample', () {
      final tracker = BallTracker();
      tracker.update(_frame(0, 0.20, 0.5));
      expect(tracker.currentSide, TableSide.left);
      tracker.update(_frame(33, 0.90, 0.5));
      expect(tracker.currentSide, TableSide.right);
    });

    test('reset clears trajectory state', () {
      final tracker = BallTracker();
      tracker.update(_frame(0, 0.20, 0.5));
      tracker.reset();
      expect(tracker.lastSample, isNull);
      expect(tracker.currentSide, isNull);
      // After reset the next detection is treated as a fresh start (no cross).
      final events = tracker.update(_frame(33, 0.90, 0.5));
      expect(events, isEmpty);
    });

    test('ignores non-increasing timestamps', () {
      final tracker = BallTracker();
      tracker.update(_frame(100, 0.30, 0.5));
      // Duplicate/out-of-order frame: should be absorbed, not crash or cross.
      final events = tracker.update(_frame(100, 0.70, 0.5));
      expect(events, isEmpty);
    });
  });
}
