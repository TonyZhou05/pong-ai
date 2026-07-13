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

    test('containsSurface bounds the calibrated table region', () {
      const g = TableGeometry(
        netX: 0.5,
        left: 0.1,
        right: 0.9,
        top: 0.3,
        bottom: 0.7,
      );
      // Inside the band.
      expect(g.containsSurface(0.5, 0.5), isTrue);
      // On the edges (inclusive).
      expect(g.containsSurface(0.1, 0.3), isTrue);
      expect(g.containsSurface(0.9, 0.7), isTrue);
      // Off the ends / below the surface (e.g. a floor bounce at y=0.85).
      expect(g.containsSurface(0.05, 0.5), isFalse);
      expect(g.containsSurface(0.5, 0.85), isFalse);
    });

    test('defaults to the whole frame as the surface', () {
      const g = TableGeometry();
      expect(g.containsSurface(0.0, 0.0), isTrue);
      expect(g.containsSurface(1.0, 1.0), isTrue);
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

    test('drops a direction change that happens off the table surface', () {
      // Table surface only spans y in [0.2, 0.6]; the apex at y=0.85 is below
      // it (a floor bounce), so no BounceEvent should fire.
      final tracker = BallTracker(
        geometry: const TableGeometry(top: 0.2, bottom: 0.6),
      );
      final events = _run(tracker, [
        _frame(0, 0.30, 0.70),
        _frame(33, 0.30, 0.85), // descending apex below the surface
        _frame(66, 0.30, 0.80), // ascending
      ]);
      expect(events.whereType<BounceEvent>(), isEmpty);
    });

    test('keeps a bounce that lands on the calibrated surface', () {
      final tracker = BallTracker(
        geometry: const TableGeometry(top: 0.2, bottom: 0.6),
      );
      final events = _run(tracker, [
        _frame(0, 0.30, 0.40),
        _frame(33, 0.30, 0.55), // apex within [0.2, 0.6]
        _frame(66, 0.30, 0.50),
      ]);
      expect(events.whereType<BounceEvent>(), hasLength(1));
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

  group('BallTracker — Kalman prediction through gaps', () {
    test('no estimate before the first detection', () {
      final tracker = BallTracker();
      expect(tracker.hasEstimate, isFalse);
      expect(tracker.estimateBallAt(0), isNull);
      expect(tracker.estimatedVelocity, isNull);
    });

    test('extrapolates the ball position across a detector dropout', () {
      final tracker = BallTracker(maxGapFrames: 5);
      // Steady rightward track at +0.05 x per 33 ms frame.
      _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.25, 0.5),
        _frame(66, 0.30, 0.5),
        _frame(99, 0.35, 0.5),
      ]);
      expect(tracker.hasEstimate, isTrue);
      expect(tracker.estimatedVelocity!.vx, greaterThan(0));

      // Ball is lost this frame; the estimate should extrapolate forward,
      // landing past the last seen x rather than freezing on it.
      tracker.update(_empty(132));
      final est = tracker.estimateBallAt(132)!;
      expect(est.x, greaterThan(0.35));
      expect(est.x, closeTo(0.40, 0.03));
      expect(est.y, closeTo(0.5, 0.02));
    });

    test('drops the estimate once the gap exceeds the tolerance', () {
      final tracker = BallTracker(maxGapFrames: 2);
      _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.25, 0.5),
        _empty(66),
        _empty(99),
        _empty(132), // exceeds maxGapFrames -> BallLost, trajectory dropped
      ]);
      expect(tracker.hasEstimate, isFalse);
      expect(tracker.estimateBallAt(132), isNull);
    });

    test('reset clears the trajectory estimate', () {
      final tracker = BallTracker();
      tracker.update(_frame(0, 0.4, 0.5));
      tracker.update(_frame(33, 0.45, 0.5));
      tracker.reset();
      expect(tracker.hasEstimate, isFalse);
      expect(tracker.estimateBallAt(66), isNull);
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

  group('BallTracker — outlier gate (maxJump)', () {
    test('is disabled by default: an implausible jump is still accepted', () {
      final tracker = BallTracker();
      _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.25, 0.5),
        _frame(66, 0.95, 0.5), // huge jump, but no gate
      ]);
      expect(tracker.outlierCount, 0);
      expect(tracker.lastSample!.x, 0.95);
    });

    test('rejects a detection that jumps implausibly far from prediction', () {
      final tracker = BallTracker(maxJump: 0.4);
      // Establish a steady rightward track (prediction ~0.30 next).
      _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.25, 0.5),
      ]);
      // A spurious detection across the frame: >0.4 from the ~0.30 prediction.
      final events = tracker.update(_frame(66, 0.95, 0.5));
      expect(events, isEmpty);
      expect(tracker.outlierCount, 1);
      // The trajectory is NOT teleported — last accepted sample is unchanged.
      expect(tracker.lastSample!.x, 0.25);
    });

    test('a rejected outlier does not manufacture a net-cross', () {
      // Without the gate the 0.25 -> 0.95 jump would register a L->R crossing.
      final tracker = BallTracker(maxJump: 0.4);
      final events = _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.25, 0.5),
        _frame(66, 0.95, 0.5), // spurious, would-be crossing — rejected
      ]);
      expect(events.whereType<NetCrossEvent>(), isEmpty);
      expect(tracker.currentSide, TableSide.left);
    });

    test('still accepts real detections that stay near the prediction', () {
      final tracker = BallTracker(maxJump: 0.4);
      final events = _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.25, 0.5),
        _frame(66, 0.30, 0.5),
        _frame(99, 0.35, 0.5),
      ]);
      expect(tracker.outlierCount, 0);
      expect(tracker.lastSample!.x, 0.35);
      // A genuine steady track still crosses if it reaches the far side.
      expect(events.whereType<NetCrossEvent>(), isEmpty);
    });

    test('never rejects the first samples that seed the trajectory', () {
      // Only one prior sample -> no established velocity -> gate must not fire.
      final tracker = BallTracker(maxJump: 0.1);
      _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.90, 0.5), // second sample: seeds velocity, not gated
      ]);
      expect(tracker.outlierCount, 0);
      expect(tracker.lastSample!.x, 0.90);
    });

    test('persistent outliers end the rally via BallLost after the gap', () {
      final tracker = BallTracker(maxJump: 0.4, maxGapFrames: 2);
      final events = _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.25, 0.5),
        _frame(66, 0.95, 0.5), // outlier 1
        _frame(99, 0.96, 0.5), // outlier 2
        _frame(132, 0.97, 0.5), // outlier 3 -> exceeds gap -> BallLost + reset
      ]);
      expect(events.whereType<BallLostEvent>(), hasLength(1));
      // After the ball-lost reset the outlier counter is cleared and the next
      // detection re-seeds a fresh trajectory.
      expect(tracker.outlierCount, 0);
      expect(tracker.hasEstimate, isFalse);
    });

    test('reset clears the outlier counter', () {
      final tracker = BallTracker(maxJump: 0.4);
      _run(tracker, [
        _frame(0, 0.20, 0.5),
        _frame(33, 0.25, 0.5),
        _frame(66, 0.95, 0.5),
      ]);
      expect(tracker.outlierCount, 1);
      tracker.reset();
      expect(tracker.outlierCount, 0);
    });
  });
}
