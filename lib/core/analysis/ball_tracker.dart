/// Pure-Dart analysis of the ball's trajectory across camera frames.
///
/// This layer sits **between** the vision pipeline (which emits per-frame
/// [Detection]s of the ball) and the [ScoringEngine] (which only understands
/// "player X won the point"). It has no Flutter or plugin dependencies, so the
/// rally logic can be unit-tested against synthetic trajectories and replayed
/// benchmark clips without a camera.
///
/// Coordinate convention (matching the vision models): the frame is normalized
/// to `[0,1] x [0,1]` with **y increasing downward**. With the phone placed on
/// the side of the table, the table spans the frame horizontally, the net is a
/// vertical line at [TableGeometry.netX], and a ball bounce on the table shows
/// up as a local *maximum* in y (the ball descends, touches, then rises).
library;

import '../vision/detection.dart';
import 'ball_trajectory_filter.dart';

/// Which half of the table the ball is over, split by the net.
enum TableSide { left, right }

extension TableSideX on TableSide {
  TableSide get other => this == TableSide.left ? TableSide.right : TableSide.left;
}

/// Static description of the table within the normalized frame.
///
/// With the phone placed on the side of the table, the playing surface does not
/// fill the whole frame — it occupies a rectangular band bounded by
/// ([left], [top]) and ([right], [bottom]). Calibrating that band lets the
/// tracker reject "bounces" that actually happen off the table (the ball
/// hitting the floor below, or bouncing on a chair beside the table), which
/// would otherwise be scored as legal rally bounces. The bounds default to the
/// full frame so an uncalibrated tracker behaves exactly as before.
class TableGeometry {
  const TableGeometry({
    this.netX = 0.5,
    this.left = 0.0,
    this.right = 1.0,
    this.top = 0.0,
    this.bottom = 1.0,
  })  : assert(left >= 0 && right <= 1 && left < right,
            'table x-bounds must satisfy 0 <= left < right <= 1',),
        assert(top >= 0 && bottom <= 1 && top < bottom,
            'table y-bounds must satisfy 0 <= top < bottom <= 1',),
        assert(netX > left && netX < right,
            'netX must fall strictly between the table left/right edges',);

  /// Normalized x of the net (the vertical line dividing the two sides).
  final double netX;

  /// Normalized x of the table's left edge in the frame.
  final double left;

  /// Normalized x of the table's right edge in the frame.
  final double right;

  /// Normalized y of the table surface's near/top edge in the frame.
  final double top;

  /// Normalized y of the table surface's far/bottom edge in the frame.
  final double bottom;

  /// The side of the table a given normalized x falls on.
  TableSide sideOf(double x) => x < netX ? TableSide.left : TableSide.right;

  /// Whether ([x], [y]) lies within the calibrated table surface region — used
  /// to accept only bounces that happen on the table, not off it.
  bool containsSurface(double x, double y) =>
      x >= left && x <= right && y >= top && y <= bottom;
}

/// One accepted position of the ball at a moment in time.
class BallSample {
  const BallSample(this.timestampMs, this.x, this.y);

  final int timestampMs;
  final double x;
  final double y;
}

/// Something notable the tracker inferred from the trajectory.
sealed class TrackerEvent {
  const TrackerEvent(this.timestampMs);
  final int timestampMs;
}

/// The ball touched the table surface (a local maximum in y).
class BounceEvent extends TrackerEvent {
  const BounceEvent(super.timestampMs, this.x, this.y, this.side);

  final double x;
  final double y;

  /// Which side of the table the bounce happened on.
  final TableSide side;

  @override
  String toString() => 'Bounce(@$timestampMs, side=$side, x=${x.toStringAsFixed(2)})';
}

/// The ball crossed the net from one side to the other.
class NetCrossEvent extends TrackerEvent {
  const NetCrossEvent(super.timestampMs, this.from, this.to);

  final TableSide from;
  final TableSide to;

  @override
  String toString() => 'NetCross(@$timestampMs, $from->$to)';
}

/// The ball detection was lost for longer than the allowed gap; the current
/// trajectory (velocity estimate) has been dropped.
class BallLostEvent extends TrackerEvent {
  const BallLostEvent(super.timestampMs);

  @override
  String toString() => 'BallLost(@$timestampMs)';
}

/// Incrementally consumes ball detections and emits [TrackerEvent]s.
///
/// The tracker is intentionally stateful and single-pass so it can run live on
/// the camera stream. Feed it one [FrameResult] per frame via [update]; it
/// returns any events inferred from *that* frame.
///
/// Robustness choices:
/// * Missing detections (the model drops the ball for a frame) are tolerated up
///   to [maxGapFrames]; beyond that the velocity estimate is reset and a
///   [BallLostEvent] is emitted so downstream logic can end the rally.
/// * A bounce requires the vertical velocity to flip from clearly *descending*
///   to clearly *ascending*, each above [minBounceSpeed], which rejects the
///   jitter of a near-stationary or noisily-detected ball.
class BallTracker {
  BallTracker({
    this.geometry = const TableGeometry(),
    this.minBounceSpeed = 0.004,
    this.maxGapFrames = 6,
    BallTrajectoryFilter? filter,
  })  : assert(minBounceSpeed >= 0),
        assert(maxGapFrames >= 0),
        _filter = filter ?? BallTrajectoryFilter();

  final TableGeometry geometry;

  /// Minimum |Δy| per frame for a direction change to count as a real bounce.
  final double minBounceSpeed;

  /// How many consecutive frames without a detection are tolerated before the
  /// trajectory is considered broken.
  final int maxGapFrames;

  /// Constant-velocity Kalman smoother/predictor kept in lock-step with the
  /// accepted samples so we can estimate the ball's position through detector
  /// dropouts (see [estimateBallAt]). It runs alongside — never replaces — the
  /// raw-detection event logic, so scoring stays driven by real detections.
  final BallTrajectoryFilter _filter;

  BallSample? _prev;

  /// Vertical velocity (Δy) between the two most recent accepted samples.
  double? _lastVy;

  int _missedFrames = 0;

  /// The most recent accepted ball sample, or null before the first detection.
  BallSample? get lastSample => _prev;

  /// The side the ball was last seen on, or null before the first detection.
  TableSide? get currentSide =>
      _prev == null ? null : geometry.sideOf(_prev!.x);

  /// Whether the Kalman filter holds a usable trajectory estimate.
  bool get hasEstimate => _filter.hasEstimate;

  /// Current estimated ball velocity in normalized units/second, or null before
  /// enough detections have been seen.
  ({double vx, double vy})? get estimatedVelocity => _filter.velocity;

  /// Best estimate of the ball's normalized position at [timestampMs], using
  /// the constant-velocity Kalman model. Returns the smoothed detection when the
  /// ball is visible and an *extrapolated* position when it is not — so the live
  /// overlay can keep drawing the ball through motion-blur dropouts (within
  /// [maxGapFrames], after which the trajectory is dropped and this returns
  /// null). Returns null before the first detection.
  ({double x, double y})? estimateBallAt(int timestampMs) =>
      _filter.estimateAt(timestampMs);

  /// Feed one frame; returns the events inferred from it (possibly empty).
  List<TrackerEvent> update(FrameResult frame) {
    final ball = frame.ball;
    if (ball == null) {
      return _handleMissingBall(frame.timestampMs);
    }

    _missedFrames = 0;
    final sample = BallSample(
      frame.timestampMs,
      ball.box.centerX,
      ball.box.centerY,
    );
    _filter.observe(sample.timestampMs, sample.x, sample.y);

    final prev = _prev;
    if (prev == null) {
      _prev = sample;
      return const [];
    }

    // Guard against a zero/negative timestamp step (replayed or duplicate
    // frames) which would make velocity meaningless.
    if (sample.timestampMs <= prev.timestampMs) {
      _prev = sample;
      return const [];
    }

    final events = <TrackerEvent>[];
    final vy = sample.y - prev.y;

    final netEvent = _detectNetCross(prev, sample);
    if (netEvent != null) events.add(netEvent);

    final bounce = _detectBounce(prev, vy);
    if (bounce != null) events.add(bounce);

    _lastVy = vy;
    _prev = sample;
    return events;
  }

  List<TrackerEvent> _handleMissingBall(int timestampMs) {
    if (_prev == null) return const [];
    _missedFrames++;
    if (_missedFrames > maxGapFrames) {
      _prev = null;
      _lastVy = null;
      _missedFrames = 0;
      _filter.reset();
      return [BallLostEvent(timestampMs)];
    }
    return const [];
  }

  NetCrossEvent? _detectNetCross(BallSample prev, BallSample now) {
    final from = geometry.sideOf(prev.x);
    final to = geometry.sideOf(now.x);
    if (from == to) return null;
    return NetCrossEvent(now.timestampMs, from, to);
  }

  /// A bounce is the apex of a downward-then-upward arc: the *previous* sample
  /// was the lowest point, so it is reported as the bounce location/time.
  ///
  /// A direction change that happens outside the calibrated table surface (e.g.
  /// the ball hitting the floor below the table) is not a legal rally bounce, so
  /// it is dropped rather than emitted.
  BounceEvent? _detectBounce(BallSample apex, double vyNow) {
    final vyPrev = _lastVy;
    if (vyPrev == null) return null;
    final descending = vyPrev > minBounceSpeed;
    final ascending = vyNow < -minBounceSpeed;
    if (descending && ascending && geometry.containsSurface(apex.x, apex.y)) {
      return BounceEvent(
        apex.timestampMs,
        apex.x,
        apex.y,
        geometry.sideOf(apex.x),
      );
    }
    return null;
  }

  /// Forget all trajectory state (e.g. between rallies).
  void reset() {
    _prev = null;
    _lastVy = null;
    _missedFrames = 0;
    _filter.reset();
  }
}
