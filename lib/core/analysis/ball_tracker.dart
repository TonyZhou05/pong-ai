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
  const BallLostEvent(super.timestampMs, {this.lostOutside});

  /// When the ball's last tracked position was already past the table's outer
  /// edge, the side it exited over (left of the left edge / right of the
  /// right edge); null when it vanished over the playing area. Lets the
  /// referee distinguish "flew long past the baseline" (out of bounds) from a
  /// mid-air detection dropout.
  final TableSide? lostOutside;

  @override
  String toString() => 'BallLost(@$timestampMs'
      '${lostOutside == null ? '' : ', outside $lostOutside'})';
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
/// * Once a trajectory is established, a detection that lands implausibly far
///   ([maxJump]) from the Kalman-predicted position is rejected as a spurious
///   detection (the detector latching onto a round object / bright logo
///   elsewhere in the frame) rather than teleporting the trajectory — see
///   [update]. Disabled by default (`maxJump == null`).
class BallTracker {
  BallTracker({
    this.geometry = const TableGeometry(),
    this.minBounceSpeed = 0.004,
    this.maxGapFrames = 6,
    this.maxJump,
    this.netBounceExclusion = 0,
    BallTrajectoryFilter? filter,
  })  : assert(minBounceSpeed >= 0),
        assert(maxGapFrames >= 0),
        assert(maxJump == null || maxJump > 0),
        assert(netBounceExclusion >= 0),
        _filter = filter ?? BallTrajectoryFilter();

  final TableGeometry geometry;

  /// Minimum |Δy| per frame for a direction change to count as a real bounce.
  final double minBounceSpeed;

  /// How many consecutive frames without a detection are tolerated before the
  /// trajectory is considered broken.
  final int maxGapFrames;

  /// Maximum normalized distance a detection may sit from the Kalman-predicted
  /// position before it is treated as a spurious detection (a physical-
  /// plausibility gate). `null` disables gating, so raw detections are always
  /// accepted — the default, preserving the pre-gate behaviour. A generous
  /// value (e.g. `0.4`, ~40% of the frame) rejects only gross teleports: the
  /// residual gated here is *after* the constant-velocity prediction, so a true
  /// ball's frame-to-frame residual (measurement noise + gentle bounce reversal)
  /// stays far below it while a detection latching onto something across the
  /// table does not.
  final double? maxJump;

  /// Reject a bounce whose apex lies within this normalized x-distance of the
  /// net line. A downward-then-upward reversal *at* the net plane is usually
  /// the ball clipping the net (or the sampled trajectory kinking as it
  /// crosses), not a table landing — and such a phantom bounce pairs with a
  /// real one into a bogus double-bounce point. `0` (the default) disables the
  /// exclusion, preserving historical behaviour; real bounces do land near the
  /// net (drop shots), so keep the zone tight (e.g. `0.03`).
  final double netBounceExclusion;

  /// Constant-velocity Kalman smoother/predictor kept in lock-step with the
  /// accepted samples so we can estimate the ball's position through detector
  /// dropouts (see [estimateBallAt]). It runs alongside — never replaces — the
  /// raw-detection event logic, so scoring stays driven by real detections.
  final BallTrajectoryFilter _filter;

  BallSample? _prev;

  /// Vertical velocity (Δy) between the two most recent accepted samples.
  double? _lastVy;

  int _missedFrames = 0;

  int _outlierCount = 0;

  /// The most recent accepted ball sample, or null before the first detection.
  BallSample? get lastSample => _prev;

  /// How many detections have been rejected as spurious by the [maxJump] gate
  /// on the current trajectory (cleared on [reset] and when the trajectory is
  /// dropped via a [BallLostEvent]). Useful for observability/tests; always 0
  /// when gating is disabled.
  int get outlierCount => _outlierCount;

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

    // Physical-plausibility gate: once a trajectory is established, a detection
    // that jumps implausibly far from the Kalman prediction is almost certainly
    // a false positive (the detector latching onto something across the frame),
    // so reject it. But the *real* ball may also have been detected this frame
    // at lower confidence (frame.ballCandidates), so before giving up on the
    // frame we try to recover the alternative that lands within the gate. If
    // none does, route this frame through the missing-ball path — the filter
    // extrapolates over it, and persistent spurious detections still end the
    // rally cleanly instead of the trajectory teleporting into a bogus event.
    var accepted = ball;
    if (_isOutlier(frame.timestampMs, ball.box.centerX, ball.box.centerY)) {
      _outlierCount++;
      final recovered = _recoverCandidate(frame);
      if (recovered == null) {
        return _handleMissingBall(frame.timestampMs);
      }
      accepted = recovered;
    }

    _missedFrames = 0;
    final sample = BallSample(
      frame.timestampMs,
      accepted.box.centerX,
      accepted.box.centerY,
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

  /// Whether a detection at ([x], [y], [timestampMs]) is too far from the
  /// Kalman-predicted position to be the ball. Only fires once a velocity is
  /// established (≥2 accepted samples, i.e. [_lastVy] is set) so the very first
  /// samples that seed the trajectory are never rejected.
  bool _isOutlier(int timestampMs, double x, double y) {
    final gate = maxJump;
    if (gate == null || _lastVy == null) return false;
    final predicted = _filter.estimateAt(timestampMs);
    if (predicted == null) return false;
    final dx = x - predicted.x;
    final dy = y - predicted.y;
    return dx * dx + dy * dy > gate * gate;
  }

  /// When the primary ball is rejected by the [maxJump] gate, pick the
  /// alternative candidate (if any) closest to the Kalman prediction that itself
  /// falls within the gate — the real ball the detector reported at lower
  /// confidence than a spurious round object. Returns null when no candidate is
  /// plausible (so the frame is treated as a dropout). Only meaningful when
  /// gating is active; [frame.ballCandidates] is empty on the synthetic path.
  Detection? _recoverCandidate(FrameResult frame) {
    final gate = maxJump;
    if (gate == null || frame.ballCandidates.isEmpty) return null;
    final predicted = _filter.estimateAt(frame.timestampMs);
    if (predicted == null) return null;
    Detection? best;
    var bestDist2 = gate * gate;
    for (final c in frame.ballCandidates) {
      final dx = c.box.centerX - predicted.x;
      final dy = c.box.centerY - predicted.y;
      final dist2 = dx * dx + dy * dy;
      if (dist2 <= bestDist2) {
        bestDist2 = dist2;
        best = c;
      }
    }
    return best;
  }

  List<TrackerEvent> _handleMissingBall(int timestampMs) {
    if (_prev == null) return const [];
    _missedFrames++;
    if (_missedFrames > maxGapFrames) {
      return [_loseBall(timestampMs)];
    }
    return const [];
  }

  /// Drop the current trajectory and report the loss, noting whether the last
  /// tracked position had already left the table's x-extent.
  BallLostEvent _loseBall(int timestampMs) {
    final last = _prev!;
    TableSide? lostOutside;
    if (last.x < geometry.left) {
      lostOutside = TableSide.left;
    } else if (last.x > geometry.right) {
      lostOutside = TableSide.right;
    }
    _prev = null;
    _lastVy = null;
    _missedFrames = 0;
    _outlierCount = 0;
    _filter.reset();
    return BallLostEvent(timestampMs, lostOutside: lostOutside);
  }

  /// Force-ends the in-flight trajectory (if any), as when the frame source
  /// itself ends — a footage clip playing out with the ball still tracked. The
  /// rally can't continue without frames, so the ball is declared lost *now*
  /// instead of never, letting the referee resolve the rally.
  List<TrackerEvent> flush(int timestampMs) {
    if (_prev == null) return const [];
    return [_loseBall(timestampMs)];
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
    final awayFromNet = netBounceExclusion == 0 ||
        (apex.x - geometry.netX).abs() >= netBounceExclusion;
    if (descending &&
        ascending &&
        awayFromNet &&
        geometry.containsSurface(apex.x, apex.y)) {
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
    _outlierCount = 0;
    _filter.reset();
  }
}
