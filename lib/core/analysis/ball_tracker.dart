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
  const NetCrossEvent(
    super.timestampMs,
    this.from,
    this.to, {
    this.originNearPlayer,
    this.originOffFrame = false,
  });

  final TableSide from;
  final TableSide to;

  /// Whether the shot that produced this crossing plausibly came off a
  /// racket: the ball's most recent direction reversal happened close to a
  /// detected player (or off-frame where a player had gone). False when the
  /// reversal happened in open space with both players visible elsewhere —
  /// the signature of a dead ball rebounding (off the floor or barrier)
  /// rather than a return. Null when the tracker couldn't classify (no
  /// reversal observed, or no people data), preserving default handling.
  ///
  /// Caution: near-player is *suggestive*, not conclusive — a dead ball can
  /// rebound right where a player happens to stand. [originOffFrame] marks
  /// the one sub-case that is strong evidence on its own.
  final bool? originNearPlayer;

  /// Whether the reversal behind this crossing happened *out of view while a
  /// player was off-frame* — the ball left the frame to a player who had
  /// chased it out, and came back: a return nobody on-camera could have
  /// faked. Strong evidence the crossing is a real shot.
  final bool originOffFrame;

  @override
  String toString() => 'NetCross(@$timestampMs, $from->$to'
      '${originNearPlayer == null ? '' : ', nearPlayer=$originNearPlayer'}'
      '${originOffFrame ? ', offFrame' : ''})';
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
    this.netCrossHysteresis = 0,
    this.extendedGapFrames,
    this.frameTopExitY = 0.15,
    BallTrajectoryFilter? filter,
  })  : assert(minBounceSpeed >= 0),
        assert(maxGapFrames >= 0),
        assert(maxJump == null || maxJump > 0),
        assert(netBounceExclusion >= 0),
        assert(extendedGapFrames == null || extendedGapFrames >= maxGapFrames),
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

  /// Minimum normalized distance the ball must travel PAST the net line
  /// before a side change registers as a crossing. A dead ball dribbling
  /// along the net oscillates across the line by a pixel or two and, on a
  /// dense detection track, fires a storm of phantom crossings; real shots
  /// clear the line by far more. `0` (the default) keeps the raw
  /// line-crossing behaviour.
  final double netCrossHysteresis;

  /// The hysteresis-confirmed side of the ball (null until the ball has been
  /// clearly on one side). Only used when [netCrossHysteresis] > 0.
  TableSide? _confirmedSide;

  /// A more patient ball-loss budget applied when the evidence says play is
  /// probably still live despite the missing ball: the ball was last tracked
  /// near the top of the frame (a lob arcing out of view — it will come back
  /// down), or fewer than two players are visible (someone has left the frame
  /// to chase the ball, and the camera can't see the whole exchange). Without
  /// this, a high defensive lob or an off-frame retrieval gets scored as a
  /// rally-ending ball loss while the point is still being played. `null`
  /// (the default) disables the patience, preserving historical behaviour.
  final int? extendedGapFrames;

  /// Normalized y above which (i.e. smaller than) a last-tracked ball counts
  /// as having exited via the top of the frame for [extendedGapFrames].
  final double frameTopExitY;

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
      return _handleMissingBall(frame.timestampMs, frame.people.length);
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
        return _handleMissingBall(frame.timestampMs, frame.people.length);
      }
      accepted = recovered;
    }

    _missedBeforeSample = _missedFrames;
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

    // Track x-direction reversals so a crossing can be classified as a real
    // return (reversal at a player) vs dead-ball drift (reversal in space).
    final vx = sample.x - prev.x;
    final vxSign = vx > 0.003 ? 1 : (vx < -0.003 ? -1 : 0);
    if (vxSign != 0) {
      if (_lastVxSign != 0 && vxSign != _lastVxSign) {
        final gapFrames = _missedBeforeSample;
        _lastReversalNearPlayer =
            _classifyReversal(prev, frame, afterGap: gapFrames >= 6);
      }
      _lastVxSign = vxSign;
    }

    final netEvent = _detectNetCross(prev, sample);
    if (netEvent != null) events.add(netEvent);

    final bounce = _detectBounce(prev, vy);
    if (bounce != null) events.add(bounce);

    _lastVy = vy;
    _prev = sample;
    _missedBeforeSample = 0;
    return events;
  }

  /// Missed-frame count observed just before the current accepted sample —
  /// lets a reversal detected right after a gap be classified as having
  /// happened out of view.
  int _missedBeforeSample = 0;

  /// Sign of the last observed horizontal velocity, and whether the ball's
  /// most recent x-direction *reversal* happened near a detected player (a
  /// racket contact) rather than in open space (a dead-ball rebound). Used to
  /// classify each net crossing's origin — see [NetCrossEvent.originNearPlayer].
  int _lastVxSign = 0;
  bool? _lastReversalNearPlayer;
  bool _lastReversalOffFrame = false;

  /// Distance (normalized) within which a reversal counts as near a player.
  static const double _playerReach = 0.15;

  bool? _classifyReversal(
    BallSample at,
    FrameResult frame, {
    required bool afterGap,
  }) {
    final people = frame.people;
    _lastReversalOffFrame = false;
    if (afterGap) {
      // The reversal happened while the ball was out of view. If a player is
      // also off-frame, this is consistent with an off-frame return; if both
      // players are visible elsewhere, nobody can have hit it — the reversal
      // was the dead ball rebounding (floor, barrier, net post).
      if (people.isEmpty) return null;
      _lastReversalOffFrame = people.length < 2;
      return people.length < 2;
    }
    if (people.isEmpty) return null;
    for (final p in people) {
      final dx = (at.x - at.x.clamp(p.box.left, p.box.left + p.box.width));
      final dy = (at.y - at.y.clamp(p.box.top, p.box.top + p.box.height));
      if (dx * dx + dy * dy <= _playerReach * _playerReach) return true;
    }
    return false;
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

  List<TrackerEvent> _handleMissingBall(int timestampMs, int peopleVisible) {
    final last = _prev;
    if (last == null) return const [];
    _missedFrames++;
    // Patience: a ball that left via the top of the frame (a lob) is coming
    // back, and a missing player is probably off-frame playing it — hold the
    // rally open longer before declaring the ball lost.
    final patient = extendedGapFrames != null &&
        (last.y < frameTopExitY || peopleVisible < 2);
    final budget = patient ? extendedGapFrames! : maxGapFrames;
    if (_missedFrames > budget) {
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
    TableSide from;
    TableSide to;
    if (netCrossHysteresis > 0) {
      // Hysteresis: the side only flips once the ball is clearly past the
      // line, so net-dribble jitter cannot fire crossings.
      final clear = (now.x - geometry.netX).abs() >= netCrossHysteresis;
      final side = geometry.sideOf(now.x);
      if (_confirmedSide == null) {
        // Seed from the previous sample (the first ever tracked position
        // never reaches this method on its own), so a crossing on the very
        // next sample is not silently swallowed as the seed.
        if ((prev.x - geometry.netX).abs() >= netCrossHysteresis) {
          _confirmedSide = geometry.sideOf(prev.x);
        } else {
          if (clear) _confirmedSide = side;
          return null;
        }
      }
      if (!clear || side == _confirmedSide) return null;
      from = _confirmedSide!;
      to = side;
      _confirmedSide = side;
    } else {
      from = geometry.sideOf(prev.x);
      to = geometry.sideOf(now.x);
    }
    if (from == to) return null;
    // A ball passing the net line *below* the entire table-surface band went
    // under (or into) the net, not over it — that is not a legal crossing,
    // so don't report one (the rally then ends via the no-return/loss paths,
    // which attribute the fault to the hitter correctly).
    final yAtNet = (prev.y + now.y) / 2;
    if (geometry.bottom < 1.0 && yAtNet > geometry.bottom) return null;
    return NetCrossEvent(
      now.timestampMs,
      from,
      to,
      originNearPlayer: _lastReversalNearPlayer,
      originOffFrame: _lastReversalOffFrame,
    );
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
    _confirmedSide = null;
    _lastVxSign = 0;
    _lastReversalNearPlayer = null;
    _lastReversalOffFrame = false;
    _missedBeforeSample = 0;
    _prev = null;
    _lastVy = null;
    _missedFrames = 0;
    _outlierCount = 0;
    _filter.reset();
  }
}
