/// Pure-Dart Kalman smoother/predictor for the ball's trajectory.
///
/// The [BallTracker] only *tolerates* frames where the detector loses the
/// (small, motion-blurred) ball — it holds the last accepted sample and waits.
/// It cannot say *where* the ball is during that gap, so the live overlay
/// freezes and a bounce/net-cross that happens mid-blur is measured against a
/// stale position. The architecture (docs/ARCHITECTURE.md §1) calls for fusing
/// the detector with a Kalman-filter tracker to bridge exactly those frames.
///
/// This is that filter. It maintains a constant-velocity motion model in the
/// normalized `[0,1] x [0,1]` image space, smoothing noisy detections and — the
/// point of it — extrapolating a position estimate for any timestamp, including
/// ones with no detection at all.
///
/// Because the constant-velocity state transition is block-diagonal per axis
/// (x and y never couple), it is implemented as two independent 1-D filters,
/// each with a 2-element state `[position, velocity]` and a 2x2 covariance.
/// That keeps the linear algebra small and exact — no 4x4 matrix code — while
/// being mathematically identical to the full 4-state constant-velocity filter.
library;

/// One axis of a constant-velocity Kalman filter.
///
/// State is `[p, v]` (position, velocity); covariance is the symmetric 2x2
/// `[[p00, p01], [p10, p11]]`. Time is in seconds so velocity is units/second.
class _Axis1DKalman {
  _Axis1DKalman({
    required this.measurementNoise,
    required this.accelNoise,
    required this.initialVelocityVariance,
  });

  /// Variance of the position measurement (detection jitter), `R`.
  final double measurementNoise;

  /// Variance of the (unmodeled) acceleration driving the process noise, `q`.
  final double accelNoise;

  /// Initial variance of the velocity estimate — large, because a single
  /// detection tells us nothing about velocity.
  final double initialVelocityVariance;

  double _p = 0; // position
  double _v = 0; // velocity
  double _p00 = 0, _p01 = 0, _p10 = 0, _p11 = 0; // covariance
  bool _initialized = false;

  bool get isInitialized => _initialized;
  double get position => _p;
  double get velocity => _v;

  /// Fold in a new position measurement observed [dtSeconds] after the previous
  /// one (ignored for the very first measurement).
  void observe(double z, double dtSeconds) {
    if (!_initialized) {
      _p = z;
      _v = 0;
      _p00 = measurementNoise;
      _p01 = 0;
      _p10 = 0;
      _p11 = initialVelocityVariance;
      _initialized = true;
      return;
    }
    _predict(dtSeconds);
    _correct(z);
  }

  /// Predict the position [dtSeconds] into the future from the current state,
  /// without mutating the filter (used to answer "where is the ball now?" on a
  /// frame that carried no detection).
  double predictPosition(double dtSeconds) => _p + _v * dtSeconds;

  void _predict(double dt) {
    // State: p += v*dt, v unchanged.
    _p = _p + _v * dt;

    // Covariance: P = F P F^T + Q, with F = [[1, dt], [0, 1]].
    final fp00 = _p00 + dt * _p10 + dt * (_p01 + dt * _p11);
    final fp01 = _p01 + dt * _p11;
    final fp10 = _p10 + dt * _p11;
    final fp11 = _p11;

    // Process noise for a constant-velocity model driven by white acceleration.
    final dt2 = dt * dt;
    final dt3 = dt2 * dt;
    final dt4 = dt2 * dt2;
    final q = accelNoise;
    _p00 = fp00 + q * dt4 / 4;
    _p01 = fp01 + q * dt3 / 2;
    _p10 = fp10 + q * dt3 / 2;
    _p11 = fp11 + q * dt2;
  }

  void _correct(double z) {
    // Innovation covariance S = H P H^T + R, H = [1, 0].
    final s = _p00 + measurementNoise;
    // Kalman gain K = P H^T / S.
    final k0 = _p00 / s;
    final k1 = _p10 / s;

    final y = z - _p; // innovation
    _p = _p + k0 * y;
    _v = _v + k1 * y;

    // P = (I - K H) P.
    final np00 = (1 - k0) * _p00;
    final np01 = (1 - k0) * _p01;
    final np10 = _p10 - k1 * _p00;
    final np11 = _p11 - k1 * _p01;
    _p00 = np00;
    _p01 = np01;
    _p10 = np10;
    _p11 = np11;
  }

  void reset() {
    _initialized = false;
    _p = 0;
    _v = 0;
    _p00 = _p01 = _p10 = _p11 = 0;
  }
}

/// A 2-D constant-velocity Kalman filter over the ball's normalized position.
///
/// Feed it accepted detections via [observe]; read the smoothed [position] and
/// [velocity], or extrapolate to an arbitrary time with [estimateAt] — which is
/// what lets the tracker/UI keep a ball position through detector dropouts.
class BallTrajectoryFilter {
  BallTrajectoryFilter({
    double measurementNoise = 1e-4,
    double accelNoise = 4.0,
    double initialVelocityVariance = 1.0,
  })  : _x = _Axis1DKalman(
          measurementNoise: measurementNoise,
          accelNoise: accelNoise,
          initialVelocityVariance: initialVelocityVariance,
        ),
        _y = _Axis1DKalman(
          measurementNoise: measurementNoise,
          accelNoise: accelNoise,
          initialVelocityVariance: initialVelocityVariance,
        );

  final _Axis1DKalman _x;
  final _Axis1DKalman _y;
  int? _lastTimestampMs;

  /// Whether at least one detection has been folded in.
  bool get hasEstimate => _x.isInitialized;

  /// Timestamp (ms) of the most recent observed detection, or null.
  int? get lastTimestampMs => _lastTimestampMs;

  /// Current smoothed ball position, or null before the first detection.
  ({double x, double y})? get position =>
      hasEstimate ? (x: _x.position, y: _y.position) : null;

  /// Current estimated velocity in normalized units/second, or null.
  ({double vx, double vy})? get velocity =>
      hasEstimate ? (vx: _x.velocity, vy: _y.velocity) : null;

  /// Fold in a detection at [timestampMs]. Out-of-order or duplicate timestamps
  /// are ignored so the filter shares the tracker's strictly-rising clock.
  void observe(int timestampMs, double x, double y) {
    final last = _lastTimestampMs;
    if (last != null && timestampMs <= last) return;
    final dt = last == null ? 0.0 : (timestampMs - last) / 1000.0;
    _x.observe(x, dt);
    _y.observe(y, dt);
    _lastTimestampMs = timestampMs;
  }

  /// Extrapolate the ball position to [timestampMs] using the constant-velocity
  /// model. Returns null before the first detection. For timestamps at or
  /// before the last observation this returns the current smoothed position
  /// (the model only extrapolates forward).
  ({double x, double y})? estimateAt(int timestampMs) {
    if (!hasEstimate) return null;
    final last = _lastTimestampMs!;
    final dt = timestampMs <= last ? 0.0 : (timestampMs - last) / 1000.0;
    return (x: _x.predictPosition(dt), y: _y.predictPosition(dt));
  }

  /// Forget all trajectory state (e.g. between rallies or after a ball-lost).
  void reset() {
    _x.reset();
    _y.reset();
    _lastTimestampMs = null;
  }
}
