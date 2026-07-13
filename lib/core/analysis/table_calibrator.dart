/// Pure-Dart auto-calibration of the table geometry from observed detections.
///
/// The objective is "place the phone on the side of the table and the app just
/// works". For the [BallTracker] to reject off-table (floor/chair) bounces it
/// needs a calibrated [TableGeometry], but forcing the user to hand-mark the
/// table corners is friction. This calibrator watches a short warm-up of live
/// [FrameResult]s and infers the table surface band plus the net line from where
/// the ball actually travels — and, when available, where the two players stand.
///
/// It is intentionally free of any Flutter or plugin dependency so it can be
/// unit-tested against synthetic detections and run inline on the live stream.
///
/// Robustness: the surface bounds are taken from a **trimmed** percentile range
/// of the accumulated ball positions rather than the raw min/max, so a stray
/// floor bounce or a spurious off-table detection does not blow the calibrated
/// rectangle out to the whole frame.
library;

import '../vision/detection.dart';
import 'ball_tracker.dart';

class TableCalibrator {
  TableCalibrator({
    this.minBallSamples = 20,
    this.trim = 0.05,
  })  : assert(minBallSamples > 0, 'need at least one ball sample to calibrate'),
        assert(trim >= 0 && trim < 0.5, 'trim must be in [0, 0.5)');

  /// How many accepted ball samples must accumulate before [calibrate] returns
  /// a geometry instead of null.
  final int minBallSamples;

  /// Fraction trimmed from each end of the sorted coordinate lists before the
  /// surface bounds are read off, to reject outlier (off-table) detections.
  final double trim;

  final List<double> _ballX = <double>[];
  final List<double> _ballY = <double>[];
  final List<double> _playerX = <double>[];

  /// Number of ball positions accumulated so far.
  int get ballSampleCount => _ballX.length;

  /// Whether enough ball samples have been seen to produce a calibration.
  bool get isReady => _ballX.length >= minBallSamples;

  /// Accumulate the ball centroid and any player centroids from one frame.
  void observe(FrameResult frame) {
    final ball = frame.ball;
    if (ball != null) addBall(ball.box.centerX, ball.box.centerY);
    for (final person in frame.people) {
      addPlayer(person.box.centerX);
    }
  }

  /// Accumulate a single ball position (normalized frame coordinates).
  void addBall(double x, double y) {
    _ballX.add(x);
    _ballY.add(y);
  }

  /// Accumulate a single player centroid x (normalized frame coordinate).
  void addPlayer(double x) => _playerX.add(x);

  /// Estimate the table geometry from the accumulated samples, or null if there
  /// are too few samples or the observed spread is degenerate (all samples on a
  /// line), in which case calibration cannot be trusted.
  TableGeometry? calibrate() {
    if (_ballX.length < minBallSamples) return null;

    final xs = List<double>.from(_ballX)..sort();
    final ys = List<double>.from(_ballY)..sort();

    final left = _clamp01(_percentile(xs, trim));
    final right = _clamp01(_percentile(xs, 1 - trim));
    final top = _clamp01(_percentile(ys, trim));
    final bottom = _clamp01(_percentile(ys, 1 - trim));

    // A table has to have area; a degenerate band means the ball never really
    // moved and we should not pretend to have a calibration.
    const minExtent = 0.02;
    if (right - left < minExtent || bottom - top < minExtent) return null;

    final netX = _estimateNet(left, right);

    return TableGeometry(
      netX: netX,
      left: left,
      right: right,
      top: top,
      bottom: bottom,
    );
  }

  /// Discard all accumulated samples (e.g. to re-calibrate).
  void reset() {
    _ballX.clear();
    _ballY.clear();
    _playerX.clear();
  }

  /// The net sits between the two players when we can see them (the phone is on
  /// the side, so the two ends map to the min/max player x); otherwise it falls
  /// back to the horizontal midpoint of the ball's travel. The result is kept
  /// strictly inside the table edges, which [TableGeometry] requires.
  double _estimateNet(double left, double right) {
    double candidate;
    if (_playerX.length >= 2) {
      final pxs = List<double>.from(_playerX)..sort();
      candidate = (pxs.first + pxs.last) / 2;
    } else {
      candidate = (left + right) / 2;
    }
    final inset = (right - left) * 0.01;
    return candidate.clamp(left + inset, right - inset);
  }

  /// Linearly-interpolated quantile of an already-sorted list, [q] in [0,1].
  static double _percentile(List<double> sorted, double q) {
    if (sorted.length == 1) return sorted.first;
    final pos = q * (sorted.length - 1);
    final lo = pos.floor();
    final hi = pos.ceil();
    if (lo == hi) return sorted[lo];
    final frac = pos - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
  }

  static double _clamp01(double v) => v.clamp(0.0, 1.0);
}
