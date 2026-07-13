/// Pure-Dart real-world ball-speed estimation.
///
/// Match-analysis apps (e.g. "Ball AI") headline **ball speed** — the km/h of a
/// smash or a rally exchange — because it is the single most legible measure of
/// how hard a player is hitting. Every prior analytics layer worked in the
/// vision pipeline's *normalized* `[0,1]` frame coordinates, which have no
/// physical meaning on their own; this layer turns the ball's frame motion into
/// metres-per-second / km/h using the calibrated [TableGeometry] as the ruler.
///
/// The scale comes from a known real-world dimension: an ITTF table is
/// **2.74 m** long, and with the phone placed on the *side* of the table that
/// length spans the frame horizontally (the net is a vertical line at
/// [TableGeometry.netX], so the ball's back-and-forth travels along the frame
/// x-axis). So one normalized x-unit maps to `tableLengthMeters / (right - left)`
/// metres. We deliberately estimate speed from the **horizontal** component
/// only: from a side camera the vertical (frame-y) motion is dominated by the
/// ball's *height* and the foreshortened near/far depth, which cannot be scaled
/// to metres reliably, whereas the along-table x-motion is the dominant, cleanly
/// scalable direction of a rally shot.
///
/// Like the rest of `core/`, it has no Flutter or vision-plugin dependencies, so
/// it is unit-testable against synthetic trajectories and replayed benchmark
/// clips without a camera.
library;

import '../vision/detection.dart';
import 'ball_tracker.dart';

/// The ITTF regulation table length (net-to-net-to-baseline), in metres.
const double kTableLengthMeters = 2.74;

/// Incrementally estimates the ball's real-world speed from per-frame
/// detections.
///
/// Feed it each [FrameResult] via [observe] (frames with no ball are skipped);
/// it converts the along-table displacement between consecutive detections into
/// a km/h reading and accumulates [maxKmh] / [averageKmh]. Rebuild it on the
/// calibrated geometry (as [MatchController] does) so the metres-per-unit ruler
/// matches the inferred table.
class BallSpeedEstimator {
  BallSpeedEstimator({
    this.geometry = const TableGeometry(),
    this.tableLengthMeters = kTableLengthMeters,
    this.maxGapMs = 100,
    this.maxPlausibleKmh = 250,
  })  : assert(tableLengthMeters > 0),
        assert(maxGapMs > 0),
        assert(maxPlausibleKmh > 0);

  /// The calibrated table, whose x-span is the physical ruler.
  final TableGeometry geometry;

  /// Physical table length the x-span maps to (default ITTF 2.74 m).
  final double tableLengthMeters;

  /// Intervals longer than this (ms) are assumed to bridge a ball-loss gap and
  /// are not turned into a speed reading — otherwise a small displacement over a
  /// long gap would understate the speed. At ~30 fps this tolerates a few
  /// dropped frames while rejecting reappearances after a real loss.
  final int maxGapMs;

  /// Readings implying a faster-than-physically-plausible ball are dropped as
  /// spurious detections (a detector false positive teleporting across the
  /// frame) rather than polluting the max.
  final double maxPlausibleKmh;

  double? _prevX;
  int? _prevMs;
  final List<double> _speeds = [];

  /// Metres each normalized x-unit represents, given the calibrated table span.
  double get metersPerUnitX =>
      tableLengthMeters / (geometry.right - geometry.left);

  /// Number of accepted km/h readings so far.
  int get sampleCount => _speeds.length;

  /// Whether at least one speed reading has been accumulated.
  bool get hasData => _speeds.isNotEmpty;

  /// The fastest ball speed seen, in km/h (0 when no data).
  double get maxKmh =>
      _speeds.isEmpty ? 0 : _speeds.reduce((a, b) => a > b ? a : b);

  /// The mean ball speed across all readings, in km/h (0 when no data).
  double get averageKmh =>
      _speeds.isEmpty ? 0 : _speeds.reduce((a, b) => a + b) / _speeds.length;

  /// Fold one frame's ball detection into the running speed stats. Frames with
  /// no ball are ignored (they leave the previous sample in place so a short
  /// dropout does not reset the estimate).
  void observe(FrameResult frame) {
    final ball = frame.ball;
    if (ball == null) return;
    addBall(frame.timestampMs, ball.box.centerX);
  }

  /// Lower-level entry point: fold a ball centre-x at [timestampMs].
  void addBall(int timestampMs, double x) {
    final px = _prevX;
    final pms = _prevMs;
    _prevX = x;
    _prevMs = timestampMs;
    if (px == null || pms == null) return;

    final dtMs = timestampMs - pms;
    // Out-of-order/duplicate frames, or a gap large enough that the ball was
    // lost in between, don't yield a meaningful speed.
    if (dtMs <= 0 || dtMs > maxGapMs) return;

    final dxMeters = (x - px).abs() * metersPerUnitX;
    final kmh = dxMeters / (dtMs / 1000.0) * 3.6;
    if (kmh > maxPlausibleKmh) return;
    _speeds.add(kmh);
  }

  /// Forget all state (e.g. to start a new match).
  void reset() {
    _prevX = null;
    _prevMs = null;
    _speeds.clear();
  }
}
