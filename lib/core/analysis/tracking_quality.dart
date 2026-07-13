/// Pure-Dart tracking-quality (detection-health) analytics.
///
/// The objective's headline deployment story is "place the phone table-side and
/// let the app keep score" — which only works if the phone is positioned so the
/// model can actually *see* the ball and both players. Every prior analytics
/// layer assumed the detections were good; none measured whether they were. The
/// vision pipeline has carried a per-detection [Detection.confidence] (and
/// per-keypoint confidences) since iteration 1, but they were only ever used to
/// *select* the best detection, never mined into a health signal the user can
/// act on.
///
/// [TrackingQualityAnalyzer] folds every frame into a running measure of how
/// reliably the pipeline is tracking: how often the ball is detected, how
/// confident those detections are, and how often *both* players (both ends of
/// the table) are in view. It rolls those into a single [qualityScore] / letter
/// [grade] plus a plain-language placement [hint] — so a user can tell at a
/// glance whether the phone is well placed, and a summary/report can flag when
/// the scoring was working from thin evidence.
///
/// Like the rest of `core/`, it has no Flutter or vision-plugin dependencies, so
/// it is unit-testable against synthetic frame streams without a camera.
library;

import '../vision/detection.dart';

/// Incrementally measures detection health across a stream of [FrameResult]s.
///
/// Feed it every frame via [observe] (including calibration warm-up frames —
/// tracking health is independent of scoring). Read [qualityScore] / [grade] /
/// [hint] for an overall verdict, or the component rates ([ballDetectionRate],
/// [averageBallConfidence], [twoPlayerRate]) individually.
class TrackingQualityAnalyzer {
  TrackingQualityAnalyzer({
    this.ballWeight = 0.4,
    this.ballConfidenceWeight = 0.2,
    this.playerWeight = 0.4,
  }) : assert(
          (ballWeight + ballConfidenceWeight + playerWeight - 1.0).abs() < 1e-9,
          'weights must sum to 1',
        );

  /// How much the fraction of frames with a ball detection contributes to
  /// [qualityScore].
  final double ballWeight;

  /// How much the mean ball-detection confidence contributes to [qualityScore].
  final double ballConfidenceWeight;

  /// How much the fraction of frames with *both* players visible contributes to
  /// [qualityScore].
  final double playerWeight;

  int _frames = 0;
  int _ballFrames = 0;
  double _ballConfidenceSum = 0;
  int _anyPlayerFrames = 0;
  int _twoPlayerFrames = 0;

  /// Total frames observed.
  int get frameCount => _frames;

  /// Whether any frame has been observed (rates are undefined otherwise).
  bool get hasData => _frames > 0;

  /// Fraction of frames in which the ball was detected (0..1).
  double get ballDetectionRate => _frames == 0 ? 0 : _ballFrames / _frames;

  /// Mean confidence of the frames where the ball *was* detected (0..1). Zero
  /// when the ball was never seen.
  double get averageBallConfidence =>
      _ballFrames == 0 ? 0 : _ballConfidenceSum / _ballFrames;

  /// Fraction of frames in which at least one player was detected (0..1).
  double get anyPlayerRate => _frames == 0 ? 0 : _anyPlayerFrames / _frames;

  /// Fraction of frames in which *both* players (both ends of the table) were
  /// detected (0..1) — the strongest signal that the phone frames the whole
  /// table rather than clipping one end.
  double get twoPlayerRate => _frames == 0 ? 0 : _twoPlayerFrames / _frames;

  /// An overall detection-health score in [0, 1], a weighted blend of the ball
  /// detection rate, mean ball confidence, and the both-players-visible rate.
  double get qualityScore {
    if (_frames == 0) return 0;
    return ballWeight * ballDetectionRate +
        ballConfidenceWeight * averageBallConfidence +
        playerWeight * twoPlayerRate;
  }

  /// An A–F letter grade for the tracking health, from [qualityScore]. Uses the
  /// same thresholds as the training-session grade.
  String get grade {
    if (_frames == 0) return 'N/A';
    final s = qualityScore;
    if (s >= 0.85) return 'A';
    if (s >= 0.7) return 'B';
    if (s >= 0.55) return 'C';
    if (s >= 0.4) return 'D';
    return 'F';
  }

  /// A plain-language placement hint, targeting whichever signal is weakest so
  /// the user knows how to reposition the phone for better tracking.
  String get hint {
    if (_frames == 0) return 'No frames analysed yet.';
    if (twoPlayerRate < 0.5) {
      return 'Both players are often out of frame — move the phone back or '
          'lower so the whole table and both ends are visible.';
    }
    if (ballDetectionRate < 0.4) {
      return 'The ball is frequently lost — improve lighting or move the phone '
          'closer to the table for a clearer view of the ball.';
    }
    if (averageBallConfidence < 0.4) {
      return 'Ball detections are low-confidence — reduce background clutter '
          'and glare behind the table.';
    }
    return 'Tracking looks healthy for this phone placement.';
  }

  /// Fold one frame's detections into the running health stats.
  void observe(FrameResult frame) {
    _frames++;
    final ball = frame.ball;
    if (ball != null) {
      _ballFrames++;
      _ballConfidenceSum += ball.confidence;
    }
    final players = frame.people.length;
    if (players >= 1) _anyPlayerFrames++;
    if (players >= 2) _twoPlayerFrames++;
  }

  /// A human-readable tracking-health section for the exported report.
  String report() {
    if (_frames == 0) {
      return 'Tracking quality\n  • no frames analysed';
    }
    return [
      'Tracking quality — grade $grade '
          '(${(qualityScore * 100).round()}%)',
      '  • ball detected in ${(ballDetectionRate * 100).round()}% of frames '
          '(avg confidence ${(averageBallConfidence * 100).round()}%)',
      '  • both players visible in ${(twoPlayerRate * 100).round()}% of frames',
      '  • $hint',
    ].join('\n');
  }

  /// Forget all state (e.g. to start a new match).
  void reset() {
    _frames = 0;
    _ballFrames = 0;
    _ballConfidenceSum = 0;
    _anyPlayerFrames = 0;
    _twoPlayerFrames = 0;
  }
}
