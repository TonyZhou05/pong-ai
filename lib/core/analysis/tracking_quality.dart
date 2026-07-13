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

import 'dart:collection';

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
    this.requireBothPlayers = true,
    this.recentWindow = 30,
    this.recentMinFrames = 10,
  }) : assert(
          (ballWeight + ballConfidenceWeight + playerWeight - 1.0).abs() < 1e-9,
          'weights must sum to 1',
        );

  /// How much the fraction of frames with a ball detection contributes to
  /// [qualityScore].
  final double ballWeight;

  /// How much the mean ball-detection confidence contributes to [qualityScore].
  final double ballConfidenceWeight;

  /// How much the player-visibility rate contributes to [qualityScore].
  final double playerWeight;

  /// Whether good placement needs *both* players (both ends of the table) in
  /// frame. `true` for a match — the phone must see both halves; `false` for
  /// **training mode**, where a single player practises against a rebound net so
  /// only that one player needs to be visible. Controls which player-visibility
  /// rate ([twoPlayerRate] vs [anyPlayerRate]) feeds [qualityScore] / [hint] /
  /// [report].
  final bool requireBothPlayers;

  /// How many of the most-recent frames feed the *live* placement signals
  /// ([recentQualityScore] / [recentGrade] / [recentHint] / [isPlacementPoor]).
  /// The whole-session rates (used for the post-match report) accumulate from
  /// frame 1 and so never recover once placement is fixed mid-session — a live
  /// "reposition the phone" nudge needs a short trailing window that reflects
  /// the *current* placement instead.
  final int recentWindow;

  /// How many frames must be in the trailing window before the live placement
  /// signals ([hasRecentData]) are trusted — avoids nagging on the first frame
  /// or two before there is enough evidence.
  final int recentMinFrames;

  /// The player-visibility rate that feeds the health score, per
  /// [requireBothPlayers].
  double get playerVisibilityRate =>
      requireBothPlayers ? twoPlayerRate : anyPlayerRate;

  int _frames = 0;
  int _ballFrames = 0;
  double _ballConfidenceSum = 0;
  int _anyPlayerFrames = 0;
  int _twoPlayerFrames = 0;

  /// The trailing window of per-frame health used for the live placement nudge.
  final Queue<_FrameHealth> _recent = Queue<_FrameHealth>();

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
    return _scoreFrom(
      ballDetectionRate,
      averageBallConfidence,
      playerVisibilityRate,
    );
  }

  /// An A–F letter grade for the tracking health, from [qualityScore]. Uses the
  /// same thresholds as the training-session grade.
  String get grade => _frames == 0 ? 'N/A' : _gradeFor(qualityScore);

  /// A plain-language placement hint, targeting whichever signal is weakest so
  /// the user knows how to reposition the phone for better tracking.
  String get hint {
    if (_frames == 0) return 'No frames analysed yet.';
    return _hintFor(
      playerVisibilityRate,
      ballDetectionRate,
      averageBallConfidence,
    );
  }

  // --- Live (trailing-window) placement signals ---------------------------

  /// Whether the trailing window holds enough frames to trust the live signals.
  bool get hasRecentData => _recent.length >= recentMinFrames;

  /// Ball-detection rate over the trailing [recentWindow] frames.
  double get recentBallDetectionRate =>
      _recent.isEmpty ? 0 : _recent.where((f) => f.hasBall).length / _recent.length;

  /// Mean ball confidence over the trailing window's ball frames (0 when none).
  double get recentAverageBallConfidence {
    final balls = _recent.where((f) => f.hasBall);
    if (balls.isEmpty) return 0;
    final sum = balls.fold<double>(0, (a, f) => a + f.ballConfidence);
    return sum / balls.length;
  }

  /// Player-visibility rate over the trailing window, per [requireBothPlayers].
  double get recentPlayerVisibilityRate {
    if (_recent.isEmpty) return 0;
    final need = requireBothPlayers ? 2 : 1;
    return _recent.where((f) => f.playerCount >= need).length / _recent.length;
  }

  /// Detection-health score over the trailing window — the live counterpart to
  /// [qualityScore] that reflects the *current* phone placement.
  double get recentQualityScore => _recent.isEmpty
      ? 0
      : _scoreFrom(
          recentBallDetectionRate,
          recentAverageBallConfidence,
          recentPlayerVisibilityRate,
        );

  /// A–F grade over the trailing window.
  String get recentGrade =>
      _recent.isEmpty ? 'N/A' : _gradeFor(recentQualityScore);

  /// A live placement hint over the trailing window.
  String get recentHint => _recent.isEmpty
      ? 'No frames analysed yet.'
      : _hintFor(
          recentPlayerVisibilityRate,
          recentBallDetectionRate,
          recentAverageBallConfidence,
        );

  /// Whether the phone currently seems poorly placed — enough recent evidence
  /// and a trailing-window score at grade D or worse (< 0.55). Drives the live
  /// "reposition the phone" nudge; recovers on its own once placement improves.
  bool get isPlacementPoor => hasRecentData && recentQualityScore < 0.55;

  double _scoreFrom(double ballRate, double ballConf, double playerRate) =>
      ballWeight * ballRate +
      ballConfidenceWeight * ballConf +
      playerWeight * playerRate;

  static String _gradeFor(double s) {
    if (s >= 0.85) return 'A';
    if (s >= 0.7) return 'B';
    if (s >= 0.55) return 'C';
    if (s >= 0.4) return 'D';
    return 'F';
  }

  String _hintFor(double playerRate, double ballRate, double ballConf) {
    if (playerRate < 0.5) {
      return requireBothPlayers
          ? 'Both players are often out of frame — move the phone back or '
              'lower so the whole table and both ends are visible.'
          : 'You are often out of frame — reposition the phone so your whole '
              'body and the target half are visible.';
    }
    if (ballRate < 0.4) {
      return 'The ball is frequently lost — improve lighting or move the phone '
          'closer to the table for a clearer view of the ball.';
    }
    if (ballConf < 0.4) {
      return 'Ball detections are low-confidence — reduce background clutter '
          'and glare behind the table.';
    }
    return 'Tracking looks healthy for this phone placement.';
  }

  /// Fold one frame's detections into the running health stats.
  void observe(FrameResult frame) {
    _frames++;
    final ball = frame.ball;
    final ballConfidence = ball?.confidence ?? 0;
    if (ball != null) {
      _ballFrames++;
      _ballConfidenceSum += ballConfidence;
    }
    final players = frame.people.length;
    if (players >= 1) _anyPlayerFrames++;
    if (players >= 2) _twoPlayerFrames++;

    _recent.addLast(_FrameHealth(ball != null, ballConfidence, players));
    while (_recent.length > recentWindow) {
      _recent.removeFirst();
    }
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
      '  • ${requireBothPlayers ? 'both players' : 'player'} visible in '
          '${(playerVisibilityRate * 100).round()}% of frames',
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
    _recent.clear();
  }
}

/// One frame's detection health, retained in the trailing window.
class _FrameHealth {
  const _FrameHealth(this.hasBall, this.ballConfidence, this.playerCount);

  final bool hasBall;
  final double ballConfidence;
  final int playerCount;
}
