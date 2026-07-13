import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/training/shot_analyzer.dart';
import 'package:pong_ai/core/training/training_feedback.dart';

/// Build a summary from raw (depth, lateral, score, timestampMs) shots.
TrainingSummary _summary(List<Shot> shots) => TrainingSummary(shots);

Shot _shot({
  double depth = 0.75,
  double lateral = 0.5,
  double score = 0.8,
  int t = 0,
}) =>
    Shot(timestampMs: t, speed: 1, depth: depth, lateral: lateral, score: score);

void main() {
  group('TrainingFeedback', () {
    test('empty session has no data and a null focus', () {
      final fb = TrainingFeedback(_summary(const []));
      expect(fb.hasData, isFalse);
      expect(fb.dimensions, isEmpty);
      expect(fb.weakest, isNull);
      expect(fb.strongest, isNull);
      expect(fb.focusTip, isNull);
      expect(fb.report(), contains('No shots recorded yet'));
    });

    test('single shot only assesses placement accuracy', () {
      final fb = TrainingFeedback(_summary([_shot()]));
      expect(fb.dimensions, hasLength(1));
      expect(fb.dimensions.single.name, 'Placement accuracy');
    });

    test('off-target depth makes placement the focus with a direction cue', () {
      // Target 0.75 with tolerance 0.35; landing short at 0.30 → big error.
      final shots = [
        for (var i = 0; i < 4; i++)
          _shot(depth: 0.30, lateral: 0.5, t: i * 1000),
      ];
      final fb = TrainingFeedback(_summary(shots));
      final worst = fb.weakest!;
      expect(worst.name, 'Placement accuracy');
      expect(worst.score, lessThan(0.5));
      expect(fb.focusTip, contains('deeper'));
    });

    test('overshooting suggests bringing the ball in shorter', () {
      final shots = [
        for (var i = 0; i < 4; i++)
          _shot(depth: 1.0, lateral: 0.5, t: i * 1000),
      ];
      final fb = TrainingFeedback(_summary(shots));
      expect(fb.weakest!.name, 'Placement accuracy');
      expect(fb.focusTip, contains('shorter'));
    });

    test('scattered lateral placement becomes the focus', () {
      // On-target depth + steady tempo, but lateral all over the width.
      final laterals = [0.0, 1.0, 0.0, 1.0];
      final shots = [
        for (var i = 0; i < laterals.length; i++)
          _shot(depth: 0.75, lateral: laterals[i], t: i * 1000),
      ];
      final fb = TrainingFeedback(_summary(shots));
      expect(fb.weakest!.name, 'Lateral consistency');
      expect(fb.focusTip, contains('side-to-side'));
    });

    test('erratic timing makes rhythm the weak point', () {
      // Same on-target depth & lateral, but wildly irregular gaps.
      final times = [0, 200, 5000, 5200];
      final shots = [
        for (var i = 0; i < times.length; i++)
          _shot(depth: 0.75, lateral: 0.5, t: times[i]),
      ];
      final fb = TrainingFeedback(_summary(shots));
      expect(fb.weakest!.name, 'Rhythm');
      expect(fb.focusTip, contains('tempo'));
    });

    test('a clean session earns encouragement, not a fix-it cue', () {
      // On-target depth, tight lateral, steady tempo → everything near ideal.
      final shots = [
        for (var i = 0; i < 4; i++)
          _shot(depth: 0.75, lateral: 0.5, t: i * 1000),
      ];
      final fb = TrainingFeedback(_summary(shots));
      expect(fb.weakest!.score, greaterThanOrEqualTo(0.8));
      expect(fb.focusTip, contains('keep'));
      expect(fb.report(), contains('Focus next'));
    });

    test('report lists every scored dimension', () {
      final shots = [
        for (var i = 0; i < 3; i++)
          _shot(depth: 0.6, lateral: 0.4 + i * 0.1, t: i * 800),
      ];
      final report = TrainingFeedback(_summary(shots)).report();
      expect(report, contains('Placement accuracy'));
      expect(report, contains('Depth consistency'));
      expect(report, contains('Lateral consistency'));
      expect(report, contains('Rhythm'));
    });

    test('respects a custom target depth via config', () {
      // Landing deep at 0.95; with a shallow 0.3 target that is a big overshoot.
      final shots = [
        for (var i = 0; i < 4; i++)
          _shot(depth: 0.95, lateral: 0.5, t: i * 1000),
      ];
      final fb = TrainingFeedback(
        _summary(shots),
        config: const TrainingConfig(targetDepth: 0.3),
      );
      expect(fb.weakest!.name, 'Placement accuracy');
      expect(fb.focusTip, contains('shorter'));
    });
  });
}
