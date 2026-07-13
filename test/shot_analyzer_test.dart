import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/training/shot_analyzer.dart';
import 'package:pong_ai/core/vision/detection.dart';

/// One frame whose ball detection is centered at ([x], [y]).
FrameResult _frame(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(
        label: 'ball',
        confidence: 0.9,
        box: BBox(x - 0.01, y - 0.01, 0.02, 0.02),
      ),
    );

/// A down-up arc across [xs]; the y-path makes the apex (bounce) land on the
/// 3rd sample, so [xs][2] is the landing/bounce x.
List<FrameResult> _arc(List<double> xs, {int startT = 0, int step = 33}) {
  const ys = [0.30, 0.42, 0.50, 0.40, 0.30];
  assert(xs.length == ys.length);
  return [
    for (var i = 0; i < xs.length; i++) _frame(startT + i * step, xs[i], ys[i]),
  ];
}

/// Ball-less frames long enough to trip [BallTracker.maxGapFrames], cleanly
/// separating one drill stroke from the next.
List<FrameResult> _gap(int startT, {int n = 8, int step = 33}) =>
    [for (var i = 0; i < n; i++) FrameResult(timestampMs: startT + i * step)];

List<Shot> _run(ShotAnalyzer analyzer, List<FrameResult> frames) {
  final out = <Shot>[];
  for (final f in frames) {
    final s = analyzer.onFrame(f);
    if (s != null) out.add(s);
  }
  return out;
}

void main() {
  group('ShotAnalyzer — single stroke', () {
    test('records one shot with the target-side landing depth', () {
      final analyzer = ShotAnalyzer();
      // Player left, target right; apex lands at x=0.875 → depth 0.75.
      final shots = _run(analyzer, _arc([0.30, 0.60, 0.875, 0.95, 0.98]));

      expect(shots, hasLength(1));
      expect(shots.single.depth, closeTo(0.75, 1e-9));
      expect(analyzer.shots, hasLength(1));
    });

    test('a well-placed fast stroke grades as excellent', () {
      final analyzer = ShotAnalyzer();
      final shots = _run(analyzer, _arc([0.30, 0.60, 0.875, 0.95, 0.98]));

      // depth == targetDepth (placement 1) and pace saturates → score 1.
      expect(shots.single.score, closeTo(1.0, 1e-9));
      expect(shots.single.grade, ShotGrade.excellent);
    });

    test('a poorly placed stroke grades as poor (placement-only config)', () {
      final analyzer = ShotAnalyzer(
        config: const TrainingConfig(placementWeight: 1),
      );
      // Lands at x=0.70 → depth 0.4, far from target 0.75 by exactly tolerance.
      final shots = _run(analyzer, _arc([0.30, 0.60, 0.70, 0.80, 0.85]));

      expect(shots.single.depth, closeTo(0.4, 1e-9));
      expect(shots.single.score, closeTo(0.0, 1e-9));
      expect(shots.single.grade, ShotGrade.poor);
    });

    test('onFrame returns exactly the shot it appended', () {
      final analyzer = ShotAnalyzer();
      Shot? returned;
      for (final f in _arc([0.30, 0.60, 0.875, 0.95, 0.98])) {
        returned ??= analyzer.onFrame(f);
      }
      expect(returned, isNotNull);
      expect(identical(returned, analyzer.shots.single), isTrue);
    });
  });

  group('ShotAnalyzer — what does NOT count', () {
    test('a bounce on the player side is not a shot', () {
      final analyzer = ShotAnalyzer();
      // Ball stays on the left (player) half the whole arc: no target bounce.
      final shots = _run(analyzer, _arc([0.10, 0.20, 0.30, 0.40, 0.45]));
      expect(shots, isEmpty);
    });

    test('losing the ball mid-flight cancels the outgoing stroke', () {
      final analyzer = ShotAnalyzer();
      // Cross the net, then lose the ball before any bounce is seen.
      final frames = [
        _frame(0, 0.30, 0.30),
        _frame(33, 0.60, 0.35), // net-cross → outgoing
        ..._gap(66), // ball lost → outgoing cancelled
      ];
      expect(_run(analyzer, frames), isEmpty);
    });
  });

  group('ShotAnalyzer — mirrored player side', () {
    test('player on the right grades a bounce on the left half', () {
      final analyzer = ShotAnalyzer(
        config: const TrainingConfig(playerSide: TableSide.right),
      );
      // Player right, target left; apex lands at x=0.125 → depth 0.75.
      final shots = _run(analyzer, _arc([0.70, 0.40, 0.125, 0.05, 0.02]));

      expect(shots, hasLength(1));
      expect(shots.single.depth, closeTo(0.75, 1e-9));
    });
  });

  group('TrainingSummary', () {
    ShotAnalyzer sessionLanding(List<double> landings) {
      final analyzer = ShotAnalyzer();
      var t = 0;
      for (final x in landings) {
        _run(analyzer, _arc([0.30, 0.60, x, x + 0.05, x + 0.07], startT: t));
        t += 5 * 33;
        _run(analyzer, _gap(t));
        t += 8 * 33;
      }
      return analyzer;
    }

    test('aggregates count, average depth and grade buckets', () {
      // Two on-target (0.75) strokes and one short (0.40) stroke.
      final summary = sessionLanding([0.875, 0.875, 0.70]).summary;

      expect(summary.shotCount, 3);
      expect(summary.averageDepth, closeTo((0.75 + 0.75 + 0.4) / 3, 1e-9));
      expect(summary.gradeCount(ShotGrade.excellent), 2);
    });

    test('identical placement yields full consistency', () {
      final summary = sessionLanding([0.875, 0.875, 0.875]).summary;
      expect(summary.consistency, closeTo(1.0, 1e-9));
      expect(summary.overallGrade, 'A');
    });

    test('spread placement lowers consistency', () {
      final tight = sessionLanding([0.80, 0.85, 0.90]).summary;
      final wide = sessionLanding([0.55, 0.875, 1.00]).summary;
      expect(wide.consistency, lessThan(tight.consistency));
    });

    test('empty session reports no shots', () {
      const summary = TrainingSummary([]);
      expect(summary.shotCount, 0);
      expect(summary.overallGrade, '–');
      expect(summary.consistency, 0);
      expect(summary.report(), contains('No shots recorded'));
    });

    test('report is human-readable and deterministic', () {
      final report = sessionLanding([0.875, 0.875]).summary.report();
      expect(report, contains('Training summary'));
      expect(report, contains('2 shots'));
      expect(report, contains('grade A'));
    });
  });

  test('reset clears all session state', () {
    final analyzer = ShotAnalyzer();
    _run(analyzer, _arc([0.30, 0.60, 0.875, 0.95, 0.98]));
    expect(analyzer.shots, isNotEmpty);

    analyzer.reset();
    expect(analyzer.shots, isEmpty);
    // A fresh stroke after reset is tracked from a clean slate.
    _run(analyzer, _arc([0.30, 0.60, 0.875, 0.95, 0.98], startT: 10000));
    expect(analyzer.shots, hasLength(1));
  });
}
