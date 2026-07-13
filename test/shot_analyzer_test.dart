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
  group('TrainingConfig.copyWith', () {
    test('flips playerSide (and thus targetSide) while keeping other fields', () {
      const base = TrainingConfig(targetDepth: 0.6, depthTolerance: 0.2);
      final flipped = base.copyWith(playerSide: TableSide.right);

      expect(flipped.playerSide, TableSide.right);
      expect(flipped.targetSide, TableSide.left);
      expect(flipped.targetDepth, 0.6);
      expect(flipped.depthTolerance, 0.2);
      // Untouched call returns an equivalent config.
      expect(base.copyWith().playerSide, base.playerSide);
    });
  });

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
      // The stroke crossed the net but never landed on the target half: it went
      // off the table and is counted as a missed attempt.
      expect(analyzer.missCount, 1);
    });
  });

  group('ShotAnalyzer — on-table accuracy', () {
    test('a stroke that lands on the target half is not a miss', () {
      final analyzer = ShotAnalyzer();
      _run(analyzer, _arc([0.30, 0.60, 0.75, 0.72, 0.70]));
      expect(analyzer.summary.shotCount, 1);
      expect(analyzer.missCount, 0);
      expect(analyzer.summary.onTableRate, 1.0);
    });

    test('landing one of two attempts on the table reads 50% accuracy', () {
      final analyzer = ShotAnalyzer();
      // Stroke 1: lands deep on the target half (a real shot).
      final landed = _arc([0.30, 0.60, 0.75, 0.72, 0.70]);
      // Stroke 2: crosses the net but the ball is lost before any bounce — off
      // the table, a miss.
      final missed = [
        _frame(400, 0.30, 0.30),
        _frame(433, 0.60, 0.35), // net-cross → outgoing
        ..._gap(466), // ball lost → miss
      ];
      _run(analyzer, [...landed, ..._gap(200), ...missed]);

      final summary = analyzer.summary;
      expect(summary.shotCount, 1);
      expect(summary.missedShots, 1);
      expect(summary.attemptedShots, 2);
      expect(summary.onTableRate, closeTo(0.5, 1e-9));
    });

    test('reset clears the miss count', () {
      final analyzer = ShotAnalyzer();
      _run(analyzer, [
        _frame(0, 0.30, 0.30),
        _frame(33, 0.60, 0.35),
        ..._gap(66),
      ]);
      expect(analyzer.missCount, 1);
      analyzer.reset();
      expect(analyzer.missCount, 0);
      expect(analyzer.summary.missedShots, 0);
    });
  });

  group('ShotAnalyzer — real-world speed', () {
    test('a shot carries a physical km/h pace scaled by the table ruler', () {
      final analyzer = ShotAnalyzer();
      final shots = _run(analyzer, _arc([0.30, 0.60, 0.875, 0.95, 0.98]));

      final shot = shots.single;
      // Default full-frame geometry → 2.74 m per x-unit; km/h = units/s · m · 3.6.
      expect(shot.speedKmh, closeTo(shot.speed * 2.74 * 3.6, 1e-9));
      // Unlike the saturated normalized pace, the km/h is a legible, non-zero
      // radar-gun number (a hard drive is tens of km/h).
      expect(shot.speedKmh, greaterThan(20));
    });

    test('currentSpeedKmh exposes the latest per-frame reading, cleared on loss',
        () {
      final analyzer = ShotAnalyzer();
      // No reading before two frames establish a velocity.
      expect(analyzer.currentSpeedKmh, isNull);
      analyzer.onFrame(_frame(0, 0.30, 0.40));
      expect(analyzer.currentSpeedKmh, isNull);

      // A second frame 0.30 x-units / 33 ms later → a live km/h reading.
      analyzer.onFrame(_frame(33, 0.60, 0.40));
      final reading = analyzer.currentSpeedKmh;
      expect(reading, isNotNull);
      // 0.30 units / 0.033 s · 2.74 m · 3.6 ≈ 89 km/h.
      expect(reading!, closeTo(0.30 / (33 / 1000.0) * 2.74 * 3.6, 1e-6));

      // Losing the ball clears the stale reading so no ghost km/h lingers.
      for (final f in _gap(66)) {
        analyzer.onFrame(f);
      }
      expect(analyzer.currentSpeedKmh, isNull);
    });

    test('a narrower calibrated table span reads a slower km/h', () {
      // Same motion, but the table only spans half the frame → half the metres
      // per x-unit → half the km/h.
      final analyzer = ShotAnalyzer(
        config: const TrainingConfig(
          geometry: TableGeometry(left: 0.25, right: 0.75, netX: 0.5),
        ),
      );
      // Player-left target-right against a net at 0.5; land deep on the right.
      final shots = _run(analyzer, _arc([0.30, 0.60, 0.70, 0.72, 0.74]));

      final shot = shots.single;
      // right-left = 0.5 → metresPerUnit = 2.74 / 0.5 = 5.48.
      expect(shot.speedKmh, closeTo(shot.speed * 5.48 * 3.6, 1e-9));
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

    test('averages and consistency mine the lateral (across-table) axis', () {
      Shot shot(double lateral) => Shot(
            timestampMs: 0,
            speed: 1,
            depth: 0.5,
            lateral: lateral,
            score: 0.8,
          );
      final grouped = TrainingSummary([shot(0.5), shot(0.5), shot(0.5)]);
      expect(grouped.averageLateral, closeTo(0.5, 1e-9));
      expect(grouped.lateralConsistency, closeTo(1.0, 1e-9));

      final spread = TrainingSummary([shot(0.1), shot(0.5), shot(0.9)]);
      expect(spread.averageLateral, closeTo(0.5, 1e-9));
      expect(spread.lateralConsistency, lessThan(grouped.lateralConsistency));

      expect(grouped.report(), contains('Lateral consistency: 100%'));
    });

    test('tempo and rhythm consistency mine the shot timestamps', () {
      Shot at(int t) => Shot(
            timestampMs: t,
            speed: 1,
            depth: 0.5,
            lateral: 0.5,
            score: 0.8,
          );

      // Perfectly metronomic: a shot every 500 ms → 120 shots/min, 100% rhythm.
      final steady = TrainingSummary([at(0), at(500), at(1000), at(1500)]);
      expect(steady.shotIntervalsMs, [500, 500, 500]);
      expect(steady.averageIntervalMs, closeTo(500, 1e-9));
      expect(steady.shotsPerMinute, closeTo(120, 1e-9));
      expect(steady.rhythmConsistency, closeTo(1.0, 1e-9));
      expect(steady.report(), contains('Tempo: 120.0 shots/min'));
      expect(steady.report(), contains('Rhythm consistency: 100%'));

      // Irregular gaps (200/1000/300) have the same mean tempo but a much
      // lower rhythm consistency.
      final erratic = TrainingSummary([at(0), at(200), at(1200), at(1500)]);
      expect(erratic.averageIntervalMs, closeTo(500, 1e-9));
      expect(erratic.rhythmConsistency, lessThan(steady.rhythmConsistency));

      // A single shot establishes no interval, so tempo stats are neutral and
      // the tempo lines are omitted from the report.
      final single = TrainingSummary([at(0)]);
      expect(single.shotIntervalsMs, isEmpty);
      expect(single.shotsPerMinute, 0);
      expect(single.rhythmConsistency, 0);
      expect(single.report(), isNot(contains('Tempo:')));
    });

    test('empty session reports no shots', () {
      const summary = TrainingSummary([]);
      expect(summary.shotCount, 0);
      expect(summary.overallGrade, '–');
      expect(summary.consistency, 0);
      expect(summary.averageLateral, 0);
      expect(summary.lateralConsistency, 0);
      expect(summary.report(), contains('No shots recorded'));
    });

    test('report is human-readable and deterministic', () {
      final report = sessionLanding([0.875, 0.875]).summary.report();
      expect(report, contains('Training summary'));
      expect(report, contains('2 shots'));
      expect(report, contains('grade A'));
      expect(report, contains('km/h'));
    });

    test('missedShots set the on-table accuracy denominator and report line', () {
      const summary = TrainingSummary(
        [
          Shot(timestampMs: 0, speed: 1, depth: 0.7, score: 0.8),
          Shot(timestampMs: 100, speed: 1, depth: 0.7, score: 0.8),
          Shot(timestampMs: 200, speed: 1, depth: 0.7, score: 0.8),
        ],
        missedShots: 1,
      );
      expect(summary.attemptedShots, 4);
      expect(summary.onTableRate, closeTo(0.75, 1e-9));
      expect(summary.report(), contains('On-table accuracy: 75%'));
      expect(summary.report(), contains('3 of 4 on the table'));
    });

    test('the accuracy line is omitted when nothing was missed', () {
      const summary = TrainingSummary([
        Shot(timestampMs: 0, speed: 1, depth: 0.7, score: 0.8),
      ]);
      expect(summary.attemptedShots, 1);
      expect(summary.onTableRate, 1.0);
      expect(summary.report(), isNot(contains('On-table accuracy')));
    });

    test('the km/h line is omitted when no shot carries a real-world pace', () {
      // Directly-built shots default speedKmh to 0 (no calibrated scale).
      const summary = TrainingSummary([
        Shot(timestampMs: 0, speed: 1, depth: 0.5, score: 0.8),
      ]);
      expect(summary.maxSpeedKmh, 0);
      expect(summary.report(), isNot(contains('km/h')));
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
