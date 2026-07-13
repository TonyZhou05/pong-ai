/// Offline evaluation of the scoring pipeline against labeled [ClipFixture]s.
///
/// This is the `benchmark/` harness promised in docs/ARCHITECTURE.md (roadmap
/// item 5). It replays a fixture's frames through the *exact same*
/// [BallTracker] → [RallyReferee] → [ScoringEngine] path the live camera uses
/// (via [MatchController]) and compares the auto-detected outcome to the clip's
/// ground truth, emitting scoring-accuracy metrics.
///
/// Keeping it pure-Dart (no camera, no plugin, no Flutter) means the whole
/// evaluation runs in unit tests and CI, so a model or logic change can be
/// scored objectively before it ever reaches a device.
library;

import '../analysis/ball_tracker.dart';
import '../analysis/match_controller.dart';
import '../analysis/rally_referee.dart';
import '../scoring/scoring_engine.dart';
import 'clip_fixture.dart';

/// The scoring-accuracy metrics for one clip.
class BenchmarkResult {
  const BenchmarkResult({
    required this.clipName,
    required this.source,
    required this.detectedPointsA,
    required this.detectedPointsB,
    required this.expectedPointsA,
    required this.expectedPointsB,
    required this.undeterminedCount,
    required this.orderedMatches,
    required this.orderedComparable,
  });

  final String clipName;
  final String source;

  /// Points the pipeline auto-attributed to each player.
  final int detectedPointsA;
  final int detectedPointsB;

  /// Ground-truth points for each player.
  final int expectedPointsA;
  final int expectedPointsB;

  /// Rallies the referee could not attribute (surfaced for manual resolution).
  /// These are *not* counted as detected points.
  final int undeterminedCount;

  /// Number of ordered point winners that matched ground truth, and how many
  /// positions were comparable — 0 when the fixture has no ordered labels.
  final int orderedMatches;
  final int orderedComparable;

  /// Total detected points across both players.
  int get detectedTotal => detectedPointsA + detectedPointsB;

  /// Total ground-truth points across both players.
  int get expectedTotal => expectedPointsA + expectedPointsB;

  /// L1 error between the detected and true per-player point totals.
  int get pointTotalError =>
      (detectedPointsA - expectedPointsA).abs() +
      (detectedPointsB - expectedPointsB).abs();

  /// Whether the aggregate per-player point totals exactly match ground truth.
  bool get finalScoreCorrect =>
      detectedPointsA == expectedPointsA &&
      detectedPointsB == expectedPointsB;

  /// Fraction of rallies given the correct winner in order, or null when the
  /// fixture carries no ordered labels.
  double? get orderedAccuracy =>
      orderedComparable == 0 ? null : orderedMatches / orderedComparable;

  /// Fraction of true points the pipeline recovered (recall on total volume),
  /// clamped to 1.0 when it over-counts.
  double get pointRecall {
    if (expectedTotal == 0) return detectedTotal == 0 ? 1.0 : 0.0;
    final recovered = detectedTotal.clamp(0, expectedTotal);
    return recovered / expectedTotal;
  }

  String report() {
    final buf = StringBuffer()
      ..writeln('Clip: $clipName  (source: $source)')
      ..writeln(
        '  Detected A-B: $detectedPointsA-$detectedPointsB   '
        'Truth A-B: $expectedPointsA-$expectedPointsB',
      )
      ..writeln(
        '  Final score correct: ${finalScoreCorrect ? 'YES' : 'NO'}   '
        'point-total error: $pointTotalError',
      )
      ..writeln('  Undetermined (needs manual call): $undeterminedCount');
    final acc = orderedAccuracy;
    if (acc != null) {
      buf.writeln(
        '  Ordered point accuracy: ${(acc * 100).toStringAsFixed(1)}% '
        '($orderedMatches/$orderedComparable)',
      );
    }
    return buf.toString();
  }
}

/// Runs [ClipFixture]s through the live scoring pipeline and scores accuracy.
class BenchmarkRunner {
  const BenchmarkRunner();

  /// Evaluate a single clip.
  BenchmarkResult run(ClipFixture clip) {
    final controller = MatchController(
      tracker: BallTracker(geometry: TableGeometry(netX: clip.netX)),
      referee: RallyReferee(leftPlayer: clip.leftPlayer),
      engine: ScoringEngine(
        firstServer: clip.firstServer,
        pointsPerGame: clip.pointsPerGame,
        bestOf: clip.bestOf,
      ),
    );

    for (final frame in clip.frames) {
      controller.onFrame(frame);
    }

    final winners = controller.points.map((p) => p.winner).toList();
    final detectedA = winners.where((w) => w == Player.a).length;
    final detectedB = winners.where((w) => w == Player.b).length;

    var orderedMatches = 0;
    var orderedComparable = 0;
    final truth = clip.groundTruth.pointWinners;
    if (truth != null) {
      final n = winners.length < truth.length ? winners.length : truth.length;
      // Count comparable positions as the longer of the two so missed or extra
      // detections are penalised, not silently ignored.
      orderedComparable =
          winners.length > truth.length ? winners.length : truth.length;
      for (var i = 0; i < n; i++) {
        if (winners[i] == truth[i]) orderedMatches++;
      }
    }

    return BenchmarkResult(
      clipName: clip.name,
      source: clip.source,
      detectedPointsA: detectedA,
      detectedPointsB: detectedB,
      expectedPointsA: clip.groundTruth.pointsA,
      expectedPointsB: clip.groundTruth.pointsB,
      undeterminedCount: controller.undetermined.length,
      orderedMatches: orderedMatches,
      orderedComparable: orderedComparable,
    );
  }

  /// Evaluate a set of clips and aggregate into a suite report.
  BenchmarkSuiteResult runAll(Iterable<ClipFixture> clips) =>
      BenchmarkSuiteResult(clips.map(run).toList(growable: false));
}

/// Aggregate metrics across a set of evaluated clips.
class BenchmarkSuiteResult {
  const BenchmarkSuiteResult(this.results);

  final List<BenchmarkResult> results;

  int get clipCount => results.length;

  /// Number of clips whose per-player point totals exactly matched.
  int get clipsExactlyCorrect =>
      results.where((r) => r.finalScoreCorrect).length;

  /// Mean of each clip's [BenchmarkResult.pointRecall].
  double get meanPointRecall {
    if (results.isEmpty) return 0;
    final sum = results.fold<double>(0, (a, r) => a + r.pointRecall);
    return sum / results.length;
  }

  /// Total rallies left undetermined across all clips.
  int get totalUndetermined =>
      results.fold(0, (a, r) => a + r.undeterminedCount);

  String report() {
    final buf = StringBuffer()
      ..writeln('=== Benchmark suite: $clipCount clip(s) ===');
    for (final r in results) {
      buf.write(r.report());
    }
    buf
      ..writeln('--- Aggregate ---')
      ..writeln(
        '  Clips with exact score: $clipsExactlyCorrect/$clipCount',
      )
      ..writeln(
        '  Mean point recall: ${(meanPointRecall * 100).toStringAsFixed(1)}%',
      )
      ..writeln('  Total undetermined rallies: $totalUndetermined');
    return buf.toString();
  }
}
