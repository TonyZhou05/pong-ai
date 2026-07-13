/// Runnable corpus evaluation: discover labeled [ClipFixture]s on disk and score
/// them through the offline benchmark stages in one pass.
///
/// The individual stages ([BenchmarkRunner] scoring accuracy,
/// [DetectionBenchmark] perception accuracy) already existed, but they could
/// only be driven from unit tests — there was no single entrypoint that loads
/// the whole `benchmark/clips/` corpus and prints a consolidated report. This
/// file is that entrypoint's reusable core: [loadClipFixtures] /
/// [loadClipDirectory] read fixtures from disk and [buildCorpusReport] composes
/// every stage's report for a set of clips.
///
/// It is pure Dart (only `dart:io` + `dart:convert`, no Flutter/plugin), so the
/// `bin/benchmark.dart` wrapper runs via `dart run` and this logic is unit
/// testable with `flutter test`.
library;

import 'dart:convert';
import 'dart:io';

import 'benchmark_runner.dart';
import 'clip_fixture.dart';
import 'detection_metrics.dart';

/// The default corpus directory, relative to the package root.
const String defaultClipDir = 'benchmark/clips';

/// Parse each JSON file at [paths] into a [ClipFixture].
///
/// Throws [FormatException] (with the offending path) if a file is not valid
/// JSON or not a fixture object, so a corrupt clip fails loudly rather than
/// being silently skipped.
List<ClipFixture> loadClipFixtures(Iterable<String> paths) {
  final clips = <ClipFixture>[];
  for (final path in paths) {
    final raw = File(path).readAsStringSync();
    final Object? decoded;
    try {
      decoded = jsonDecode(raw);
    } on FormatException catch (e) {
      throw FormatException('Invalid JSON in clip "$path": ${e.message}');
    }
    if (decoded is! Map<String, dynamic>) {
      throw FormatException('Clip "$path" is not a fixture object');
    }
    clips.add(ClipFixture.fromJson(decoded));
  }
  return clips;
}

/// Discover every `*.json` clip in [dir] (sorted by name) and load them.
///
/// Returns an empty list when the directory does not exist so callers can
/// report "no clips" rather than crash.
List<ClipFixture> loadClipDirectory([String dir = defaultClipDir]) {
  final directory = Directory(dir);
  if (!directory.existsSync()) return const [];
  final paths = directory
      .listSync()
      .whereType<File>()
      .map((f) => f.path)
      .where((p) => p.toLowerCase().endsWith('.json'))
      .toList()
    ..sort();
  return loadClipFixtures(paths);
}

/// Compose a consolidated report over [clips]: the scoring-accuracy suite for
/// every clip, followed by the perception (per-frame detection) stage for those
/// clips that carry `groundTruthFrames`.
String buildCorpusReport(
  List<ClipFixture> clips, {
  BenchmarkRunner runner = const BenchmarkRunner(),
  DetectionBenchmark detection = const DetectionBenchmark(),
}) {
  final buf = StringBuffer()
    ..writeln('######## pong-ai benchmark corpus ########')
    ..writeln('Clips loaded: ${clips.length}')
    ..writeln();
  if (clips.isEmpty) {
    buf.writeln('No clips found — add fixtures under $defaultClipDir/.');
    return buf.toString();
  }

  buf
    ..writeln('=== Stage 1: scoring accuracy ===')
    ..write(runner.runAll(clips).report())
    ..writeln();

  final perception = <DetectionBenchmarkResult>[];
  for (final clip in clips) {
    final result = detection.evaluateClip(clip);
    if (result != null) perception.add(result);
  }

  buf.writeln('=== Stage 2: perception accuracy ===');
  if (perception.isEmpty) {
    buf.writeln(
      'No clips carry per-frame ground truth (groundTruthFrames) to score.',
    );
  } else {
    for (final r in perception) {
      buf.write(r.report());
    }
  }
  return buf.toString();
}
