/// Runnable offline benchmark: score the pong-ai scoring/perception pipeline
/// against the labeled clip corpus and print a consolidated report.
///
/// Usage:
///   dart run bin/benchmark.dart                 # score benchmark/clips/*.json
///   dart run bin/benchmark.dart path/a.json ... # score the given fixtures
///
/// Exits non-zero when no clips are found so it can gate CI.
library;

import 'dart:io';

import 'package:pong_ai/core/benchmark/benchmark_corpus.dart';

void main(List<String> args) {
  final clips =
      args.isEmpty ? loadClipDirectory() : loadClipFixtures(args);
  stdout.write(buildCorpusReport(clips));
  if (clips.isEmpty) exitCode = 1;
}
