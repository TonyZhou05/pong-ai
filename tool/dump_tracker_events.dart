/// Dumps the raw BallTracker event stream for a footage fixture with the
/// footage-mode tuning, so referee rules can be designed against what the
/// tracker actually sees on real clips.
///
/// Usage: dart run tool/dump_tracker_events.dart <fixture.json>
library;

import 'dart:convert';
import 'dart:io';

import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/benchmark/clip_fixture.dart';

void main(List<String> args) {
  final fixture = ClipFixture.fromJson(
    jsonDecode(File(args.first).readAsStringSync()) as Map<String, dynamic>,
  );
  final tracker = BallTracker(
    geometry: fixture.geometry,
    maxGapFrames: 30,
    netBounceExclusion: 0.03,
  );
  for (final frame in fixture.frames) {
    for (final event in tracker.update(frame)) {
      final t = (event.timestampMs / 1000).toStringAsFixed(2);
      stdout.writeln('t=${t}s  $event');
    }
  }
}
