/// Tracker **event-detection** accuracy metrics for the benchmark harness.
///
/// The two existing benchmark stages bracket the analysis layer without
/// scoring it directly:
///
///   * [DetectionBenchmark] answers "did the *model* see the ball / players?"
///     (per-frame precision/recall) — pure perception.
///   * [BenchmarkRunner] answers "did the *whole* pipeline get the final score
///     right?" — end-to-end, so a bounce mis-detection and a scoring mistake
///     look the same.
///
/// The middle layer — does [BallTracker] fire a **bounce** (or net-crossing) at
/// the right instant? — was never scored on its own, yet it is the heart of the
/// app: every point is awarded off these events. This module fills that gap.
///
/// It replays a stream of [FrameResult]s through a real [BallTracker] and
/// matches the [TrackerEvent]s it emits against a list of ground-truth events,
/// pairing each emitted event to the nearest same-type ground-truth event
/// within a temporal tolerance window. From the true-positive / false-positive
/// / false-negative counts it derives per-type precision / recall / F1 and the
/// mean temporal error of matched events.
///
/// Ground truth comes from datasets like **OpenTTGames**, whose
/// `events_markup.json` labels the frame index of every bounce and net hit; see
/// `openTtGamesBounceEvents` in `openttgames_converter.dart` for the converter.
library;

import '../analysis/ball_tracker.dart';
import '../vision/detection.dart';

/// A tracker event category that can be benchmarked against ground truth.
///
/// Only [BounceEvent] and [NetCrossEvent] are *scored* here — a [BallLostEvent]
/// is an internal trajectory-management signal, not a physical court event, so
/// it is neither ground-truthable nor matched.
enum TrackedEventType { bounce, netCross }

/// One labeled court event at a known moment in time — the ground truth an
/// emitted [TrackerEvent] is scored against.
class GroundTruthEvent {
  const GroundTruthEvent(this.timestampMs, this.type);

  /// When the event happened, in the same millisecond clock as the frames.
  final int timestampMs;

  /// Which kind of event (a table bounce or a crossing of the net).
  final TrackedEventType type;

  @override
  String toString() => 'GTEvent($type @$timestampMs)';
}

/// Event-detection accuracy for a single [TrackedEventType], accumulated over a
/// clip.
class EventTypeMetrics {
  const EventTypeMetrics({
    required this.type,
    required this.truePositives,
    required this.falsePositives,
    required this.falseNegatives,
    required this.temporalErrorSum,
  });

  final TrackedEventType type;

  /// Emitted events matched to a ground-truth event within the tolerance.
  final int truePositives;

  /// Emitted events that matched no ground-truth event (spurious detections).
  final int falsePositives;

  /// Ground-truth events no emitted event matched (missed detections).
  final int falseNegatives;

  /// Sum of |emitted − ground-truth| ms over matched pairs, for [meanTemporalErrorMs].
  final double temporalErrorSum;

  int get groundTruthCount => truePositives + falseNegatives;
  int get predictedCount => truePositives + falsePositives;

  /// Fraction of emitted events that were real. Vacuously 1.0 when nothing was
  /// emitted (mirrors [BallDetectionMetrics.precision]).
  double get precision =>
      predictedCount == 0 ? 1.0 : truePositives / predictedCount;

  /// Fraction of ground-truth events the tracker caught. Vacuously 1.0 when
  /// there were no ground-truth events of this type.
  double get recall =>
      groundTruthCount == 0 ? 1.0 : truePositives / groundTruthCount;

  double get f1 {
    final p = precision;
    final r = recall;
    return (p + r) == 0 ? 0.0 : 2 * p * r / (p + r);
  }

  /// Mean absolute timing error (ms) of matched events; 0 when none matched.
  double get meanTemporalErrorMs =>
      truePositives == 0 ? 0.0 : temporalErrorSum / truePositives;
}

/// Per-clip tracker-event-detection metrics, one [EventTypeMetrics] per type.
class EventBenchmarkResult {
  const EventBenchmarkResult({
    required this.clipName,
    required this.frameCount,
    required this.bounce,
    required this.netCross,
  });

  final String clipName;
  final int frameCount;
  final EventTypeMetrics bounce;
  final EventTypeMetrics netCross;

  EventTypeMetrics forType(TrackedEventType type) =>
      type == TrackedEventType.bounce ? bounce : netCross;

  String report() {
    String line(String label, EventTypeMetrics m) =>
        '  $label  P/R/F1: ${_pct(m.precision)}/${_pct(m.recall)}/'
        '${_pct(m.f1)}  meanErr ${m.meanTemporalErrorMs.toStringAsFixed(0)}ms  '
        '(tp ${m.truePositives}, fp ${m.falsePositives}, '
        'fn ${m.falseNegatives})';
    return (StringBuffer()
          ..writeln('Events: $clipName  ($frameCount frame(s))')
          ..writeln(line('Bounce  ', bounce))
          ..writeln(line('NetCross', netCross)))
        .toString();
  }

  static String _pct(double v) => '${(v * 100).toStringAsFixed(1)}%';
}

/// Thresholds controlling how emitted events are matched to ground truth.
class EventBenchmarkConfig {
  const EventBenchmarkConfig({this.toleranceMs = 100});

  /// Max |emitted − ground-truth| ms for a match. The tracker reports a bounce
  /// one frame after the apex (the vertical-velocity sign flip), so at 30 fps a
  /// perfect bounce still lands ~33 ms late; a ~3-frame window absorbs that plus
  /// labeling jitter without letting an unrelated event steal the match.
  final int toleranceMs;
}

/// Replays frames through a [BallTracker] and scores its emitted events against
/// ground-truth event timings.
class EventDetectionBenchmark {
  const EventDetectionBenchmark({this.config = const EventBenchmarkConfig()});

  final EventBenchmarkConfig config;

  /// Run [frames] through a tracker built on [geometry] (or the supplied
  /// [tracker]) and score the emitted bounce / net-cross events against
  /// [groundTruth].
  ///
  /// Matching is greedy nearest-in-time per type: emitted events are matched in
  /// the order they fire, each claiming the closest still-unclaimed same-type
  /// ground-truth event within [EventBenchmarkConfig.toleranceMs]. Unmatched
  /// emissions are false positives; unclaimed ground-truth events are false
  /// negatives.
  EventBenchmarkResult evaluate({
    required String name,
    required List<FrameResult> frames,
    required List<GroundTruthEvent> groundTruth,
    TableGeometry geometry = const TableGeometry(),
    BallTracker? tracker,
  }) {
    final t = tracker ?? BallTracker(geometry: geometry);

    // Collect every scored emitted event with its type + timestamp.
    final emitted = <({TrackedEventType type, int timestampMs})>[];
    for (final frame in frames) {
      for (final ev in t.update(frame)) {
        switch (ev) {
          case BounceEvent():
            emitted.add((type: TrackedEventType.bounce, timestampMs: ev.timestampMs));
          case NetCrossEvent():
            emitted.add((type: TrackedEventType.netCross, timestampMs: ev.timestampMs));
          case BallLostEvent():
            break; // not a court event; not scored
        }
      }
    }

    return EventBenchmarkResult(
      clipName: name,
      frameCount: frames.length,
      bounce: _matchType(
        TrackedEventType.bounce,
        emitted,
        groundTruth,
      ),
      netCross: _matchType(
        TrackedEventType.netCross,
        emitted,
        groundTruth,
      ),
    );
  }

  /// Greedy nearest-in-time matching for a single event type.
  EventTypeMetrics _matchType(
    TrackedEventType type,
    List<({TrackedEventType type, int timestampMs})> emitted,
    List<GroundTruthEvent> groundTruth,
  ) {
    final preds = emitted
        .where((e) => e.type == type)
        .map((e) => e.timestampMs)
        .toList(growable: false);
    final truths = groundTruth
        .where((g) => g.type == type)
        .map((g) => g.timestampMs)
        .toList(growable: false);

    final claimed = List<bool>.filled(truths.length, false);
    var tp = 0;
    var errorSum = 0.0;
    for (final pred in preds) {
      var best = -1;
      var bestDelta = config.toleranceMs;
      for (var j = 0; j < truths.length; j++) {
        if (claimed[j]) continue;
        final delta = (pred - truths[j]).abs();
        if (delta <= bestDelta) {
          bestDelta = delta;
          best = j;
        }
      }
      if (best >= 0) {
        claimed[best] = true;
        tp++;
        errorSum += bestDelta;
      }
    }

    return EventTypeMetrics(
      type: type,
      truePositives: tp,
      falsePositives: preds.length - tp,
      falseNegatives: truths.length - tp,
      temporalErrorSum: errorSum,
    );
  }
}
