import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/training/shot_analyzer.dart';
import 'package:pong_ai/core/training/training_report_json.dart';
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

/// A down-up arc across [xs]; apex (bounce) lands on the 3rd sample.
List<FrameResult> _arc(List<double> xs, {int startT = 0, int step = 33}) {
  const ys = [0.30, 0.42, 0.50, 0.40, 0.30];
  return [
    for (var i = 0; i < xs.length; i++) _frame(startT + i * step, xs[i], ys[i]),
  ];
}

List<FrameResult> _gap(int startT, {int n = 8, int step = 33}) =>
    [for (var i = 0; i < n; i++) FrameResult(timestampMs: startT + i * step)];

TrainingSummary _run(List<FrameResult> frames) {
  final analyzer = ShotAnalyzer();
  for (final f in frames) {
    analyzer.onFrame(f);
  }
  return analyzer.summary;
}

void main() {
  test('structured report round-trips through JSON and mirrors the session', () {
    // Two clean strokes separated by a ball-loss gap.
    final summary = _run([
      ..._arc([0.30, 0.60, 0.875, 0.95, 0.98]),
      ..._gap(500),
      ..._arc([0.30, 0.60, 0.875, 0.95, 0.98], startT: 1000),
    ]);
    expect(summary.shotCount, 2);

    final jsonString = trainingReportJsonString(summary);
    final decoded = jsonDecode(jsonString) as Map<String, dynamic>;
    // Re-parses to an equal map — the export is genuinely machine-readable.
    expect(decoded, equals(buildTrainingReportJson(summary)));

    expect(decoded['schemaVersion'], trainingReportSchemaVersion);

    final session = decoded['session'] as Map<String, dynamic>;
    expect(session['shotCount'], summary.shotCount);
    expect(session['overallGrade'], summary.overallGrade);
    expect(session['longestOnTargetStreak'], summary.longestOnTargetStreak);

    // Both shots are serialized with a grade.
    final shots = decoded['shots'] as List;
    expect(shots, hasLength(2));
    expect((shots.first as Map)['grade'], isA<String>());

    // Two shots establish an interval, so tempo is populated.
    expect(decoded['tempo'], isNotNull);
    expect((decoded['tempo'] as Map)['shotsPerMinute'], greaterThan(0));

    // With shots recorded, the coaching section carries a prioritized focus.
    final coaching = decoded['coaching'] as Map<String, dynamic>;
    expect(coaching['focus'], isA<String>());
    expect(coaching['focusTip'], isA<String>());
    expect(coaching['strength'], isA<String>());
    // Two shots unlock the consistency/rhythm dimensions beyond placement.
    expect((coaching['dimensions'] as List).length, greaterThan(1));
  });

  test('empty session emits a well-formed, null-populated report', () {
    final json = buildTrainingReportJson(const TrainingSummary([]));

    expect((json['session'] as Map)['shotCount'], 0);
    expect(json['shots'], isEmpty);
    // No pace scale and <2 shots, so the optional sections are explicit nulls.
    expect((json['pace'] as Map)['maxSpeedKmh'], isNull);
    expect(json['tempo'], isNull);
    expect(json['coaching'], isNull);

    // Still valid JSON.
    expect(
      jsonDecode(trainingReportJsonString(const TrainingSummary([]))),
      isA<Map>(),
    );
  });

  test('config section reflects the drill target side', () {
    final json = buildTrainingReportJson(
      const TrainingSummary([]),
      config: const TrainingConfig(playerSide: TableSide.right),
    );
    final config = json['config'] as Map<String, dynamic>;
    expect(config['playerSide'], 'right');
    expect(config['targetSide'], 'left');
  });
}
