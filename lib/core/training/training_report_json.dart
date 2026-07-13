/// Structured (JSON) training-session report.
///
/// [TrainingSummary.report] already folds a drill's shots into one *human*
/// readable string for the clipboard / share sheet, and the match pipeline
/// gained a machine-readable companion in [buildMatchReportJson]. Training mode
/// had no such structured export: a coach who wants to store a drill as history,
/// diff pace/placement/rhythm across sessions, or feed the numbers to another
/// tool could only screen-scrape the text blob.
///
/// [buildTrainingReportJson] emits the full session analytics off a
/// [TrainingSummary] (plus the [TrainingConfig] the drill targeted) as a plain,
/// JSON-encodable [Map] with a versioned schema, and [trainingReportJsonString]
/// pretty-prints it. Like the rest of `core/`, it has no Flutter or vision
/// dependencies, so the export is deterministic and unit-testable end-to-end
/// from a synthetic frame stream — and it round-trips through `dart:convert`.
library;

import 'dart:convert';

import '../analysis/ball_tracker.dart';
import 'shot_analyzer.dart';
import 'training_feedback.dart';

/// Bumped whenever the emitted structure changes in a backward-incompatible way,
/// so a stored report can be migrated or rejected by a future reader.
const int trainingReportSchemaVersion = 1;

String _sideKey(TableSide s) => s == TableSide.left ? 'left' : 'right';

/// Round doubles to a stable number of decimals so the JSON is compact and
/// deterministic (no 0.30000000000000004 noise) while staying re-parseable.
double _round(double v, [int places = 3]) {
  final factor = <int, double>{1: 10, 2: 100, 3: 1000}[places] ?? 1000;
  return (v * factor).round() / factor;
}

Map<String, Object?> _shotJson(Shot s) => {
      'timestampMs': s.timestampMs,
      'depth': _round(s.depth),
      'lateral': _round(s.lateral),
      'speed': _round(s.speed),
      'speedKmh': _round(s.speedKmh, 1),
      'score': _round(s.score),
      'grade': s.grade.name,
    };

/// Compose the full training report from a [summary] (and the [config] the drill
/// aimed at) as a versioned, JSON-encodable map. Deterministic and Flutter-free.
Map<String, Object?> buildTrainingReportJson(
  TrainingSummary summary, {
  TrainingConfig config = const TrainingConfig(),
}) {
  final hasShots = summary.shots.isNotEmpty;
  final feedback = TrainingFeedback(summary, config: config);
  return {
    'schemaVersion': trainingReportSchemaVersion,
    'config': {
      'playerSide': _sideKey(config.playerSide),
      'targetSide': _sideKey(config.targetSide),
      'targetDepth': _round(config.targetDepth),
      'depthTolerance': _round(config.depthTolerance),
    },
    'session': {
      'shotCount': summary.shotCount,
      'overallGrade': summary.overallGrade,
      'averageScore': _round(summary.averageScore),
      'durationMs': summary.durationMs,
    },
    'placement': {
      'averageDepth': _round(summary.averageDepth),
      'depthConsistency': _round(summary.consistency),
      'averageLateral': _round(summary.averageLateral),
      'lateralConsistency': _round(summary.lateralConsistency),
    },
    'pace': {
      'averageSpeed': _round(summary.averageSpeed),
      'maxSpeedKmh': summary.maxSpeedKmh > 0 ? _round(summary.maxSpeedKmh, 1) : null,
      'averageSpeedKmh':
          summary.maxSpeedKmh > 0 ? _round(summary.averageSpeedKmh, 1) : null,
    },
    // Tempo needs at least two shots to establish an interval.
    'tempo': summary.shotCount >= 2
        ? {
            'shotsPerMinute': _round(summary.shotsPerMinute, 1),
            'averageIntervalMs': _round(summary.averageIntervalMs, 1),
            'rhythmConsistency': _round(summary.rhythmConsistency),
          }
        : null,
    'grades': {
      'excellent': summary.gradeCount(ShotGrade.excellent),
      'good': summary.gradeCount(ShotGrade.good),
      'fair': summary.gradeCount(ShotGrade.fair),
      'poor': summary.gradeCount(ShotGrade.poor),
    },
    'coaching': feedback.hasData
        ? {
            'focus': feedback.weakest!.name,
            'focusTip': feedback.focusTip!,
            'strength': feedback.strongest!.name,
            'dimensions': [
              for (final d in feedback.dimensions)
                {'name': d.name, 'score': _round(d.score)},
            ],
          }
        : null,
    'shots': hasShots ? [for (final s in summary.shots) _shotJson(s)] : const [],
  };
}

/// The structured report pretty-printed as a JSON string, ready for the
/// clipboard, a share sheet, or a `.json` export file.
String trainingReportJsonString(
  TrainingSummary summary, {
  TrainingConfig config = const TrainingConfig(),
}) =>
    const JsonEncoder.withIndent('  ')
        .convert(buildTrainingReportJson(summary, config: config));
