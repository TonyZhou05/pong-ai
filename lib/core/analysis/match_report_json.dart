/// Structured (JSON) post-match report.
///
/// [buildMatchReport] already folds every analytics layer into one *human*
/// readable string for the clipboard / share sheet. But a text blob can't be
/// re-parsed: it can't be stored as match history, diffed across sessions, or
/// fed to another tool or backend. The objective's "produce summary" goal is
/// better served if the same analytics are also available in a machine-readable
/// form.
///
/// [buildMatchReportJson] emits the full analytics off a [MatchController] as a
/// plain, JSON-encodable [Map] (a versioned schema), and [matchReportJsonString]
/// pretty-prints it. Like the rest of `core/`, it has no Flutter or vision
/// dependencies, so the export is deterministic and unit-testable end-to-end
/// from a synthetic frame stream — and it round-trips through `dart:convert`.
library;

import 'dart:convert';

import '../scoring/scoring_engine.dart';
import 'ball_tracker.dart';
import 'bounce_placement.dart';
import 'match_controller.dart';
import 'match_insights.dart';
import 'match_summary.dart';
import 'player_movement.dart';
import 'rally_analyzer.dart';

/// Bumped whenever the emitted structure changes in a backward-incompatible way,
/// so a stored report can be migrated or rejected by a future reader.
const int matchReportSchemaVersion = 1;

String _playerKey(Player p) => p == Player.a ? 'A' : 'B';

String _sideKey(TableSide s) => s == TableSide.left ? 'left' : 'right';

/// Round doubles to a stable number of decimals so the JSON is compact and
/// deterministic (no 0.30000000000000004 noise) while staying re-parseable.
double _round(double v, [int places = 3]) {
  final factor = <int, double>{2: 100, 3: 1000}[places] ?? 1000;
  return (v * factor).round() / factor;
}

Map<String, Object?> _playerSummaryJson(MatchSummary s, Player p) {
  final serveRate = s.serveWinRateFor(p);
  final json = <String, Object?>{
    'points': s.pointsWonBy(p),
    'forcedErrors': s.forcedErrorsWonBy(p),
    'openPlay': s.openPlayPointsWonBy(p),
    'longestStreak': s.longestStreakFor(p),
    'biggestLead': s.largestLeadBy(p),
    'comeback': s.largestDeficitOvercomeBy(p),
    'serve': serveRate == null
        ? null
        : {
            'played': s.servePointsPlayedBy(p),
            'won': s.servePointsWonBy(p),
            'winRate': _round(serveRate),
          },
    'gamePoints': s.hasPressureData
        ? {
            'held': s.gamePointsHeldBy(p),
            'converted': s.gamePointsConvertedBy(p),
            'faced': s.gamePointsFacedBy(p),
            'saved': s.gamePointsSavedBy(p),
            'conversionRate': () {
              final r = s.gamePointConversionRateFor(p);
              return r == null ? null : _round(r);
            }(),
          }
        : null,
  };
  return json;
}

Map<String, Object?> _rallyJson(RallyStats r) {
  final json = <String, Object?>{
    'count': r.rallyCount,
    'averageStrokes': _round(r.averageStrokes),
    'longestStrokes': r.longestStrokes,
    'averageDurationMs': _round(r.averageDurationMs),
    'short': r.shortRallies,
    'medium': r.mediumRallies,
    'long': r.longRallies,
  };
  if (r.hasWinData) {
    json['wins'] = {
      for (final p in Player.values)
        _playerKey(p): {
          'total': r.ralliesWonBy(p),
          'short': r.ralliesWonByLength(p, RallyLength.short),
          'medium': r.ralliesWonByLength(p, RallyLength.medium),
          'long': r.ralliesWonByLength(p, RallyLength.long),
        },
    };
  }
  return json;
}

Map<String, Object?>? _movementJson(PlayerMovementStats m) {
  if (!m.wasTracked) return null;
  return {
    'distanceTravelled': _round(m.distanceTravelled),
    'coverageWidth': _round(m.coverageWidth),
    'coverageDepth': _round(m.coverageDepth),
    'mobilityPerSecond': _round(m.mobilityPerSecond),
    'averageStanceWidth':
        m.averageStanceWidth == null ? null : _round(m.averageStanceWidth!),
  };
}

Map<String, Object?>? _coachingJson(PlayerInsights pi) {
  if (!pi.hasData) return null;
  return {
    'focus': pi.weakest!.name,
    'focusTip': pi.focusTip!,
    'strength': pi.strongest!.name,
    'dimensions': [
      for (final d in pi.dimensions) {'name': d.name, 'score': _round(d.score)},
    ],
  };
}

Map<String, Object?>? _placementJson(SidePlacementStats p) {
  if (p.count == 0) return null;
  return {
    'count': p.count,
    'averageDepth': _round(p.averageDepth),
    'depthConsistency': _round(p.depthConsistency),
    'lateralSpread': _round(p.lateralSpread),
    'short': p.shortCount,
    'middle': p.middleCount,
    'deep': p.deepCount,
  };
}

/// Compose the full match report from a [controller]'s live analytics as a
/// versioned, JSON-encodable map. Deterministic and Flutter-free.
Map<String, Object?> buildMatchReportJson(MatchController controller) {
  final summary = controller.summary;
  final state = controller.score;
  final winner = summary.matchWinner;
  final insights = MatchInsights(summary);

  return {
    'schemaVersion': matchReportSchemaVersion,
    'score': {
      'gamesA': state.gamesA,
      'gamesB': state.gamesB,
      'pointsA': state.pointsA,
      'pointsB': state.pointsB,
      'isMatchOver': state.isMatchOver,
      'winner': winner == null ? null : _playerKey(winner),
    },
    'summary': {
      'totalPoints': summary.totalPoints,
      'durationMs': summary.durationMs,
      'leadChanges': summary.leadChanges,
      'games': [
        for (final g in summary.gameScores) {'a': g.pointsA, 'b': g.pointsB},
      ],
      'players': {
        for (final p in Player.values)
          _playerKey(p): _playerSummaryJson(summary, p),
      },
    },
    'rallies': _rallyJson(controller.rallyStats),
    'ballSpeed': controller.hasBallSpeedData
        ? {
            'maxKmh': _round(controller.maxBallSpeedKmh, 1),
            'averageKmh': _round(controller.averageBallSpeedKmh, 1),
          }
        : null,
    'trackingQuality': controller.trackingQuality.hasData
        ? {
            'grade': controller.trackingQuality.grade,
            'score': _round(controller.trackingQuality.qualityScore),
            'ballDetectionRate':
                _round(controller.trackingQuality.ballDetectionRate),
            'averageBallConfidence':
                _round(controller.trackingQuality.averageBallConfidence),
            'twoPlayerRate': _round(controller.trackingQuality.twoPlayerRate),
          }
        : null,
    'movement': {
      for (final p in Player.values)
        _playerKey(p): _movementJson(controller.movementFor(p)),
    },
    'placement': {
      for (final side in TableSide.values)
        _sideKey(side): _placementJson(controller.placementFor(side)),
    },
    'coaching': insights.hasData
        ? {
            for (final p in Player.values)
              _playerKey(p): _coachingJson(insights.insightsFor(p)),
          }
        : null,
  };
}

/// The structured report pretty-printed as a JSON string, ready for the
/// clipboard, a share sheet, or a `.json` export file.
String matchReportJsonString(MatchController controller) =>
    const JsonEncoder.withIndent('  ').convert(buildMatchReportJson(controller));
