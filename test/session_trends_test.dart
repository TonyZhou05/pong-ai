import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/history/session_history_store.dart';
import 'package:pong_ai/core/history/session_trends.dart';

/// Build a stored training session with the fields SessionTrends reads.
StoredSession _training(
  String id,
  DateTime at, {
  required double averageScore,
  required String grade,
  int shots = 6,
  double? depthConsistency,
  double? maxSpeedKmh,
  double? rhythmConsistency,
  String? focus,
  double? onTableRate,
  int? longestOnTargetStreak,
}) {
  return StoredSession(
    id: id,
    kind: SessionKind.training,
    savedAt: at,
    report: {
      'session': {
        'shotCount': shots,
        'overallGrade': grade,
        'averageScore': averageScore,
        if (onTableRate != null) 'onTableRate': onTableRate,
        if (longestOnTargetStreak != null)
          'longestOnTargetStreak': longestOnTargetStreak,
      },
      'placement': {'depthConsistency': depthConsistency},
      'pace': {'maxSpeedKmh': maxSpeedKmh},
      'tempo': {'rhythmConsistency': rhythmConsistency},
      if (focus != null) 'coaching': {'focus': focus},
    },
  );
}

StoredSession _match(
  String id,
  DateTime at, {
  int? totalPoints,
  int? longestStrokes,
  double? averageStrokes,
  double? maxKmh,
  double? averageKmh,
  String? winner,
  int? durationMs,
}) =>
    StoredSession(
      id: id,
      kind: SessionKind.match,
      savedAt: at,
      report: {
        'score': {'gamesA': 3, 'gamesB': 1, 'winner': winner},
        if (totalPoints != null || durationMs != null)
          'summary': {
            if (totalPoints != null) 'totalPoints': totalPoints,
            if (durationMs != null) 'durationMs': durationMs,
          },
        if (longestStrokes != null || averageStrokes != null)
          'rallies': {
            if (longestStrokes != null) 'longestStrokes': longestStrokes,
            if (averageStrokes != null) 'averageStrokes': averageStrokes,
          },
        if (maxKmh != null || averageKmh != null)
          'ballSpeed': {
            if (maxKmh != null) 'maxKmh': maxKmh,
            if (averageKmh != null) 'averageKmh': averageKmh,
          },
      },
    );

void main() {
  final t0 = DateTime(2026, 7, 1, 9);
  final t1 = DateTime(2026, 7, 5, 9);
  final t2 = DateTime(2026, 7, 10, 9);

  group('TrainingTrendPoint.fromStored', () {
    test('parses the headline metrics out of a training report', () {
      final point = TrainingTrendPoint.fromStored(
        _training(
          'training-1',
          t0,
          averageScore: 0.72,
          grade: 'B',
          shots: 8,
          depthConsistency: 0.05,
          maxSpeedKmh: 81.2,
          rhythmConsistency: 0.9,
        ),
      );
      expect(point, isNotNull);
      expect(point!.averageScore, closeTo(0.72, 1e-9));
      expect(point.overallGrade, 'B');
      expect(point.shotCount, 8);
      expect(point.depthConsistency, closeTo(0.05, 1e-9));
      expect(point.maxSpeedKmh, closeTo(81.2, 1e-9));
      expect(point.rhythmConsistency, closeTo(0.9, 1e-9));
    });

    test('returns null for a match session or malformed report', () {
      expect(TrainingTrendPoint.fromStored(_match('m', t0)), isNull);
      final malformed = StoredSession(
        id: 'x',
        kind: SessionKind.training,
        savedAt: t0,
        report: const {'session': 'not-a-map'},
      );
      expect(TrainingTrendPoint.fromStored(malformed), isNull);
    });
  });

  group('SessionTrends', () {
    test('sorts training sessions oldest-first regardless of input order', () {
      final trends = SessionTrends.fromSessions([
        _training('c', t2, averageScore: 0.8, grade: 'A'),
        _training('a', t0, averageScore: 0.5, grade: 'C'),
        _training('b', t1, averageScore: 0.6, grade: 'B'),
      ]);
      expect(
        trends.trainingSessions.map((p) => p.id),
        ['a', 'b', 'c'],
      );
      expect(trends.firstSession!.id, 'a');
      expect(trends.latestSession!.id, 'c');
    });

    test('counts matches separately and excludes them from training', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C'),
        _match('m1', t1),
        _match('m2', t2),
      ]);
      expect(trends.matchCount, 2);
      expect(trends.trainingCount, 1);
      expect(trends.hasTrainingTrend, isFalse);
    });

    test('computes score improvement first -> latest', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.50, grade: 'C'),
        _training('b', t2, averageScore: 0.74, grade: 'B'),
      ]);
      expect(trends.hasTrainingTrend, isTrue);
      expect(trends.scoreImprovement, closeTo(0.24, 1e-9));
      expect(trends.meanScore, closeTo(0.62, 1e-9));
    });

    test('scoreImprovement is null with fewer than two sessions', () {
      expect(SessionTrends.fromSessions(const []).scoreImprovement, isNull);
      expect(
        SessionTrends.fromSessions(
          [_training('a', t0, averageScore: 0.5, grade: 'C')],
        ).scoreImprovement,
        isNull,
      );
    });

    test('totalShotsPracticed sums shot counts across drills', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', shots: 6),
        _training('b', t1, averageScore: 0.9, grade: 'A', shots: 10),
        _training('c', t2, averageScore: 0.6, grade: 'B', shots: 8),
      ]);
      expect(trends.totalShotsPracticed, 24);
    });

    test('totalShotsPracticed is 0 with no training sessions', () {
      expect(SessionTrends.fromSessions(const []).totalShotsPracticed, 0);
    });

    test('report includes total shots practiced', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', shots: 6),
        _training('b', t2, averageScore: 0.7, grade: 'B', shots: 9),
      ]);
      expect(trends.report(), contains('Total shots practiced: 15'));
    });

    test('best session picks the highest average score', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C'),
        _training('b', t1, averageScore: 0.9, grade: 'A', shots: 10),
        _training('c', t2, averageScore: 0.6, grade: 'B'),
      ]);
      expect(trends.bestSession!.id, 'b');
      expect(trends.bestSession!.overallGrade, 'A');
    });

    test('bestMaxSpeedKmh takes the fastest across sessions, null if none', () {
      final withSpeed = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', maxSpeedKmh: 70.0),
        _training('b', t1, averageScore: 0.6, grade: 'B', maxSpeedKmh: 88.5),
        _training('c', t2, averageScore: 0.6, grade: 'B'),
      ]);
      expect(withSpeed.bestMaxSpeedKmh, closeTo(88.5, 1e-9));

      final noSpeed = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C'),
      ]);
      expect(noSpeed.bestMaxSpeedKmh, isNull);
    });

    test('depthConsistencyImprovement is first minus latest (tighter=+)', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', depthConsistency: 0.08),
        _training('b', t1, averageScore: 0.6, grade: 'B'),
        _training('c', t2, averageScore: 0.7, grade: 'A', depthConsistency: 0.03),
      ]);
      // Skips the session with no depth metric; first 0.08 → latest 0.03.
      expect(trends.depthConsistencyImprovement, closeTo(0.05, 1e-9));
    });

    test('rhythmConsistencyImprovement is latest minus first (steadier=+)', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', rhythmConsistency: 0.6),
        _training('b', t2, averageScore: 0.7, grade: 'A', rhythmConsistency: 0.9),
      ]);
      expect(trends.rhythmConsistencyImprovement, closeTo(0.3, 1e-9));
    });

    test('consistency improvements are null without two recording sessions', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', depthConsistency: 0.05),
        _training('b', t2, averageScore: 0.7, grade: 'A'),
      ]);
      expect(trends.depthConsistencyImprovement, isNull);
      expect(trends.rhythmConsistencyImprovement, isNull);
    });

    test('speedImprovement is latest minus first km/h, skipping unscaled', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', maxSpeedKmh: 70.0),
        _training('b', t1, averageScore: 0.6, grade: 'B'),
        _training('c', t2, averageScore: 0.7, grade: 'A', maxSpeedKmh: 88.5),
      ]);
      // Skips the middle session with no km/h; first 70.0 → latest 88.5.
      expect(trends.speedImprovement, closeTo(18.5, 1e-9));
    });

    test('speedImprovement is null without two scaled sessions', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', maxSpeedKmh: 72.0),
        _training('b', t2, averageScore: 0.7, grade: 'A'),
      ]);
      expect(trends.speedImprovement, isNull);
    });

    test('report reflects the improvement trend and best session', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.50, grade: 'C'),
        _training('b', t1, averageScore: 0.62, grade: 'B'),
        _training('c', t2, averageScore: 0.80, grade: 'A', maxSpeedKmh: 90.0),
        _match('m1', t2),
      ]);
      final report = trends.report();
      expect(report, contains('3 training'));
      expect(report, contains('1 match'));
      expect(report, contains('50% → 80%'));
      expect(report, contains('up +30%'));
      expect(report, contains('Best session: grade A'));
      expect(report, contains('90.0 km/h'));
    });

    test('report surfaces placement and rhythm consistency trends', () {
      final trends = SessionTrends.fromSessions([
        _training(
          'a',
          t0,
          averageScore: 0.50,
          grade: 'C',
          depthConsistency: 0.09,
          rhythmConsistency: 0.60,
        ),
        _training(
          'b',
          t2,
          averageScore: 0.80,
          grade: 'A',
          depthConsistency: 0.04,
          rhythmConsistency: 0.85,
        ),
      ]);
      final report = trends.report();
      expect(report, contains('Placement consistency: tighter'));
      expect(report, contains('Rhythm consistency: up +25%'));
    });

    test('report surfaces the shot-speed trend', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.50, grade: 'C', maxSpeedKmh: 68.0),
        _training('b', t2, averageScore: 0.80, grade: 'A', maxSpeedKmh: 84.0),
      ]);
      expect(trends.report(), contains('Shot speed: up 16.0 km/h'));
    });

    test('report handles an empty history', () {
      final report = SessionTrends.fromSessions(const []).report();
      expect(report, contains('No training drills saved yet.'));
    });
  });

  group('SessionTrends on-table accuracy', () {
    test('parses onTableRate out of the session block', () {
      final point = TrainingTrendPoint.fromStored(
        _training('a', t0, averageScore: 0.7, grade: 'B', onTableRate: 0.75),
      );
      expect(point!.onTableRate, closeTo(0.75, 1e-9));
    });

    test('accuracyImprovement is latest minus first, skipping untracked', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', onTableRate: 0.60),
        _training('b', t1, averageScore: 0.6, grade: 'B'),
        _training('c', t2, averageScore: 0.7, grade: 'A', onTableRate: 0.90),
      ]);
      // Skips the middle session with no accuracy; first 0.60 → latest 0.90.
      expect(trends.accuracyImprovement, closeTo(0.30, 1e-9));
      expect(trends.bestOnTableRate, closeTo(0.90, 1e-9));
    });

    test('accuracyImprovement is null without two tracked sessions', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', onTableRate: 0.7),
        _training('b', t2, averageScore: 0.7, grade: 'A'),
      ]);
      expect(trends.accuracyImprovement, isNull);
      expect(trends.bestOnTableRate, closeTo(0.7, 1e-9));
    });

    test('report surfaces the on-table accuracy trend', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.50, grade: 'C', onTableRate: 0.55),
        _training('b', t2, averageScore: 0.80, grade: 'A', onTableRate: 0.85),
      ]);
      expect(trends.report(), contains('On-table accuracy: up +30%'));
    });
  });

  group('SessionTrends on-target streak', () {
    test('parses longestOnTargetStreak out of the session block', () {
      final point = TrainingTrendPoint.fromStored(
        _training(
          'a',
          t0,
          averageScore: 0.7,
          grade: 'B',
          longestOnTargetStreak: 4,
        ),
      );
      expect(point!.longestOnTargetStreak, 4);
    });

    test('bestOnTargetStreak is the max across tracked sessions', () {
      final trends = SessionTrends.fromSessions([
        _training(
          'a',
          t0,
          averageScore: 0.5,
          grade: 'C',
          longestOnTargetStreak: 3,
        ),
        _training('b', t1, averageScore: 0.6, grade: 'B'),
        _training(
          'c',
          t2,
          averageScore: 0.7,
          grade: 'A',
          longestOnTargetStreak: 5,
        ),
      ]);
      expect(trends.bestOnTargetStreak, 5);
    });

    test('bestOnTargetStreak is null when no session tracked it', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C'),
      ]);
      expect(trends.bestOnTargetStreak, isNull);
    });

    test('report surfaces a best on-target streak of at least two', () {
      final trends = SessionTrends.fromSessions([
        _training(
          'a',
          t0,
          averageScore: 0.5,
          grade: 'C',
          longestOnTargetStreak: 2,
        ),
        _training(
          'b',
          t2,
          averageScore: 0.8,
          grade: 'A',
          longestOnTargetStreak: 6,
        ),
      ]);
      expect(trends.report(), contains('Best on-target streak: 6 in a row'));
    });
  });

  group('SessionTrends recurring focus', () {
    final t3 = DateTime(2026, 7, 15, 9);

    test('parses coaching focus out of a training report', () {
      final point = TrainingTrendPoint.fromStored(
        _training('a', t0, averageScore: 0.6, grade: 'B', focus: 'Rhythm'),
      );
      expect(point!.focusArea, 'Rhythm');
    });

    test('leaves focusArea null when no coaching section was recorded', () {
      final point = TrainingTrendPoint.fromStored(
        _training('a', t0, averageScore: 0.6, grade: 'B'),
      );
      expect(point!.focusArea, isNull);
    });

    test('tallies the most common focus and its count', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', focus: 'Placement accuracy'),
        _training('b', t1, averageScore: 0.6, grade: 'B', focus: 'Rhythm'),
        _training('c', t2, averageScore: 0.7, grade: 'B', focus: 'Placement accuracy'),
      ]);
      expect(trends.focusCounts['Placement accuracy'], 2);
      expect(trends.focusCounts['Rhythm'], 1);
      expect(trends.recurringFocus, 'Placement accuracy');
      expect(trends.recurringFocusCount, 2);
      expect(trends.hasRecurringFocus, isTrue);
    });

    test('breaks count ties toward the more recent focus', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', focus: 'Rhythm'),
        _training('b', t1, averageScore: 0.6, grade: 'B', focus: 'Placement accuracy'),
        _training('c', t2, averageScore: 0.7, grade: 'B', focus: 'Rhythm'),
        _training('d', t3, averageScore: 0.8, grade: 'A', focus: 'Placement accuracy'),
      ]);
      // Both appear twice; Placement accuracy's latest session (d) is newest.
      expect(trends.recurringFocus, 'Placement accuracy');
      expect(trends.recurringFocusCount, 2);
    });

    test('a one-off focus does not count as recurring', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', focus: 'Rhythm'),
        _training('b', t1, averageScore: 0.6, grade: 'B', focus: 'Placement accuracy'),
      ]);
      expect(trends.hasRecurringFocus, isFalse);
      expect(trends.recurringFocusCount, 1);
    });

    test('report surfaces a recurring focus', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C', focus: 'Rhythm'),
        _training('b', t1, averageScore: 0.6, grade: 'B', focus: 'Rhythm'),
      ]);
      expect(trends.report(), contains('Recurring focus: Rhythm (2 of 2 drills)'));
    });
  });

  group('SessionTrends match career totals', () {
    test('parses cumulative metrics out of a match report', () {
      final point = MatchTrendPoint.fromStored(
        _match(
          'm1',
          t0,
          totalPoints: 21,
          longestStrokes: 8,
          maxKmh: 74.5,
          winner: 'A',
        ),
      );
      expect(point, isNotNull);
      expect(point!.totalPoints, 21);
      expect(point.longestRallyStrokes, 8);
      expect(point.maxBallSpeedKmh, closeTo(74.5, 1e-9));
      expect(point.winner, 'A');
    });

    test('MatchTrendPoint.fromStored returns null for a training session', () {
      final training = _training('t', t0, averageScore: 0.5, grade: 'C');
      expect(MatchTrendPoint.fromStored(training), isNull);
    });

    test('a sparse match (score only) still parses with null metrics', () {
      final point = MatchTrendPoint.fromStored(_match('m', t0));
      expect(point, isNotNull);
      expect(point!.totalPoints, isNull);
      expect(point.maxBallSpeedKmh, isNull);
    });

    test('aggregates points, longest rally and fastest ball across matches', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, totalPoints: 19, longestStrokes: 5, maxKmh: 70.0),
        _match('m2', t1, totalPoints: 25, longestStrokes: 11, maxKmh: 88.2),
        _match('m3', t2, totalPoints: 21), // no rally/speed recorded
      ]);
      expect(trends.matchCount, 3);
      expect(trends.hasMatchData, isTrue);
      expect(trends.totalMatchPoints, 19 + 25 + 21);
      expect(trends.longestMatchRallyStrokes, 11);
      expect(trends.fastestMatchBallSpeedKmh, closeTo(88.2, 1e-9));
    });

    test('aggregates are null when no match recorded them', () {
      final trends = SessionTrends.fromSessions([_match('m', t0)]);
      expect(trends.totalMatchPoints, isNull);
      expect(trends.longestMatchRallyStrokes, isNull);
      expect(trends.fastestMatchBallSpeedKmh, isNull);
      expect(trends.totalMatchDurationMs, isNull);
    });

    test('sums total play time across matches that recorded a duration', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, durationMs: 125000),
        _match('m2', t1, durationMs: 90000),
        _match('m3', t2), // no duration recorded
      ]);
      expect(trends.totalMatchDurationMs, 125000 + 90000);
    });

    test('total play time is null when no match recorded a duration', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, totalPoints: 19),
      ]);
      expect(trends.totalMatchDurationMs, isNull);
    });

    test('no match data leaves the aggregates empty', () {
      final trends = SessionTrends.fromSessions([
        _training('a', t0, averageScore: 0.5, grade: 'C'),
      ]);
      expect(trends.hasMatchData, isFalse);
      expect(trends.matchCount, 0);
    });

    test('report surfaces the match career section', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, totalPoints: 19, longestStrokes: 5, maxKmh: 70.0),
        _match('m2', t1, totalPoints: 25, longestStrokes: 11, maxKmh: 88.2),
      ]);
      final report = trends.report();
      expect(report, contains('Matches: 2 played'));
      expect(report, contains('Points contested: 44'));
      expect(report, contains('Longest rally: 11 strokes'));
      expect(report, contains('Fastest ball: 88.2 km/h'));
    });

    test('report surfaces cumulative play time', () {
      final report = SessionTrends.fromSessions([
        _match('m1', t0, durationMs: 125000),
        _match('m2', t1, durationMs: 90000),
      ]).report();
      // 215000 ms = 3m 35s
      expect(report, contains('Total play time: 3m 35s'));
    });

    test('report shows the match section even with no training drills', () {
      final report = SessionTrends.fromSessions([_match('m', t0)]).report();
      expect(report, contains('No training drills saved yet.'));
      expect(report, contains('Matches: 1 played'));
    });

    test('averages the per-match typical rally length across matches', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, averageStrokes: 3.0),
        _match('m2', t1, averageStrokes: 6.0),
        _match('m3', t2), // no rally data recorded
      ]);
      // Unweighted mean of the two matches that recorded a rally average.
      expect(trends.averageMatchRallyStrokes, closeTo(4.5, 1e-9));
    });

    test('averageMatchRallyStrokes is null when no match recorded rallies', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, totalPoints: 12),
        _match('m2', t1, winner: 'A'),
      ]);
      expect(trends.averageMatchRallyStrokes, isNull);
    });

    test('report surfaces the typical rally length', () {
      final report = SessionTrends.fromSessions([
        _match('m1', t0, averageStrokes: 4.0),
        _match('m2', t1, averageStrokes: 5.0),
      ]).report();
      expect(report, contains('Average rally: 4.5 strokes'));
    });

    test('averages the per-match typical ball speed across matches', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, averageKmh: 30.0),
        _match('m2', t1, averageKmh: 40.0),
        _match('m3', t2), // no ball speed recorded
      ]);
      // Unweighted mean of the two matches that recorded an average speed.
      expect(trends.averageMatchBallSpeedKmh, closeTo(35.0, 1e-9));
    });

    test('averageMatchBallSpeedKmh is null when no match tracked speed', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, totalPoints: 12),
        _match('m2', t1, winner: 'A'),
      ]);
      expect(trends.averageMatchBallSpeedKmh, isNull);
    });

    test('report surfaces the typical ball speed', () {
      final report = SessionTrends.fromSessions([
        _match('m1', t0, maxKmh: 80.0, averageKmh: 40.0),
        _match('m2', t1, maxKmh: 88.0, averageKmh: 50.0),
      ]).report();
      expect(report, contains('Fastest ball: 88.0 km/h'));
      expect(report, contains('Average ball: 45.0 km/h'));
    });
  });

  group('SessionTrends match win record', () {
    test('tallies A vs B wins across finished matches', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, winner: 'A'),
        _match('m2', t1, winner: 'B'),
        _match('m3', t2, winner: 'A'),
      ]);
      expect(trends.hasMatchWinRecord, isTrue);
      expect(trends.decidedMatchCount, 3);
      expect(trends.matchWinsBy('A'), 2);
      expect(trends.matchWinsBy('B'), 1);
    });

    test('ignores matches with no recorded winner', () {
      final trends = SessionTrends.fromSessions([
        _match('m1', t0, winner: 'A'),
        _match('m2', t1), // unfinished / pre-winner-field
      ]);
      expect(trends.decidedMatchCount, 1);
      expect(trends.matchWinsBy('A'), 1);
      expect(trends.matchWinsBy('B'), 0);
    });

    test('no win record when no match finished', () {
      final trends = SessionTrends.fromSessions([_match('m', t0)]);
      expect(trends.hasMatchWinRecord, isFalse);
      expect(trends.decidedMatchCount, 0);
    });

    test('report surfaces the head-to-head line', () {
      final report = SessionTrends.fromSessions([
        _match('m1', t0, winner: 'A'),
        _match('m2', t1, winner: 'A'),
        _match('m3', t2, winner: 'B'),
      ]).report();
      expect(report, contains('Head-to-head: A 2–1 B'));
    });

    test('report omits the head-to-head line for unfinished matches', () {
      final report = SessionTrends.fromSessions([_match('m', t0)]).report();
      expect(report, isNot(contains('Head-to-head')));
    });
  });
}
