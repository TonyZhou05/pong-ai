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
      },
      'placement': {'depthConsistency': depthConsistency},
      'pace': {'maxSpeedKmh': maxSpeedKmh},
      'tempo': {'rhythmConsistency': rhythmConsistency},
    },
  );
}

StoredSession _match(String id, DateTime at) => StoredSession(
      id: id,
      kind: SessionKind.match,
      savedAt: at,
      report: const {
        'score': {'gamesA': 3, 'gamesB': 1},
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
}
