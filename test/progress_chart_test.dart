import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/history/session_trends.dart';
import 'package:pong_ai/features/history/progress_chart.dart';

TrainingTrendPoint _point(String id, double score) => TrainingTrendPoint(
      id: id,
      savedAt: DateTime(2026),
      shotCount: 6,
      averageScore: score,
      overallGrade: 'B',
    );

void main() {
  group('progressChartPoints', () {
    test('spreads sessions evenly on x, oldest at left', () {
      final pts = progressChartPoints(const [0.5, 0.6, 0.7]);
      expect(pts, hasLength(3));
      expect(pts.first.x, 0.0);
      expect(pts[1].x, closeTo(0.5, 1e-9));
      expect(pts.last.x, 1.0);
    });

    test('maps a higher score to a higher point (smaller y)', () {
      final pts = progressChartPoints(const [0.2, 0.9]);
      // score 0.2 -> y 0.8 (low), score 0.9 -> y 0.1 (high)
      expect(pts.first.y, closeTo(0.8, 1e-9));
      expect(pts.last.y, closeTo(0.1, 1e-9));
      expect(pts.last.y, lessThan(pts.first.y));
    });

    test('a single session is centred', () {
      final pts = progressChartPoints(const [0.6]);
      expect(pts, hasLength(1));
      expect(pts.single.x, 0.5);
      expect(pts.single.y, closeTo(0.4, 1e-9));
    });

    test('clamps out-of-range scores', () {
      final pts = progressChartPoints(const [-0.5, 1.5]);
      expect(pts.first.y, 1.0); // clamped score 0 -> y 1
      expect(pts.last.y, 0.0); // clamped score 1 -> y 0
    });

    test('empty series yields no points', () {
      expect(progressChartPoints(const []), isEmpty);
    });
  });

  group('ProgressChartView', () {
    testWidgets('shows the empty-state hint with no sessions', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(body: ProgressChartView(sessions: [])),
        ),
      );
      expect(find.text('No training drills saved yet'), findsOneWidget);
    });

    testWidgets('renders a chart when sessions exist', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ProgressChartView(
              sessions: [_point('a', 0.5), _point('b', 0.8)],
            ),
          ),
        ),
      );
      expect(find.byType(CustomPaint), findsWidgets);
      expect(find.text('No training drills saved yet'), findsNothing);
    });
  });
}
