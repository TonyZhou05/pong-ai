import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_summary.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/features/summary/momentum_chart.dart';

ScoredPoint _pt(Player winner, int t) => ScoredPoint(
      winner: winner,
      reason: PointReason.notReturned,
      timestampMs: t,
    );

void main() {
  group('momentumSeries', () {
    test('an empty point log is just the 0-0 start', () {
      expect(momentumSeries([]), [0]);
    });

    test('leads accumulate as +1 for A and -1 for B', () {
      final series = momentumSeries([
        _pt(Player.a, 0),
        _pt(Player.a, 1),
        _pt(Player.b, 2),
      ]);
      // start, +A, +A, -B
      expect(series, [0, 1, 2, 1]);
    });

    test('the curve can dip below zero when B pulls ahead', () {
      final series = momentumSeries([
        _pt(Player.b, 0),
        _pt(Player.b, 1),
        _pt(Player.a, 2),
      ]);
      expect(series, [0, -1, -2, -1]);
      expect(series.last, -1); // B still leads
    });

    test('the series always has one more entry than the point log', () {
      final points = [_pt(Player.a, 0), _pt(Player.b, 1)];
      expect(momentumSeries(points).length, points.length + 1);
    });
  });

  group('MomentumChartView', () {
    testWidgets('shows a hint when no points have been played', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(body: MomentumChartView(points: [])),
        ),
      );

      expect(find.text('No points played yet'), findsOneWidget);
      expect(find.byType(CustomPaint), findsWidgets);
    });

    testWidgets('drops the hint and paints once points exist', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: MomentumChartView(
              points: [_pt(Player.a, 0), _pt(Player.b, 1), _pt(Player.a, 2)],
            ),
          ),
        ),
      );

      expect(find.text('No points played yet'), findsNothing);
      expect(find.byType(MomentumChartView), findsOneWidget);
    });
  });
}
