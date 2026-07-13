import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/analysis/bounce_placement.dart';
import 'package:pong_ai/features/summary/shot_map.dart';

BouncePlacement _bounce(
  TableSide side, {
  required double depth,
  required double lateral,
}) =>
    BouncePlacement(
      side: side,
      depthFromNet: depth,
      lateral: lateral,
      timestampMs: 0,
    );

void main() {
  group('shotMapPosition', () {
    test('a net-hugging bounce sits on the centre line for both sides', () {
      expect(
        shotMapPosition(_bounce(TableSide.left, depth: 0, lateral: 0.5)).x,
        closeTo(0.5, 1e-9),
      );
      expect(
        shotMapPosition(_bounce(TableSide.right, depth: 0, lateral: 0.5)).x,
        closeTo(0.5, 1e-9),
      );
    });

    test('depth fans outward toward each side baseline', () {
      // Left baseline is the left edge (x=0); right baseline is the right edge.
      expect(
        shotMapPosition(_bounce(TableSide.left, depth: 1, lateral: 0)).x,
        closeTo(0.0, 1e-9),
      );
      expect(
        shotMapPosition(_bounce(TableSide.right, depth: 1, lateral: 0)).x,
        closeTo(1.0, 1e-9),
      );
    });

    test('lateral maps straight to y', () {
      expect(
        shotMapPosition(_bounce(TableSide.left, depth: 0.4, lateral: 0.2)).y,
        closeTo(0.2, 1e-9),
      );
    });

    test('out-of-range inputs are clamped into the table', () {
      final p = shotMapPosition(
        _bounce(TableSide.right, depth: 2, lateral: -1),
      );
      expect(p.x, closeTo(1.0, 1e-9));
      expect(p.y, closeTo(0.0, 1e-9));
    });
  });

  group('ShotMapView', () {
    testWidgets('shows a hint when there are no bounces', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: ShotMapView(
              left: SidePlacementStats(TableSide.left, []),
              right: SidePlacementStats(TableSide.right, []),
            ),
          ),
        ),
      );

      expect(find.text('No bounces tracked yet'), findsOneWidget);
      expect(find.byType(CustomPaint), findsWidgets);
    });

    testWidgets('drops the hint and paints once bounces exist', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: ShotMapView(
              left: SidePlacementStats(TableSide.left, [
                _bounce(TableSide.left, depth: 0.8, lateral: 0.3),
              ]),
              right: SidePlacementStats(TableSide.right, [
                _bounce(TableSide.right, depth: 0.2, lateral: 0.7),
              ]),
            ),
          ),
        ),
      );

      expect(find.text('No bounces tracked yet'), findsNothing);
      expect(find.byType(ShotMapView), findsOneWidget);
    });
  });
}
