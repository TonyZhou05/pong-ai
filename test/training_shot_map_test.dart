import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/training/shot_analyzer.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/features/training/training_shot_map.dart';

Shot _shot({required double depth, required double lateral}) => Shot(
      timestampMs: 0,
      speed: 1,
      depth: depth,
      lateral: lateral,
      score: 1,
    );

/// One frame whose ball detection is centered at ([x], [y]).
FrameResult _frame(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(
        label: 'ball',
        confidence: 0.9,
        box: BBox(x - 0.01, y - 0.01, 0.02, 0.02),
      ),
    );

/// A down-up arc whose apex (bounce) lands on the 3rd sample.
List<FrameResult> _arc(List<double> xs, List<double> ys, {int step = 33}) {
  assert(xs.length == ys.length);
  return [for (var i = 0; i < xs.length; i++) _frame(i * step, xs[i], ys[i])];
}

void main() {
  group('trainingShotMapPosition', () {
    test('depth maps to x (net at 0, baseline at 1)', () {
      expect(
        trainingShotMapPosition(_shot(depth: 0, lateral: 0.5)).x,
        closeTo(0.0, 1e-9),
      );
      expect(
        trainingShotMapPosition(_shot(depth: 1, lateral: 0.5)).x,
        closeTo(1.0, 1e-9),
      );
    });

    test('lateral maps straight to y', () {
      expect(
        trainingShotMapPosition(_shot(depth: 0.5, lateral: 0.2)).y,
        closeTo(0.2, 1e-9),
      );
    });

    test('out-of-range inputs are clamped into the table', () {
      final p = trainingShotMapPosition(_shot(depth: 2, lateral: -1));
      expect(p.x, closeTo(1.0, 1e-9));
      expect(p.y, closeTo(0.0, 1e-9));
    });
  });

  group('Shot lateral enrichment', () {
    test('a completed shot records the apex bounce lateral', () {
      final analyzer = ShotAnalyzer();
      // Apex (local-max y) lands on the 3rd sample at y=0.60 → lateral 0.60.
      final shots = <Shot>[];
      for (final f in _arc(
        [0.30, 0.60, 0.875, 0.95, 0.98],
        [0.30, 0.45, 0.60, 0.45, 0.30],
      )) {
        final s = analyzer.onFrame(f);
        if (s != null) shots.add(s);
      }
      expect(shots, hasLength(1));
      expect(shots.single.lateral, closeTo(0.60, 1e-9));
    });
  });

  group('TrainingShotMapView', () {
    testWidgets('shows a hint when there are no shots', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(body: TrainingShotMapView(shots: [])),
        ),
      );
      expect(find.text('No shots tracked yet'), findsOneWidget);
      expect(find.byType(CustomPaint), findsWidgets);
    });

    testWidgets('drops the hint and paints once shots exist', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: TrainingShotMapView(
              shots: [_shot(depth: 0.75, lateral: 0.4)],
            ),
          ),
        ),
      );
      expect(find.text('No shots tracked yet'), findsNothing);
      expect(find.byType(TrainingShotMapView), findsOneWidget);
    });
  });
}
