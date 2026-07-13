import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/player_movement.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/features/summary/player_map.dart';

FramePoint _foot(double x, double y) => (x: x, y: y);

PersonPose _personAt(double footX, double footY) {
  final kps = List.generate(
    17,
    (i) => const Keypoint(0, 0, 0),
  );
  kps[kLeftAnkleIndex] = Keypoint(footX, footY, 0.9);
  kps[kRightAnkleIndex] = Keypoint(footX, footY, 0.9);
  return PersonPose(
    box: BBox(footX - 0.05, footY - 0.3, 0.1, 0.3),
    keypoints: kps,
  );
}

void main() {
  group('playerMapPosition', () {
    test('a foot on the net maps to the centre line', () {
      expect(playerMapPosition(_foot(0.5, 0.5)).x, closeTo(0.5, 1e-9));
    });

    test('each half is scaled independently around a shifted net', () {
      // Net at 0.4: the left half [0,0.4] maps to [0,0.5], right [0.4,1] to
      // [0.5,1], so the net stays centred on the map.
      expect(playerMapPosition(_foot(0.4, 0.5), netX: 0.4).x, closeTo(0.5, 1e-9));
      expect(playerMapPosition(_foot(0.2, 0.5), netX: 0.4).x, closeTo(0.25, 1e-9));
      expect(playerMapPosition(_foot(0.7, 0.5), netX: 0.4).x, closeTo(0.75, 1e-9));
    });

    test('y passes straight through and out-of-range inputs clamp', () {
      final p = playerMapPosition(_foot(2, -1));
      expect(p.x, closeTo(1.0, 1e-9));
      expect(p.y, closeTo(0.0, 1e-9));
      expect(playerMapPosition(_foot(0.3, 0.8)).y, closeTo(0.8, 1e-9));
    });
  });

  group('PlayerMovementAnalyzer.positionsFor', () {
    test('retains one foot sample per detected frame per player', () {
      final analyzer = PlayerMovementAnalyzer();
      // Left-half player (Player.a) and right-half player (Player.b).
      for (var i = 0; i < 3; i++) {
        analyzer.observe(
          FrameResult(
            timestampMs: i * 33,
            ball: null,
            people: [_personAt(0.25 + i * 0.01, 0.6), _personAt(0.75, 0.4)],
          ),
        );
      }
      expect(analyzer.positionsFor(Player.a).length, 3);
      expect(analyzer.positionsFor(Player.b).length, 3);
      expect(analyzer.positionsFor(Player.a).first.x, closeTo(0.25, 1e-9));
      // reset clears the samples too.
      analyzer.reset();
      expect(analyzer.positionsFor(Player.a), isEmpty);
    });
  });

  group('PlayerPositionMapView', () {
    testWidgets('shows a hint when no player was tracked', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: PlayerPositionMapView(
              positions: {Player.a: [], Player.b: []},
            ),
          ),
        ),
      );
      expect(find.text('No player movement tracked yet'), findsOneWidget);
      expect(find.byType(CustomPaint), findsWidgets);
    });

    testWidgets('drops the hint and paints once positions exist', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PlayerPositionMapView(
              positions: {
                Player.a: [_foot(0.25, 0.5), _foot(0.3, 0.55)],
                Player.b: [_foot(0.75, 0.45)],
              },
            ),
          ),
        ),
      );
      expect(find.text('No player movement tracked yet'), findsNothing);
      expect(find.byType(PlayerPositionMapView), findsOneWidget);
    });
  });
}
