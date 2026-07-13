import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/benchmark/clip_fixture.dart';
import 'package:pong_ai/core/benchmark/detection_metrics.dart';
import 'package:pong_ai/core/benchmark/openttgames_converter.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';

void main() {
  // A tiny synthetic OpenTTGames-shaped ball_markup: frame index -> pixel {x,y}.
  // Out of order and with a gap to exercise sorting; 1280x720 @ 120fps.
  final markup = <String, dynamic>{
    '2': {'x': 640, 'y': 360}, // frame centre -> (0.5, 0.5)
    '0': {'x': 0, 'y': 0}, // top-left corner -> box clamped to (0,0)
    '5': {'x': -1, 'y': -1}, // ball absent
    '1': {'x': 1280, 'y': 720}, // bottom-right -> box clamped to fit
  };

  group('openTtGamesGroundTruthFrames', () {
    test('sorts by frame index and maps pixels to normalized boxes', () {
      final frames = openTtGamesGroundTruthFrames(
        ballMarkup: markup,
        frameWidth: 1280,
        frameHeight: 720,
        fps: 120,
      );

      // Sorted: indices 0, 1, 2, 5.
      expect(frames.length, 4);
      expect(
        frames.map((f) => f.timestampMs).toList(),
        [0, (1000 / 120).round(), (2 * 1000 / 120).round(), (5 * 1000 / 120).round()],
      );

      // Centre-of-frame ball (index 2) -> box centred on (0.5, 0.5).
      final centre = frames[2].ball!;
      expect(centre.box.centerX, closeTo(0.5, 1e-9));
      expect(centre.box.centerY, closeTo(0.5, 1e-9));
      expect(centre.box.width, closeTo(kOpenTtGamesBallSize, 1e-9));
      expect(centre.label, 'ball');
      expect(centre.confidence, 1.0);
    });

    test('clamps corner balls to stay inside the frame', () {
      final frames = openTtGamesGroundTruthFrames(
        ballMarkup: markup,
        frameWidth: 1280,
        frameHeight: 720,
        fps: 120,
      );

      final topLeft = frames[0].ball!.box; // pixel (0,0)
      expect(topLeft.left, 0.0);
      expect(topLeft.top, 0.0);
      expect(topLeft.left + topLeft.width, lessThanOrEqualTo(1.0));

      final bottomRight = frames[1].ball!.box; // pixel (1280,720)
      expect(bottomRight.left + bottomRight.width, closeTo(1.0, 1e-9));
      expect(bottomRight.top + bottomRight.height, closeTo(1.0, 1e-9));
    });

    test('negative coordinates yield a ball-less frame', () {
      final frames = openTtGamesGroundTruthFrames(
        ballMarkup: markup,
        frameWidth: 1280,
        frameHeight: 720,
        fps: 120,
      );
      // Index 5 has x=y=-1 -> no ball, but the frame is still present.
      expect(frames[3].timestampMs, (5 * 1000 / 120).round());
      expect(frames[3].ball, isNull);
    });

    test('ignores non-integer keys', () {
      final frames = openTtGamesGroundTruthFrames(
        ballMarkup: {'frame_3': {'x': 10, 'y': 10}, '3': {'x': 10, 'y': 10}},
        frameWidth: 100,
        frameHeight: 100,
        fps: 30,
      );
      expect(frames.length, 1);
    });
  });

  group('clipFixtureFromOpenTtGames', () {
    test('populates ground-truth frames and defaults predictions to them', () {
      final clip = clipFixtureFromOpenTtGames(
        name: 'openttgames_game1',
        ballMarkup: markup,
        frameWidth: 1280,
        frameHeight: 720,
        fps: 120,
        netX: 0.5,
      );

      expect(clip.source, 'OpenTTGames');
      expect(clip.fps, 120);
      expect(clip.groundTruthFrames, isNotNull);
      expect(clip.groundTruthFrames!.length, 4);
      // No predictions given -> frames mirror the ground truth (perfect baseline).
      expect(clip.frames.length, clip.groundTruthFrames!.length);
    });

    test('perfect-detector baseline scores full ball recall', () {
      final clip = clipFixtureFromOpenTtGames(
        name: 'baseline',
        ballMarkup: markup,
        frameWidth: 1280,
        frameHeight: 720,
        fps: 120,
      );
      final result = const DetectionBenchmark().evaluateClip(clip);
      expect(result, isNotNull);
      // 3 frames have a labeled ball; predictions equal ground truth.
      expect(result!.ball.recall, closeTo(1.0, 1e-9));
      expect(result.ball.precision, closeTo(1.0, 1e-9));
      expect(result.ball.truePositives, 3);
      expect(result.ball.falseNegatives, 0);
    });

    test('a missed prediction shows up as a perception false negative', () {
      // Model dropped the ball on one labeled frame (index 2 becomes ball-less).
      final gt = openTtGamesGroundTruthFrames(
        ballMarkup: markup,
        frameWidth: 1280,
        frameHeight: 720,
        fps: 120,
      );
      final predicted = List.of(gt);
      predicted[2] = FrameResult(timestampMs: gt[2].timestampMs); // ball dropped

      final clip = clipFixtureFromOpenTtGames(
        name: 'dropped',
        ballMarkup: markup,
        frameWidth: 1280,
        frameHeight: 720,
        fps: 120,
        predictedFrames: predicted,
      );
      final result = const DetectionBenchmark().evaluateClip(clip)!;
      expect(result.ball.truePositives, 2);
      expect(result.ball.falseNegatives, 1);
      expect(result.ball.recall, closeTo(2 / 3, 1e-9));
    });

    test('round-trips through ClipFixture JSON unchanged', () {
      final clip = clipFixtureFromOpenTtGames(
        name: 'rt',
        ballMarkup: markup,
        frameWidth: 1280,
        frameHeight: 720,
        fps: 120,
        groundTruth: const ClipGroundTruth(pointsA: 11, pointsB: 7),
        leftPlayer: Player.b,
      );
      final back = ClipFixture.fromJson(clip.toJson());
      expect(back.name, 'rt');
      expect(back.leftPlayer, Player.b);
      expect(back.groundTruth.pointsA, 11);
      expect(back.groundTruthFrames!.length, clip.groundTruthFrames!.length);
      expect(
        back.groundTruthFrames![2].ball!.box.centerX,
        closeTo(0.5, 1e-9),
      );
    });
  });
}
