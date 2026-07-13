import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/benchmark/benchmark_runner.dart';
import 'package:pong_ai/core/benchmark/clip_fixture.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';

/// A fixture built from the deterministic demo match (known 5-2 to A).
ClipFixture _demoClip({List<Player>? winners}) => ClipFixture(
      name: 'synthetic_demo',
      source: 'synthetic',
      frames: demoMatchFrames(),
      groundTruth: ClipGroundTruth(
        pointsA: 5,
        pointsB: 2,
        pointWinners: winners,
      ),
    );

void main() {
  group('BenchmarkRunner', () {
    test('scores the demo clip exactly against ground truth', () {
      final result = const BenchmarkRunner().run(_demoClip());

      expect(result.detectedPointsA, 5);
      expect(result.detectedPointsB, 2);
      expect(result.finalScoreCorrect, isTrue);
      expect(result.pointTotalError, 0);
      expect(result.undeterminedCount, 0);
      expect(result.pointRecall, 1.0);
    });

    test('computes ordered point accuracy when winners are labeled', () {
      // True demo order is A,B,A,A,B,A,A.
      const winners = [
        Player.a,
        Player.b,
        Player.a,
        Player.a,
        Player.b,
        Player.a,
        Player.a,
      ];
      final result = const BenchmarkRunner().run(_demoClip(winners: winners));

      expect(result.orderedComparable, 7);
      expect(result.orderedMatches, 7);
      expect(result.orderedAccuracy, 1.0);
    });

    test('penalises a wrong ground-truth ordering', () {
      // Flip the first two winners: two positions now mismatch.
      const winners = [
        Player.b,
        Player.a,
        Player.a,
        Player.a,
        Player.b,
        Player.a,
        Player.a,
      ];
      final result = const BenchmarkRunner().run(_demoClip(winners: winners));

      expect(result.orderedMatches, 5);
      expect(result.orderedComparable, 7);
      expect(result.orderedAccuracy, closeTo(5 / 7, 1e-9));
    });

    test('flags a mismatched aggregate score', () {
      final clip = ClipFixture(
        name: 'wrong_truth',
        frames: demoMatchFrames(),
        groundTruth: const ClipGroundTruth(pointsA: 4, pointsB: 3),
      );
      final result = const BenchmarkRunner().run(clip);

      expect(result.finalScoreCorrect, isFalse);
      expect(result.pointTotalError, 2); // |5-4| + |2-3|
      expect(result.orderedAccuracy, isNull);
    });

    test('honours a mirrored left-player mapping', () {
      // With B on the left, every right-side bounce now awards B instead of A,
      // so the demo's 5-2-to-A becomes 5-2-to-B.
      final clip = ClipFixture(
        name: 'mirrored',
        frames: demoMatchFrames(),
        leftPlayer: Player.b,
        groundTruth: const ClipGroundTruth(pointsA: 2, pointsB: 5),
      );
      final result = const BenchmarkRunner().run(clip);

      expect(result.detectedPointsA, 2);
      expect(result.detectedPointsB, 5);
      expect(result.finalScoreCorrect, isTrue);
    });

    test('counts an in-flight ball loss as undetermined, not a point', () {
      // A single arc that crosses the net (so the loss is mid-flight, no prior
      // same-side bounce to attribute) -> outOfPlay/undetermined.
      const xs = <double>[0.40, 0.55, 0.65]; // crosses net left->right
      final frames = <FrameResult>[
        for (var i = 0; i < xs.length; i++)
          FrameResult(
            timestampMs: i * kSyntheticFrameStepMs,
            ball: Detection(
              label: 'ball',
              confidence: 0.9,
              box: BBox(xs[i] - 0.01, 0.49, 0.02, 0.02),
            ),
          ),
        for (var i = xs.length; i < xs.length + 9; i++)
          FrameResult(timestampMs: i * kSyntheticFrameStepMs),
      ];
      final clip = ClipFixture(
        name: 'inflight_loss',
        frames: frames,
        groundTruth: const ClipGroundTruth(pointsA: 0, pointsB: 0),
      );
      final result = const BenchmarkRunner().run(clip);

      expect(result.detectedTotal, 0);
      expect(result.undeterminedCount, 1);
    });
  });

  group('ClipFixture JSON', () {
    test('round-trips through JSON without losing the outcome', () {
      final clip = _demoClip(
        winners: const [Player.a, Player.b, Player.a],
      );
      final encoded = jsonEncode(clip.toJson());
      final decoded = ClipFixture.fromJson(
        jsonDecode(encoded) as Map<String, dynamic>,
      );

      expect(decoded.name, clip.name);
      expect(decoded.frames.length, clip.frames.length);
      expect(decoded.groundTruth.pointsA, 5);
      expect(decoded.groundTruth.pointWinners, hasLength(3));

      // The reconstructed fixture must benchmark identically.
      final before = const BenchmarkRunner().run(clip);
      final after = const BenchmarkRunner().run(decoded);
      expect(after.detectedPointsA, before.detectedPointsA);
      expect(after.detectedPointsB, before.detectedPointsB);
    });

    test('preserves ball, people and keypoints through a frame round-trip', () {
      const clip = ClipFixture(
        name: 'rich_frame',
        source: 'unit',
        netX: 0.42,
        frames: [
          FrameResult(
            timestampMs: 0,
            fps: 28.5,
            ball: Detection(
              label: 'ball',
              confidence: 0.8,
              box: BBox(0.1, 0.2, 0.02, 0.02),
            ),
            people: [
              PersonPose(
                box: BBox(0.0, 0.1, 0.2, 0.6),
                trackId: 7,
                keypoints: [Keypoint(0.11, 0.22, 0.9)],
              ),
            ],
          ),
        ],
        groundTruth: ClipGroundTruth(pointsA: 0, pointsB: 0),
      );

      final decoded = ClipFixture.fromJson(
        jsonDecode(jsonEncode(clip.toJson())) as Map<String, dynamic>,
      );
      final frame = decoded.frames.single;

      expect(decoded.netX, 0.42);
      expect(frame.fps, 28.5);
      expect(frame.ball!.box.left, closeTo(0.1, 1e-9));
      expect(frame.people.single.trackId, 7);
      expect(frame.people.single.keypoints.single.confidence, closeTo(0.9, 1e-9));
    });
  });

  group('shipped fixture corpus', () {
    test('benchmark/clips/synthetic_demo.json scores exactly', () {
      final file = File('benchmark/clips/synthetic_demo.json');
      expect(
        file.existsSync(),
        isTrue,
        reason: 'run from the package root',
      );
      final clip = ClipFixture.fromJson(
        jsonDecode(file.readAsStringSync()) as Map<String, dynamic>,
      );
      final result = const BenchmarkRunner().run(clip);

      expect(result.finalScoreCorrect, isTrue);
      expect(result.detectedPointsA, 5);
      expect(result.detectedPointsB, 2);
      expect(result.orderedAccuracy, 1.0);
    });
  });

  group('BenchmarkSuiteResult', () {
    test('aggregates across multiple clips', () {
      final suite = const BenchmarkRunner().runAll([
        _demoClip(),
        ClipFixture(
          name: 'mirrored',
          frames: demoMatchFrames(),
          leftPlayer: Player.b,
          groundTruth: const ClipGroundTruth(pointsA: 2, pointsB: 5),
        ),
      ]);

      expect(suite.clipCount, 2);
      expect(suite.clipsExactlyCorrect, 2);
      expect(suite.meanPointRecall, 1.0);
      expect(suite.report(), contains('Benchmark suite: 2 clip(s)'));
    });
  });
}
