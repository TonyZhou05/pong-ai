import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/benchmark/benchmark_corpus.dart';
import 'package:pong_ai/core/benchmark/clip_fixture.dart';
import 'package:pong_ai/core/benchmark/event_metrics.dart';
import 'package:pong_ai/core/vision/detection.dart';

FrameResult _ball(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 1, box: BBox(x, y, 0, 0)),
    );

/// A down-then-up arc whose apex (a bounce) lands on the middle sample's t.
List<FrameResult> _bounceArc(int t0, double x, {int step = 33}) => [
      _ball(t0, x, 0.40),
      _ball(t0 + step, x, 0.60),
      _ball(t0 + 2 * step, x, 0.55),
    ];

void main() {
  group('loadClipDirectory', () {
    test('discovers and parses the shipped corpus', () {
      final clips = loadClipDirectory();
      expect(clips, isNotEmpty);
      // The shipped synthetic demo must be present and parse cleanly.
      expect(clips.map((c) => c.name), contains('synthetic_demo_5_2'));
    });

    test('returns empty for a missing directory', () {
      expect(loadClipDirectory('benchmark/does_not_exist'), isEmpty);
    });
  });

  group('loadClipFixtures', () {
    test('reads an explicit fixture path', () {
      final clips = loadClipFixtures(['benchmark/clips/synthetic_demo.json']);
      expect(clips, hasLength(1));
      expect(clips.single.frames, isNotEmpty);
    });

    test('throws a path-tagged error on invalid JSON', () {
      final tmp = File(
        '${Directory.systemTemp.path}/pong_bad_clip.json',
      )..writeAsStringSync('{not json');
      addTearDown(() => tmp.deleteSync());
      expect(
        () => loadClipFixtures([tmp.path]),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('pong_bad_clip.json'),
          ),
        ),
      );
    });
  });

  group('buildCorpusReport', () {
    test('empty corpus reports zero clips and the hint', () {
      final report = buildCorpusReport(const []);
      expect(report, contains('Clips loaded: 0'));
      expect(report, contains('No clips found'));
    });

    test('composes scoring + perception stages for the shipped corpus', () {
      final report = buildCorpusReport(loadClipDirectory());
      expect(report, contains('Stage 1: scoring accuracy'));
      expect(report, contains('Stage 2: perception accuracy'));
      expect(report, contains('synthetic_demo_5_2'));
    });

    test('notes when no clip carries per-frame ground truth', () {
      // The synthetic demo has no groundTruthFrames, so perception is empty.
      final clips = loadClipFixtures(['benchmark/clips/synthetic_demo.json']);
      final report = buildCorpusReport(clips);
      expect(report, contains('No clips carry per-frame ground truth'));
    });

    test('scores the perception stage when ground truth frames are present', () {
      // Round-trip the demo, then attach its own frames as ground truth so the
      // perception stage has something to score (a perfect self-detector).
      final base = loadClipFixtures([
        'benchmark/clips/synthetic_demo.json',
      ]).single;
      final withGt = ClipFixture.fromJson({
        ...base.toJson(),
        'groundTruthFrames':
            (base.toJson()['frames'] as List<dynamic>),
      });
      final report = buildCorpusReport([withGt]);
      expect(report, contains('Perception: ${base.name}'));
      // A perfect self-detector should score 100% ball recall.
      expect(report, contains('Ball  P/R/F1: 100.0%/100.0%'));
    });

    test('notes when no clip carries ground-truth events', () {
      final clips = loadClipFixtures(['benchmark/clips/synthetic_demo.json']);
      final report = buildCorpusReport(clips);
      expect(report, contains('Stage 3: event-detection accuracy'));
      expect(report, contains('No clips carry ground-truth events'));
    });

    test('scores the event stage when ground-truth events are present', () {
      final clip = ClipFixture(
        name: 'bounce_clip',
        frames: _bounceArc(0, 0.30),
        groundTruthEvents: const [
          GroundTruthEvent(33, TrackedEventType.bounce),
        ],
        groundTruth: const ClipGroundTruth(pointsA: 0, pointsB: 0),
      );
      final report = buildCorpusReport([clip]);
      expect(report, contains('Stage 3: event-detection accuracy'));
      expect(report, contains('Events: bounce_clip'));
      // A clean arc against a matching label is a perfect bounce detection.
      expect(report, contains('Bounce    P/R/F1: 100.0%/100.0%/100.0%'));
    });

    test('ground-truth events survive a JSON round-trip', () {
      final clip = ClipFixture(
        name: 'evt',
        frames: _bounceArc(0, 0.30),
        groundTruthEvents: const [
          GroundTruthEvent(33, TrackedEventType.bounce),
          GroundTruthEvent(200, TrackedEventType.netCross),
        ],
        groundTruth: const ClipGroundTruth(pointsA: 0, pointsB: 0),
      );
      final again = ClipFixture.fromJson(
        jsonDecode(jsonEncode(clip.toJson())) as Map<String, dynamic>,
      );
      expect(again.groundTruthEvents, hasLength(2));
      expect(again.groundTruthEvents![0].timestampMs, 33);
      expect(again.groundTruthEvents![0].type, TrackedEventType.bounce);
      expect(again.groundTruthEvents![1].type, TrackedEventType.netCross);
    });

    test('the shipped corpus round-trips through JSON without loss', () {
      for (final clip in loadClipDirectory()) {
        final again = ClipFixture.fromJson(
          jsonDecode(jsonEncode(clip.toJson())) as Map<String, dynamic>,
        );
        expect(again.name, clip.name);
        expect(again.frames.length, clip.frames.length);
      }
    });
  });
}
