import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/match_report_json.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';

/// A frame carrying the ball plus two stationary players (left + right halves),
/// so the pose-driven movement/tracking sections are populated.
FrameResult _ballAndPlayers(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, y, 0, 0)),
      people: const [
        PersonPose(box: BBox(0.20, 0.30, 0.06, 0.40), keypoints: []),
        PersonPose(box: BBox(0.74, 0.30, 0.06, 0.40), keypoints: []),
      ],
    );

/// A ball travelling right→left then double-bouncing on the left half, with
/// both players present the whole time — one decided point with real ball
/// speed, movement, placement and tracking data.
List<FrameResult> _movingRally() => [
      _ballAndPlayers(0, 0.80, 0.30),
      _ballAndPlayers(33, 0.60, 0.50),
      _ballAndPlayers(66, 0.40, 0.70),
      _ballAndPlayers(99, 0.40, 0.50), // bounce apex on left
      _ballAndPlayers(132, 0.40, 0.70),
      _ballAndPlayers(165, 0.40, 0.50), // double bounce -> point
    ];

void main() {
  test('structured report round-trips through JSON and mirrors the score', () {
    final controller = MatchController();
    for (final frame in demoMatchFrames()) {
      controller.onFrame(frame);
    }

    final jsonString = matchReportJsonString(controller);
    // Re-parses to an equal map — the export is genuinely machine-readable.
    final decoded = jsonDecode(jsonString) as Map<String, dynamic>;
    expect(decoded, equals(buildMatchReportJson(controller)));

    expect(decoded['schemaVersion'], matchReportSchemaVersion);

    final score = decoded['score'] as Map<String, dynamic>;
    expect(score['pointsA'], controller.score.pointsA);
    expect(score['pointsB'], controller.score.pointsB);
    expect(score['isMatchOver'], controller.score.isMatchOver);

    final summary = decoded['summary'] as Map<String, dynamic>;
    expect(summary['totalPoints'], controller.summary.totalPoints);
    final players = summary['players'] as Map<String, dynamic>;
    expect(players.keys, containsAll(<String>['A', 'B']));
    final playerA = players['A'] as Map<String, dynamic>;
    expect(playerA['points'], controller.summary.pointsWonBy(Player.a));
  });

  test('empty match emits a well-formed, null-populated report', () {
    final json = buildMatchReportJson(MatchController());

    expect((json['summary'] as Map)['totalPoints'], 0);
    // No data collected yet, so the optional sections are explicit nulls (not
    // missing keys) — a reader can rely on the schema shape.
    expect(json['ballSpeed'], isNull);
    expect(json['trackingQuality'], isNull);
    expect(json['coaching'], isNull);
    final movement = json['movement'] as Map<String, dynamic>;
    expect(movement['A'], isNull);
    expect(movement['B'], isNull);
    final placement = json['placement'] as Map<String, dynamic>;
    expect(placement['left'], isNull);
    expect(placement['right'], isNull);

    // Still valid JSON.
    expect(jsonDecode(matchReportJsonString(MatchController())), isA<Map>());
  });

  test('populated rally fills ball-speed, movement, placement and tracking', () {
    final controller = MatchController();
    for (final frame in _movingRally()) {
      controller.onFrame(frame);
    }

    final json = buildMatchReportJson(controller);

    // The ball moved along the table, so a physical speed is present.
    final speed = json['ballSpeed'] as Map<String, dynamic>;
    expect(speed['maxKmh'], greaterThan(0));

    // Both players were visible every frame, so tracking quality is graded and
    // movement is tracked for both.
    final tracking = json['trackingQuality'] as Map<String, dynamic>;
    expect(tracking['grade'], isA<String>());
    expect(tracking['twoPlayerRate'], closeTo(1.0, 1e-9));
    expect((json['movement'] as Map)['A'], isNotNull);
    expect((json['movement'] as Map)['B'], isNotNull);

    // A bounce landed on the left half, so left placement is populated.
    final placement = json['placement'] as Map<String, dynamic>;
    expect(placement['left'], isNotNull);
    expect((placement['left'] as Map)['count'], greaterThan(0));
  });

  test('point log lists every scored rally in order with its fields', () {
    final controller = MatchController();
    for (final frame in demoMatchFrames()) {
      controller.onFrame(frame);
    }

    final pointLog = buildMatchReportJson(controller)['pointLog'] as List;
    // One entry per scored point, ordered like the durable log.
    expect(pointLog, hasLength(controller.summary.points.length));
    expect(pointLog, isNotEmpty);

    for (var i = 0; i < pointLog.length; i++) {
      final entry = pointLog[i] as Map<String, dynamic>;
      final source = controller.summary.points[i];
      expect(entry['winner'], source.winner == Player.a ? 'A' : 'B');
      expect(entry['reason'], source.reason.name);
      expect(entry['timestampMs'], source.timestampMs);
      expect(entry['gameIndex'], source.gameIndex);
      expect(
        entry['server'],
        source.server == null
            ? isNull
            : (source.server == Player.a ? 'A' : 'B'),
      );
    }

    // The whole export still round-trips through JSON with the new section.
    final decoded = jsonDecode(matchReportJsonString(controller))
        as Map<String, dynamic>;
    expect(decoded['pointLog'], equals(pointLog));
  });

  test('empty match emits an empty (not null) point log', () {
    final json = buildMatchReportJson(MatchController());
    expect(json['pointLog'], isA<List>());
    expect(json['pointLog'], isEmpty);
  });

  test('played match exports the per-player coaching section', () {
    final controller = MatchController();
    for (final frame in demoMatchFrames()) {
      controller.onFrame(frame);
    }

    final coaching = buildMatchReportJson(controller)['coaching']
        as Map<String, dynamic>;
    // Serve data was captured per point, so at least one player has a focus.
    final playerA = coaching['A'] as Map<String, dynamic>;
    expect(playerA['grade'], isA<String>());
    expect(playerA['overallScore'], isA<num>());
    expect(playerA['focus'], isA<String>());
    expect(playerA['focusTip'], isA<String>());
    expect(playerA['strength'], isA<String>());
    final dims = playerA['dimensions'] as List;
    expect(dims, isNotEmpty);
    expect((dims.first as Map)['name'], isA<String>());
    expect((dims.first as Map)['score'], isA<num>());
  });
}
