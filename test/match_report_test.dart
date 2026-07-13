import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/match_report.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/synthetic_frames.dart';

/// A frame carrying the ball plus two stationary players (left + right halves),
/// so the pose-driven movement section is populated.
FrameResult _ballAndPlayers(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, y, 0, 0)),
      people: const [
        PersonPose(box: BBox(0.20, 0.30, 0.06, 0.40), keypoints: []),
        PersonPose(box: BBox(0.74, 0.30, 0.06, 0.40), keypoints: []),
      ],
    );

/// A double bounce on the right half (with players present) — the referee
/// awards this to Player A while the pose section tracks both players.
List<FrameResult> _doubleBounceOn(double x, {int startT = 0}) => [
      _ballAndPlayers(startT + 0, x, 0.30),
      _ballAndPlayers(startT + 33, x, 0.50),
      _ballAndPlayers(startT + 66, x, 0.70),
      _ballAndPlayers(startT + 99, x, 0.50), // bounce 1 reported (apex 0.70)
      _ballAndPlayers(startT + 132, x, 0.70),
      _ballAndPlayers(startT + 165, x, 0.50), // bounce 2 -> double bounce
      _ballAndPlayers(startT + 198, x, 0.70),
    ];

void main() {
  test('report composes every analytics section from the demo match', () {
    final controller = MatchController();
    for (final frame in demoMatchFrames()) {
      controller.onFrame(frame);
    }

    final report = buildMatchReport(controller);

    // Scoring, rally, both players' movement and both sides' placement all
    // appear in one artifact.
    expect(report, contains('Match summary'));
    expect(report, contains('Rally analysis'));
    expect(report, contains('Player A movement'));
    expect(report, contains('Player B movement'));
    expect(report, contains('Left side placement'));
    expect(report, contains('Right side placement'));

    // The demo frames carry only a ball (no people), so movement is untracked.
    expect(report, contains('not tracked'));

    // The demo scores real points, so the scoring line reflects them.
    expect(controller.summary.totalPoints, greaterThan(0));
    expect(
      report,
      contains('${controller.summary.totalPoints} points played'),
    );
  });

  test('report includes tracked footwork when players are in frame', () {
    final controller = MatchController();
    for (final frame in _doubleBounceOn(0.80)) {
      controller.onFrame(frame);
    }

    final report = buildMatchReport(controller);

    // Players were located every frame, so the movement section is filled in.
    expect(report, contains('distance travelled'));
    expect(report, contains('court coverage'));
    expect(report, contains('mobility'));

    // A bounce landed on the right half, so its placement section has a count.
    expect(report, contains('Right side placement'));
    expect(report, contains('avg depth'));
  });

  test('empty match still produces a well-formed report', () {
    final report = buildMatchReport(MatchController());

    expect(report, contains('Match summary'));
    expect(report, contains('0 points played'));
    expect(report, contains('No rallies recorded yet.'));
    expect(report, contains('no bounces recorded'));
  });
}
