import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/analysis/rally_referee.dart';
import 'package:pong_ai/core/analysis/table_calibrator.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';

FrameResult _frame(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, y, 0, 0)),
    );
FrameResult _empty(int t) => FrameResult(timestampMs: t);

/// A double bounce on side `x` whose apex sits at [highY] (the lowest on-screen
/// point of the arc), so the calibrated table surface can gate it in or out.
List<FrameResult> _doubleBounceAt(
  double x,
  double lowY,
  double highY, {
  int startT = 0,
}) {
  final mid = (lowY + highY) / 2;
  return [
    _frame(startT + 0, x, lowY),
    _frame(startT + 33, x, mid),
    _frame(startT + 66, x, highY),
    _frame(startT + 99, x, mid), // bounce 1 reported (apex was highY)
    _frame(startT + 132, x, highY),
    _frame(startT + 165, x, mid), // bounce 2 -> double bounce
  ];
}

/// A warm-up sweep of the ball across the table with both players visible, so
/// [TableCalibrator] can infer a surface band (x≈0.23–0.73, y≈0.42–0.54) and a
/// net line at the players' midpoint (≈0.5).
List<FrameResult> _warmup({int count = 8, int startT = 0}) => [
      for (var i = 0; i < count; i++)
        FrameResult(
          timestampMs: startT + i * 33,
          ball: Detection(
            label: 'ball',
            confidence: 0.9,
            box: BBox(0.20 + 0.08 * i, i.isEven ? 0.42 : 0.54, 0, 0),
          ),
          people: const [
            PersonPose(box: BBox(0.22, 0.30, 0.05, 0.40), keypoints: []),
            PersonPose(box: BBox(0.73, 0.30, 0.05, 0.40), keypoints: []),
          ],
        ),
    ];

/// A ball bouncing twice on one side (`x`) with no net crossing: down-up-down-up
/// in y. The tracker reports each low point (y=0.70) as a bounce.
List<FrameResult> _doubleBounceOn(double x, {int startT = 0}) => [
      _frame(startT + 0, x, 0.30),
      _frame(startT + 33, x, 0.50),
      _frame(startT + 66, x, 0.70),
      _frame(startT + 99, x, 0.50), // bounce 1 reported here (apex was 0.70)
      _frame(startT + 132, x, 0.70),
      _frame(startT + 165, x, 0.50), // bounce 2 -> double bounce
    ];

void main() {
  group('MatchController — auto-scoring', () {
    test('a double bounce on the right awards the left player a point', () {
      final mc = MatchController();
      List<PointDecision> last = const [];
      for (final f in _doubleBounceOn(0.75)) {
        last = mc.onFrame(f);
      }
      expect(last.single.reason, PointReason.doubleBounce);
      expect(last.single.winner, Player.a); // left player
      expect(mc.score.pointsA, 1);
      expect(mc.score.pointsB, 0);
    });

    test('two rallies won by opposite sides yield 1-1', () {
      final mc = MatchController();
      // Rally 1: double bounce on the right -> A scores.
      for (final f in _doubleBounceOn(0.75, startT: 0)) {
        mc.onFrame(f);
      }
      // Rally 2: double bounce on the left -> B scores.
      for (final f in _doubleBounceOn(0.25, startT: 1000)) {
        mc.onFrame(f);
      }
      expect(mc.score.pointsA, 1);
      expect(mc.score.pointsB, 1);
      expect(mc.undetermined, isEmpty);
    });

    test('a ball lost in flight is collected as undetermined, not scored', () {
      final mc = MatchController();
      // Establish a moving ball (no bounce), then lose it past maxGapFrames.
      mc.onFrame(_frame(0, 0.40, 0.50));
      mc.onFrame(_frame(33, 0.45, 0.50));
      final decisions = <PointDecision>[];
      for (var i = 1; i <= 8; i++) {
        decisions.addAll(mc.onFrame(_empty(33 + i * 33)));
      }
      expect(decisions.single.reason, PointReason.outOfPlay);
      expect(mc.score.pointsA, 0);
      expect(mc.score.pointsB, 0);
      expect(mc.undetermined, hasLength(1));
    });

    test('without a calibrator the controller is never in calibration', () {
      final mc = MatchController();
      expect(mc.isCalibrating, isFalse);
      expect(mc.geometry.left, 0.0);
      expect(mc.geometry.right, 1.0);
    });
  });

  group('MatchController — end changes between games', () {
    // Two unreturned double bounces on the right half (each a separate rally)
    // award the left player two points — a 2-point game.
    void winGameOnRight(MatchController mc, {required int startT}) {
      for (final f in _doubleBounceOn(0.75, startT: startT)) {
        mc.onFrame(f);
      }
      for (final f in _doubleBounceOn(0.75, startT: startT + 1000)) {
        mc.onFrame(f);
      }
    }

    // Two left-half double bounces award the right-half player two points — used
    // to hand the *other* player a game so a 1–1 deciding game can be reached.
    void winGameOnLeft(MatchController mc, {required int startT}) {
      for (final f in _doubleBounceOn(0.25, startT: startT)) {
        mc.onFrame(f);
      }
      for (final f in _doubleBounceOn(0.25, startT: startT + 1000)) {
        mc.onFrame(f);
      }
    }

    test('flips side→player attribution after a completed game (opt-in)', () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 2, bestOf: 3),
        switchEndsBetweenGames: true,
      );
      winGameOnRight(mc, startT: 0);
      expect(mc.score.gamesA, 1);
      expect(mc.score.gamesB, 0);
      // Players changed ends, so the referee now maps the left half to B.
      expect(mc.referee.leftPlayer, Player.b);

      // Game 2: the SAME physical right-side double bounce now awards B, because
      // the player standing on the right is A after the end change.
      for (final f in _doubleBounceOn(0.75, startT: 3000)) {
        mc.onFrame(f);
      }
      expect(mc.score.pointsB, 1);
      expect(mc.score.pointsA, 0);
    });

    test('is off by default so scripted clips score unchanged', () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 2, bestOf: 3),
      );
      winGameOnRight(mc, startT: 0);
      expect(mc.score.gamesA, 1);
      expect(mc.referee.leftPlayer, Player.a); // never switched

      // Game 2 right-side double bounce still awards the left player A.
      for (final f in _doubleBounceOn(0.75, startT: 3000)) {
        mc.onFrame(f);
      }
      expect(mc.score.pointsA, 1);
      expect(mc.score.pointsB, 0);
    });

    test('undo across a game boundary restores the pre-switch mapping', () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 2, bestOf: 3),
        switchEndsBetweenGames: true,
      );
      winGameOnRight(mc, startT: 0);
      expect(mc.referee.leftPlayer, Player.b);

      mc.undo(); // undo the game-winning point
      expect(mc.score.gamesA, 0);
      expect(mc.referee.leftPlayer, Player.a);
    });

    test('changes ends mid deciding game once a player reaches half the points',
        () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 2, bestOf: 3),
        switchEndsBetweenGames: true,
      );
      // Game 1: A (left) wins → ends switch, left becomes B.
      winGameOnRight(mc, startT: 0);
      // Game 2: B (now left) wins → ends switch back, left becomes A.
      winGameOnRight(mc, startT: 3000);
      expect(mc.score.gamesA, 1);
      expect(mc.score.gamesB, 1);
      expect(mc.referee.leftPlayer, Player.a); // decider starts with A on left

      // Deciding game (mid = 1): the first right-side point awards A and, because
      // A reached the midpoint, the players change ends → left becomes B.
      for (final f in _doubleBounceOn(0.75, startT: 6000)) {
        mc.onFrame(f);
      }
      expect(mc.score.pointsA, 1);
      expect(mc.referee.leftPlayer, Player.b);

      // The SAME physical right-side double bounce now awards B — the player
      // standing on the right after the mid-game end change.
      for (final f in _doubleBounceOn(0.75, startT: 7000)) {
        mc.onFrame(f);
      }
      expect(mc.score.pointsB, 1);
      expect(mc.referee.leftPlayer, Player.b); // fires only once per game
    });

    test('mid deciding-game end change stays off by default', () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 2, bestOf: 3),
      );
      winGameOnRight(mc, startT: 0); // A wins game 1
      winGameOnLeft(mc, startT: 3000); // B wins game 2 → games 1–1
      expect(mc.score.gamesA, 1);
      expect(mc.score.gamesB, 1);
      expect(mc.referee.leftPlayer, Player.a);

      // Deciding game: A reaches the midpoint, but with switching off there is no
      // end change, so the right-side point still awards A.
      for (final f in _doubleBounceOn(0.75, startT: 6000)) {
        mc.onFrame(f);
      }
      expect(mc.referee.leftPlayer, Player.a);
      expect(mc.score.pointsA, 1);
    });

    test('undo reverses a deciding-game mid-game end change', () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 2, bestOf: 3),
        switchEndsBetweenGames: true,
      );
      winGameOnRight(mc, startT: 0);
      winGameOnRight(mc, startT: 3000);
      expect(mc.referee.leftPlayer, Player.a);

      for (final f in _doubleBounceOn(0.75, startT: 6000)) {
        mc.onFrame(f);
      }
      expect(mc.referee.leftPlayer, Player.b); // mid-game switch fired

      mc.undo(); // undo the midpoint-crossing point
      expect(mc.score.pointsA, 0);
      expect(mc.referee.leftPlayer, Player.a); // switch reversed
    });
  });

  group('MatchController — auto-calibration', () {
    test('defers scoring during warm-up, then infers the table geometry', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );

      // First 7 warm-up frames: still calibrating, geometry is the full frame.
      final warm = _warmup();
      for (final f in warm.take(7)) {
        expect(mc.onFrame(f), isEmpty);
      }
      expect(mc.isCalibrating, isTrue);
      expect(mc.geometry.right, 1.0);

      // The 8th sample completes calibration.
      mc.onFrame(warm[7]);
      expect(mc.isCalibrating, isFalse);
      expect(mc.geometry.left, closeTo(0.23, 0.03));
      expect(mc.geometry.right, closeTo(0.73, 0.03));
      expect(mc.geometry.bottom, closeTo(0.54, 0.03));
      expect(mc.geometry.netX, closeTo(0.5, 0.03));

      // No points were scored during the warm-up phase.
      expect(mc.score.pointsA, 0);
      expect(mc.score.pointsB, 0);
    });

    test('after calibration, an on-surface double bounce scores normally', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );
      for (final f in _warmup()) {
        mc.onFrame(f);
      }
      // Apex y=0.52 sits inside the calibrated surface (bottom≈0.54); the
      // right-side double bounce awards the left player (A).
      List<PointDecision> last = const [];
      for (final f in _doubleBounceAt(0.65, 0.44, 0.52, startT: 1000)) {
        last = mc.onFrame(f);
      }
      expect(last.single.reason, PointReason.doubleBounce);
      expect(mc.score.pointsA, 1);
    });

    test('after calibration, an off-surface (floor) bounce is not scored', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );
      for (final f in _warmup()) {
        mc.onFrame(f);
      }
      // Apex y=0.90 is well below the calibrated table bottom (≈0.54): the
      // gated tracker drops it, so no point is awarded.
      final decisions = <PointDecision>[];
      for (final f in _doubleBounceAt(0.65, 0.80, 0.90, startT: 1000)) {
        decisions.addAll(mc.onFrame(f));
      }
      expect(decisions, isEmpty);
      expect(mc.score.pointsA, 0);
      expect(mc.score.pointsB, 0);
    });
  });

  group('MatchController — calibration progress / stall', () {
    test('progress rises with ball samples then reaches 1 once calibrated', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );
      final warm = _warmup();
      expect(mc.calibrationProgress, 0);

      mc.onFrame(warm[0]);
      mc.onFrame(warm[1]);
      expect(mc.calibrationFramesObserved, 2);
      expect(mc.calibrationProgress, closeTo(2 / 8, 1e-9));

      for (final f in warm) {
        mc.onFrame(f);
      }
      expect(mc.isCalibrating, isFalse);
      expect(mc.calibrationProgress, 1);
      // The frame counter freezes once scoring begins.
      final frozen = mc.calibrationFramesObserved;
      mc.onFrame(_empty(9999));
      expect(mc.calibrationFramesObserved, frozen);
    });

    test('flags a stall when the ball is never seen within the budget', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 20),
        calibrationStallFrames: 5,
      );
      // Ball-less frames: the calibrator never accumulates a sample.
      for (var i = 0; i < 4; i++) {
        mc.onFrame(_empty(i * 33));
      }
      expect(mc.isCalibrationStalled, isFalse);
      expect(mc.calibrationProgress, 0);

      mc.onFrame(_empty(4 * 33));
      expect(mc.calibrationFramesObserved, 5);
      expect(mc.isCalibrationStalled, isTrue);
      expect(mc.isCalibrating, isTrue);
    });

    test('a successful calibration is never reported as stalled', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
        calibrationStallFrames: 8,
      );
      for (final f in _warmup()) {
        mc.onFrame(f);
      }
      expect(mc.isCalibrating, isFalse);
      expect(mc.isCalibrationStalled, isFalse);
    });

    test('no calibrator: never calibrating, never stalled, full progress', () {
      final mc = MatchController();
      mc.onFrame(_empty(0));
      expect(mc.isCalibrating, isFalse);
      expect(mc.isCalibrationStalled, isFalse);
      expect(mc.calibrationProgress, 1);
      expect(mc.calibrationFramesObserved, 0);
    });
  });

  group('MatchController — player movement analytics', () {
    /// A frame with a left-side and right-side player whose box bottom-centre
    /// (their inferred foot) sits at the given x's.
    FrameResult playersFrame(int t, double leftX, double rightX) => FrameResult(
          timestampMs: t,
          people: [
            PersonPose(box: BBox(leftX - 0.025, 0.4, 0.05, 0.4), keypoints: const []),
            PersonPose(box: BBox(rightX - 0.025, 0.4, 0.05, 0.4), keypoints: const []),
          ],
        );

    test('accumulates per-player footwork through the live pipeline', () {
      final mc = MatchController();
      mc.onFrame(playersFrame(0, 0.20, 0.80));
      mc.onFrame(playersFrame(33, 0.30, 0.75));
      mc.onFrame(playersFrame(66, 0.30, 0.70));

      final a = mc.movementFor(Player.a); // left side
      final b = mc.movementFor(Player.b); // right side
      expect(a.framesTracked, 3);
      expect(b.framesTracked, 3);
      // Player A's foot moved 0.20→0.30→0.30 = 0.10 total.
      expect(a.distanceTravelled, closeTo(0.10, 1e-9));
      // Player B's foot moved 0.80→0.75→0.70 = 0.10 total.
      expect(b.distanceTravelled, closeTo(0.10, 1e-9));
      expect(a.wasTracked, isTrue);
    });

    test('movementJitterThreshold filters footwork jitter through the pipeline',
        () {
      final mc = MatchController(movementJitterThreshold: 0.02);
      // Player A's foot only wobbles within the deadband; B makes a real move.
      mc.onFrame(playersFrame(0, 0.200, 0.80));
      mc.onFrame(playersFrame(33, 0.205, 0.70));
      mc.onFrame(playersFrame(66, 0.198, 0.60));
      // A's sub-deadband jitter is dropped; B's 0.10 steps are counted.
      expect(mc.movementFor(Player.a).distanceTravelled, closeTo(0.0, 1e-9));
      expect(mc.movementFor(Player.b).distanceTravelled, closeTo(0.20, 1e-9));
    });

    test('movementJitterThreshold survives the calibration tracker rebuild', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
        movementJitterThreshold: 0.02,
      );
      for (final f in _warmup()) {
        mc.onFrame(f);
      }
      // Break continuity so the jitter frames below anchor on themselves rather
      // than on the warmup foot position.
      mc.onFrame(const FrameResult(timestampMs: 1000, people: []));
      // After calibration the movement analyzer is rebuilt on the inferred
      // geometry; the deadband must be preserved, so A's post-calibration
      // jitter still logs no distance.
      mc.onFrame(playersFrame(2000, 0.250, 0.75));
      mc.onFrame(playersFrame(2033, 0.255, 0.75));
      mc.onFrame(playersFrame(2066, 0.248, 0.75));
      expect(mc.movementFor(Player.a).distanceTravelled, closeTo(0.0, 1e-9));
    });

    test('movement side assignment follows the calibrated net line', () {
      final mc = MatchController(
        calibrator: TableCalibrator(minBallSamples: 8),
      );
      // The warm-up frames that only defer scoring record no movement; the one
      // that completes calibration is a live frame, so it counts (1 each).
      for (final f in _warmup()) {
        mc.onFrame(f);
      }
      expect(mc.movementFor(Player.a).framesTracked, 1);
      expect(mc.movementFor(Player.b).framesTracked, 1);

      // Post-calibration frames feed the (net≈0.5) analyzer: 0.25→left (A),
      // 0.75→right (B).
      mc.onFrame(playersFrame(2000, 0.25, 0.75));
      expect(mc.movementFor(Player.a).framesTracked, 2);
      expect(mc.movementFor(Player.b).framesTracked, 2);
      expect(mc.movementFor(Player.a).averageX, closeTo((0.245 + 0.25) / 2, 1e-9));
    });

    test('movement attribution flips with a between-games end change', () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 2, bestOf: 3),
        switchEndsBetweenGames: true,
      );
      // Game 1: A stands on the left (0.20), B on the right (0.80).
      mc.onFrame(playersFrame(0, 0.20, 0.80));
      for (final f in _doubleBounceOn(0.75, startT: 100)) {
        mc.onFrame(f);
      }
      for (final f in _doubleBounceOn(0.75, startT: 1100)) {
        mc.onFrame(f);
      }
      expect(mc.score.gamesA, 1);
      expect(mc.referee.leftPlayer, Player.b); // ends switched

      // Game 2: the players changed ends. The same physical positions now map to
      // the OTHER player — the person on the right (0.80) is A.
      mc.onFrame(playersFrame(3000, 0.20, 0.80));

      // A was seen once on the left (game 1) and once on the right (game 2), so
      // its coverage spans the whole table rather than pinning to one half.
      expect(mc.movementFor(Player.a).averageX, closeTo(0.50, 1e-9));
      expect(mc.movementFor(Player.b).averageX, closeTo(0.50, 1e-9));
    });
  });

  group('MatchController — first server', () {
    test('matchNotStarted allows setting the first server, then locks', () {
      final mc = MatchController();
      expect(mc.matchNotStarted, isTrue);
      expect(mc.setFirstServer(Player.b), isTrue);
      expect(mc.score.server, Player.b);
      expect(mc.score.initialServer, Player.b);

      mc.engine.awardPoint(Player.a);
      expect(mc.matchNotStarted, isFalse);
      expect(mc.setFirstServer(Player.a), isFalse);
      expect(mc.score.initialServer, Player.b);
    });
  });

  group('MatchController — manual point', () {
    test('awardManualPoint scores, logs with the manual reason, and undoes', () {
      final mc = MatchController();
      // Seed a real frame so the manual point is stamped in the frame clock.
      mc.onFrame(_frame(500, 0.40, 0.50));

      mc.awardManualPoint(Player.b);
      expect(mc.score.pointsB, 1);
      expect(mc.score.pointsA, 0);
      expect(mc.points.single.reason, PointReason.manual);
      expect(mc.points.single.winner, Player.b);
      expect(mc.points.single.timestampMs, 500);

      expect(mc.undo(), isTrue);
      expect(mc.score.pointsB, 0);
      expect(mc.points, isEmpty);
    });

    test('captures the current server and game index before awarding', () {
      final mc = MatchController();
      expect(mc.setFirstServer(Player.b), isTrue);

      mc.awardManualPoint(Player.a);
      // The point is credited to A but its rally was served by B (the serve
      // rotation only advances after the award).
      expect(mc.points.single.server, Player.b);
      expect(mc.points.single.gameIndex, 0);
      expect(mc.score.server, Player.b); // still B until it's held twice
    });

    test('is a no-op once the match is over', () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 1, bestOf: 1),
      );
      mc.awardManualPoint(Player.a);
      mc.awardManualPoint(Player.a); // 2-0 wins the (win-by-2) game and match
      expect(mc.score.isMatchOver, isTrue);
      final before = mc.points.length;

      mc.awardManualPoint(Player.b);
      expect(mc.points.length, before);
      expect(mc.score.pointsB, 0);
    });

    test('a manually-awarded game boundary switches ends when opted in', () {
      final mc = MatchController(
        engine: ScoringEngine(pointsPerGame: 2, bestOf: 3),
        switchEndsBetweenGames: true,
      );
      mc.awardManualPoint(Player.a);
      mc.awardManualPoint(Player.a); // completes game 1 for A
      expect(mc.score.gamesA, 1);
      // Players change ends after the game, flipping the side→player mapping.
      expect(mc.referee.leftPlayer, Player.b);
    });
  });
}
