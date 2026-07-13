import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/analysis/player_movement.dart';
import 'package:pong_ai/core/scoring/scoring_engine.dart';
import 'package:pong_ai/core/vision/detection.dart';

/// Builds a person with ankle keypoints at ([footX], [footY]) with an optional
/// ankle spread so [stanceWidthOf] can be exercised. Keypoints before the ankle
/// indices are filled with zero-confidence placeholders.
PersonPose person(
  double footX,
  double footY, {
  double stance = 0.0,
  double boxSize = 0.1,
}) {
  final kps = List<Keypoint>.generate(17, (_) => const Keypoint(0, 0, 0));
  kps[kLeftAnkleIndex] = Keypoint(footX - stance / 2, footY, 1);
  kps[kRightAnkleIndex] = Keypoint(footX + stance / 2, footY, 1);
  return PersonPose(
    box: BBox(footX - boxSize / 2, footY - boxSize, boxSize, boxSize),
    keypoints: kps,
  );
}

FrameResult frame(int ms, List<PersonPose> people) =>
    FrameResult(timestampMs: ms, people: people);

void main() {
  group('footOf', () {
    test('uses the midpoint of the visible ankles', () {
      final p = person(0.3, 0.8, stance: 0.1);
      final foot = footOf(p);
      expect(foot.x, closeTo(0.3, 1e-9));
      expect(foot.y, closeTo(0.8, 1e-9));
    });

    test('falls back to box bottom-centre when ankles are not visible', () {
      final kps = List<Keypoint>.generate(17, (_) => const Keypoint(0, 0, 0));
      final p = PersonPose(box: const BBox(0.2, 0.5, 0.1, 0.2), keypoints: kps);
      final foot = footOf(p);
      expect(foot.x, closeTo(0.25, 1e-9)); // centerX
      expect(foot.y, closeTo(0.7, 1e-9)); // top + height
    });

    test('averages a single visible ankle when only one is confident', () {
      final kps = List<Keypoint>.generate(17, (_) => const Keypoint(0, 0, 0));
      kps[kLeftAnkleIndex] = const Keypoint(0.4, 0.9, 1);
      // right ankle stays confidence 0 → ignored
      final p = PersonPose(box: const BBox(0, 0, 0.1, 0.1), keypoints: kps);
      final foot = footOf(p);
      expect(foot.x, closeTo(0.4, 1e-9));
      expect(foot.y, closeTo(0.9, 1e-9));
    });
  });

  group('stanceWidthOf', () {
    test('returns ankle separation when both are visible', () {
      expect(stanceWidthOf(person(0.5, 0.8, stance: 0.12)), closeTo(0.12, 1e-9));
    });

    test('returns null when an ankle is occluded', () {
      final kps = List<Keypoint>.generate(17, (_) => const Keypoint(0, 0, 0));
      kps[kLeftAnkleIndex] = const Keypoint(0.4, 0.9, 1);
      final p = PersonPose(box: const BBox(0, 0, 0.1, 0.1), keypoints: kps);
      expect(stanceWidthOf(p), isNull);
    });
  });

  group('PlayerMovementAnalyzer', () {
    test('attributes players by side of the net', () {
      final a = PlayerMovementAnalyzer(); // leftPlayer defaults to A
      // Left-of-net foot → Player A, right-of-net → Player B.
      a.observe(frame(0, [person(0.2, 0.8), person(0.8, 0.8)]));
      expect(a.statsFor(Player.a).framesTracked, 1);
      expect(a.statsFor(Player.b).framesTracked, 1);
      expect(a.statsFor(Player.a).averageX, closeTo(0.2, 1e-9));
      expect(a.statsFor(Player.b).averageX, closeTo(0.8, 1e-9));
    });

    test('respects a mirrored leftPlayer mapping', () {
      final a = PlayerMovementAnalyzer(leftPlayer: Player.b);
      a.observe(frame(0, [person(0.2, 0.8)]));
      expect(a.statsFor(Player.b).framesTracked, 1);
      expect(a.statsFor(Player.a).framesTracked, 0);
    });

    test('accumulates path length across consecutive frames', () {
      final a = PlayerMovementAnalyzer();
      // Player A's feet slide 0.1 right, then 0.1 right again → distance 0.2.
      a.observe(frame(0, [person(0.20, 0.80)]));
      a.observe(frame(33, [person(0.30, 0.80)]));
      a.observe(frame(66, [person(0.40, 0.80)]));
      final s = a.statsFor(Player.a);
      expect(s.framesTracked, 3);
      expect(s.distanceTravelled, closeTo(0.2, 1e-9));
      expect(s.coverageWidth, closeTo(0.2, 1e-9));
      expect(s.coverageDepth, closeTo(0.0, 1e-9));
    });

    test('does not add a teleport jump across a detection gap', () {
      final a = PlayerMovementAnalyzer();
      a.observe(frame(0, [person(0.20, 0.80)]));
      a.observe(frame(33, [])); // player A missing this frame
      a.observe(frame(66, [person(0.40, 0.80)])); // reappears far away
      final s = a.statsFor(Player.a);
      expect(s.framesTracked, 2);
      // The 0.20→0.40 jump straddles a gap, so it is NOT counted as movement.
      expect(s.distanceTravelled, closeTo(0.0, 1e-9));
      // Coverage still reflects both observed positions.
      expect(s.coverageWidth, closeTo(0.2, 1e-9));
    });

    test('computes mobility per second from tracked time span', () {
      final a = PlayerMovementAnalyzer();
      a.observe(frame(0, [person(0.20, 0.80)]));
      a.observe(frame(1000, [person(0.45, 0.80)])); // 0.25 over 1.0s
      final s = a.statsFor(Player.a);
      expect(s.distanceTravelled, closeTo(0.25, 1e-9));
      expect(s.trackedMs, 1000);
      expect(s.mobilityPerSecond, closeTo(0.25, 1e-9));
    });

    test('averages stance width only over frames where both ankles are seen',
        () {
      final a = PlayerMovementAnalyzer();
      a.observe(frame(0, [person(0.30, 0.80, stance: 0.10)]));
      // occluded-feet frame contributes position but no stance
      final kps = List<Keypoint>.generate(17, (_) => const Keypoint(0, 0, 0));
      a.observe(frame(33, [
        PersonPose(box: const BBox(0.25, 0.7, 0.1, 0.1), keypoints: kps),
      ],),);
      a.observe(frame(66, [person(0.30, 0.80, stance: 0.20)]));
      final s = a.statsFor(Player.a);
      expect(s.framesTracked, 3);
      expect(s.averageStanceWidth, closeTo(0.15, 1e-9)); // (0.10+0.20)/2
    });

    test('reports null stance when feet are never both visible', () {
      final a = PlayerMovementAnalyzer();
      final kps = List<Keypoint>.generate(17, (_) => const Keypoint(0, 0, 0));
      a.observe(frame(0, [
        PersonPose(box: const BBox(0.2, 0.7, 0.1, 0.1), keypoints: kps),
      ],),);
      expect(a.statsFor(Player.a).averageStanceWidth, isNull);
    });

    test('ignores a second same-side detection in one frame', () {
      final a = PlayerMovementAnalyzer();
      // Two left-side people in one frame: only the first counts for Player A.
      a.observe(frame(0, [person(0.20, 0.80), person(0.35, 0.80)]));
      a.observe(frame(33, [person(0.25, 0.80)]));
      final s = a.statsFor(Player.a);
      expect(s.framesTracked, 2);
      // distance uses the first detection (0.20) → 0.25 = 0.05, not the 0.35 dup.
      expect(s.distanceTravelled, closeTo(0.05, 1e-9));
    });

    test('untracked player yields a zeroed, describable stats object', () {
      final a = PlayerMovementAnalyzer();
      a.observe(frame(0, [person(0.20, 0.80)])); // only left side seen
      final b = a.statsFor(Player.b);
      expect(b.wasTracked, isFalse);
      expect(b.framesTracked, 0);
      expect(b.distanceTravelled, 0);
      expect(b.mobilityPerSecond, 0);
      expect(b.coverageArea, 0);
      expect(b.describe(), 'not tracked');
    });

    test('honours a calibrated non-centre net line when assigning sides', () {
      // Net shifted left to 0.3: a foot at 0.4 is now on the right (Player B).
      final a = PlayerMovementAnalyzer(
        geometry: const TableGeometry(netX: 0.3, left: 0.05, right: 0.95),
      );
      a.observe(frame(0, [person(0.40, 0.80)]));
      expect(a.statsFor(Player.b).framesTracked, 1);
      expect(a.statsFor(Player.a).framesTracked, 0);
    });

    test('reset clears all accumulated movement', () {
      final a = PlayerMovementAnalyzer();
      a.observe(frame(0, [person(0.20, 0.80)]));
      a.reset();
      expect(a.statsFor(Player.a).framesTracked, 0);
      expect(a.statsFor(Player.a).distanceTravelled, 0);
    });

    test('coverageArea multiplies width and depth spans', () {
      final a = PlayerMovementAnalyzer();
      a.observe(frame(0, [person(0.20, 0.60)]));
      a.observe(frame(33, [person(0.40, 0.80)]));
      final s = a.statsFor(Player.a);
      expect(s.coverageWidth, closeTo(0.2, 1e-9));
      expect(s.coverageDepth, closeTo(0.2, 1e-9));
      expect(s.coverageArea, closeTo(0.04, 1e-9));
    });

    test('switchEnds flips which player each half is attributed to', () {
      final a = PlayerMovementAnalyzer(); // A on left, B on right
      a.observe(frame(0, [person(0.20, 0.80), person(0.80, 0.80)]));
      expect(a.leftPlayer, Player.a);
      expect(a.statsFor(Player.a).averageX, closeTo(0.20, 1e-9));

      // The players change ends: the person now on the left is B.
      a.switchEnds();
      expect(a.leftPlayer, Player.b);
      a.observe(frame(33, [person(0.20, 0.80), person(0.80, 0.80)]));

      // A was seen once on the left (0.20) and once on the right (0.80).
      expect(a.statsFor(Player.a).framesTracked, 2);
      expect(a.statsFor(Player.a).averageX, closeTo(0.50, 1e-9));
      expect(a.statsFor(Player.b).averageX, closeTo(0.50, 1e-9));
    });

    test('switchEnds breaks distance continuity so the cross-court walk is not '
        'a teleport jump', () {
      final a = PlayerMovementAnalyzer();
      a.observe(frame(0, [person(0.20, 0.80)])); // A on left
      a.switchEnds(); // A walks to the right end
      a.observe(frame(33, [person(0.80, 0.80)])); // A now on the right
      // Without the continuity break this 0.20→0.80 move would log a 0.60 jump.
      expect(a.statsFor(Player.a).framesTracked, 2);
      expect(a.statsFor(Player.a).distanceTravelled, closeTo(0.0, 1e-9));
      expect(a.statsFor(Player.a).coverageWidth, closeTo(0.60, 1e-9));
    });

    test('an even number of switchEnds restores the original mapping', () {
      final a = PlayerMovementAnalyzer();
      a.switchEnds();
      a.switchEnds();
      expect(a.leftPlayer, Player.a);
    });
  });

  group('PlayerMovementAnalyzer jitter deadband (minStep)', () {
    test('a stationary player wobbling within minStep logs no distance', () {
      final a = PlayerMovementAnalyzer(minStep: 0.02);
      // Feet jitter a few thousandths around 0.20 every frame — pure noise.
      a.observe(frame(0, [person(0.200, 0.800)]));
      a.observe(frame(33, [person(0.205, 0.798)]));
      a.observe(frame(66, [person(0.198, 0.803)]));
      a.observe(frame(99, [person(0.203, 0.799)]));
      final s = a.statsFor(Player.a);
      expect(s.framesTracked, 4);
      // Every step stayed inside the 0.02 deadband, so nothing accumulates.
      expect(s.distanceTravelled, closeTo(0.0, 1e-9));
    });

    test('without the deadband the same jitter inflates the distance', () {
      final a = PlayerMovementAnalyzer(); // minStep defaults to 0
      a.observe(frame(0, [person(0.200, 0.800)]));
      a.observe(frame(33, [person(0.205, 0.798)]));
      a.observe(frame(66, [person(0.198, 0.803)]));
      a.observe(frame(99, [person(0.203, 0.799)]));
      // Raw per-frame summing counts the noise as real movement.
      expect(a.statsFor(Player.a).distanceTravelled, greaterThan(0.0));
    });

    test('real movement past the deadband is still counted in full', () {
      final a = PlayerMovementAnalyzer(minStep: 0.02);
      // Two genuine 0.10 steps, each well beyond the deadband → distance 0.20.
      a.observe(frame(0, [person(0.20, 0.80)]));
      a.observe(frame(33, [person(0.30, 0.80)]));
      a.observe(frame(66, [person(0.40, 0.80)]));
      expect(a.statsFor(Player.a).distanceTravelled, closeTo(0.20, 1e-9));
    });

    test('slow steady drift crosses the deadband and is not lost', () {
      final a = PlayerMovementAnalyzer(minStep: 0.02);
      // Each frame drifts 0.015 (< 0.02), but the anchor stays put until the
      // cumulative drift crosses 0.02, so the real travel isn't silently dropped.
      a.observe(frame(0, [person(0.200, 0.80)]));
      a.observe(frame(33, [person(0.215, 0.80)])); // 0.015 from anchor — held
      a.observe(frame(66, [person(0.230, 0.80)])); // 0.030 from anchor — counted
      a.observe(frame(99, [person(0.245, 0.80)])); // 0.015 from new anchor — held
      a.observe(frame(132, [person(0.260, 0.80)])); // 0.030 — counted
      // Two counted 0.03 chunks; the total is close to the 0.06 truly travelled.
      expect(a.statsFor(Player.a).distanceTravelled, closeTo(0.06, 1e-9));
    });
  });
}
