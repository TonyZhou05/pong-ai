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
  });
}
