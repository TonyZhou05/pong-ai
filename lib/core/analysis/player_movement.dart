/// Pure-Dart player movement / footwork analytics from the pose model.
///
/// The vision pipeline already produces a [PersonPose] per detected player each
/// frame (box + 17 COCO keypoints), but until now nothing consumed the *pose*
/// for performance analysis — the scoring/summary layers only used the ball
/// trajectory and the referee's verdicts. The objective explicitly prioritises
/// tracking the *players* well and analysing their performance, so this layer
/// mines the per-frame player positions into footwork metrics: how far a player
/// travelled, how much of the court they covered, how mobile they were, and how
/// wide/ready their stance was.
///
/// It has no Flutter or plugin dependencies — it consumes the runtime-agnostic
/// [FrameResult]s, so it can be unit-tested against synthetic frames and driven
/// by replayed benchmark clips just like the rest of `core/analysis`.
///
/// Coordinate convention matches the rest of the pipeline: the frame is
/// normalized to `[0,1] x [0,1]` with the net a vertical line at
/// [TableGeometry.netX], so a player's side is inferred from where their feet
/// are relative to the net.
library;

import 'dart:math' as math;

import '../scoring/scoring_engine.dart';
import '../vision/detection.dart';
import 'ball_tracker.dart';

/// COCO 17-keypoint ankle indices (the feet), used to locate a player's stance.
const int kLeftAnkleIndex = 15;
const int kRightAnkleIndex = 16;

/// A normalized point in the frame.
typedef FramePoint = ({double x, double y});

/// The estimated foot position of a detected player.
///
/// Prefers the midpoint of the visible ankle keypoints (confidence > 0); falls
/// back to the bottom-centre of the bounding box when no ankle is visible (e.g.
/// occluded feet or a box-only detection).
FramePoint footOf(PersonPose pose) {
  final kps = pose.keypoints;
  final visible = <Keypoint>[];
  if (kps.length > kLeftAnkleIndex && kps[kLeftAnkleIndex].confidence > 0) {
    visible.add(kps[kLeftAnkleIndex]);
  }
  if (kps.length > kRightAnkleIndex && kps[kRightAnkleIndex].confidence > 0) {
    visible.add(kps[kRightAnkleIndex]);
  }
  if (visible.isNotEmpty) {
    final x = visible.map((k) => k.x).reduce((a, b) => a + b) / visible.length;
    final y = visible.map((k) => k.y).reduce((a, b) => a + b) / visible.length;
    return (x: x, y: y);
  }
  return (x: pose.box.centerX, y: pose.box.top + pose.box.height);
}

/// Horizontal distance between the two ankles, or null when both aren't visible.
double? stanceWidthOf(PersonPose pose) {
  final kps = pose.keypoints;
  if (kps.length <= kRightAnkleIndex) return null;
  final la = kps[kLeftAnkleIndex];
  final ra = kps[kRightAnkleIndex];
  if (la.confidence <= 0 || ra.confidence <= 0) return null;
  return (la.x - ra.x).abs();
}

/// Footwork / positioning metrics for one player over a session.
class PlayerMovementStats {
  const PlayerMovementStats({
    required this.framesTracked,
    required this.distanceTravelled,
    required this.coverageWidth,
    required this.coverageDepth,
    required this.averageX,
    required this.averageY,
    required this.averageStanceWidth,
    required this.trackedMs,
  });

  /// Number of frames the player was detected in.
  final int framesTracked;

  /// Total normalized path length of the player's feet across the session.
  final double distanceTravelled;

  /// Span of x the player's feet covered (max − min), a lateral coverage proxy.
  final double coverageWidth;

  /// Span of y the player's feet covered (max − min), a depth coverage proxy.
  final double coverageDepth;

  /// Mean foot x over all tracked frames (their "home" lateral position).
  final double averageX;

  /// Mean foot y over all tracked frames.
  final double averageY;

  /// Mean ankle separation over frames where both ankles were visible, or null
  /// when the feet were never both seen (a readiness/stance-width proxy).
  final double? averageStanceWidth;

  /// Time span (ms) between the player's first and last detection.
  final int trackedMs;

  /// Whether the player was seen at all.
  bool get wasTracked => framesTracked > 0;

  /// Distance travelled per second of tracked time (movement intensity). Zero
  /// when the player spanned no time (0 or 1 frames).
  double get mobilityPerSecond =>
      trackedMs > 0 ? distanceTravelled / (trackedMs / 1000) : 0;

  /// A coarse rectangular court-coverage area (width × depth span).
  double get coverageArea => coverageWidth * coverageDepth;

  /// One-line human-readable summary.
  String describe() {
    if (!wasTracked) return 'not tracked';
    final stance = averageStanceWidth == null
        ? 'stance n/a'
        : 'stance ${averageStanceWidth!.toStringAsFixed(2)}';
    return 'moved ${distanceTravelled.toStringAsFixed(2)} '
        '(${mobilityPerSecond.toStringAsFixed(2)}/s), '
        'coverage ${coverageWidth.toStringAsFixed(2)}×'
        '${coverageDepth.toStringAsFixed(2)}, $stance';
  }
}

/// Accumulates player movement across frames and reports [PlayerMovementStats].
///
/// Detected people are attributed to [Player.a] / [Player.b] by which side of
/// the net their feet fall on, using the same left-player convention as
/// [RallyReferee]. Distance is accumulated only between *consecutive* frames a
/// player was seen in, so a detection gap doesn't register a teleport jump.
class PlayerMovementAnalyzer {
  PlayerMovementAnalyzer({
    this.geometry = const TableGeometry(),
    Player leftPlayer = Player.a,
  }) : _leftPlayer = leftPlayer;

  /// Table geometry, whose [TableGeometry.netX] splits the two players' sides.
  final TableGeometry geometry;

  /// Which player occupies the left half of the table (net-split). Mirrors
  /// [RallyReferee]'s mapping so movement stats and scoring agree on identities.
  final Player _leftPlayer;

  final Map<Player, _Accumulator> _acc = {
    Player.a: _Accumulator(),
    Player.b: _Accumulator(),
  };

  Player _playerForFoot(FramePoint foot) =>
      geometry.sideOf(foot.x) == TableSide.left ? _leftPlayer : _leftPlayer.other;

  /// Fold one frame's detected players into the running movement stats.
  void observe(FrameResult frame) {
    final seen = <Player>{};
    for (final pose in frame.people) {
      final foot = footOf(pose);
      final player = _playerForFoot(foot);
      // At most one detection per side per frame; if two people map to the same
      // side (spurious extra box) keep the first and ignore the rest so distance
      // isn't corrupted by flip-flopping between two same-side detections.
      if (!seen.add(player)) continue;
      _acc[player]!.add(frame.timestampMs, foot, stanceWidthOf(pose));
    }
    // A player missing this frame breaks distance continuity, so their next
    // appearance doesn't add a spurious jump across the gap.
    for (final player in Player.values) {
      if (!seen.contains(player)) _acc[player]!.breakContinuity();
    }
  }

  /// Movement metrics for [player] so far.
  PlayerMovementStats statsFor(Player player) => _acc[player]!.build();

  /// Forget all accumulated movement.
  void reset() {
    for (final a in _acc.values) {
      a.reset();
    }
  }
}

class _Accumulator {
  int frames = 0;
  double distance = 0;
  double sumX = 0;
  double sumY = 0;
  double? minX, maxX, minY, maxY;
  int? firstMs, lastMs;
  double stanceSum = 0;
  int stanceCount = 0;
  FramePoint? _prev;
  bool _continuous = false;

  void add(int ms, FramePoint foot, double? stance) {
    frames++;
    sumX += foot.x;
    sumY += foot.y;
    minX = minX == null ? foot.x : math.min(minX!, foot.x);
    maxX = maxX == null ? foot.x : math.max(maxX!, foot.x);
    minY = minY == null ? foot.y : math.min(minY!, foot.y);
    maxY = maxY == null ? foot.y : math.max(maxY!, foot.y);
    if (_continuous && _prev != null) {
      final dx = foot.x - _prev!.x;
      final dy = foot.y - _prev!.y;
      distance += math.sqrt(dx * dx + dy * dy);
    }
    _prev = foot;
    _continuous = true;
    firstMs ??= ms;
    lastMs = ms;
    if (stance != null) {
      stanceSum += stance;
      stanceCount++;
    }
  }

  void breakContinuity() => _continuous = false;

  PlayerMovementStats build() {
    return PlayerMovementStats(
      framesTracked: frames,
      distanceTravelled: distance,
      coverageWidth: (minX == null) ? 0 : (maxX! - minX!),
      coverageDepth: (minY == null) ? 0 : (maxY! - minY!),
      averageX: frames == 0 ? 0 : sumX / frames,
      averageY: frames == 0 ? 0 : sumY / frames,
      averageStanceWidth: stanceCount == 0 ? null : stanceSum / stanceCount,
      trackedMs: (firstMs == null || lastMs == null) ? 0 : lastMs! - firstMs!,
    );
  }

  void reset() {
    frames = 0;
    distance = 0;
    sumX = 0;
    sumY = 0;
    minX = maxX = minY = maxY = null;
    firstMs = lastMs = null;
    stanceSum = 0;
    stanceCount = 0;
    _prev = null;
    _continuous = false;
  }
}
