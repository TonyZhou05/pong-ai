/// Converts OpenTTGames dataset annotations into the benchmark harness's
/// [ClipFixture] / [FrameResult] models.
///
/// The objective asks us to "use various resources to find videos to benchmark
/// on," and docs/ARCHITECTURE.md §2 names **OpenTTGames** (ball position +
/// bounce/net/empty event labels) as a primary source. Until now that was only
/// a documented *conversion path* in `benchmark/README.md` — turning a real clip
/// into a fixture still meant hand-writing JSON. This module implements the
/// conversion in pure Dart so an OpenTTGames game folder plugs straight into the
/// perception benchmark ([DetectionBenchmark]).
///
/// OpenTTGames ships, per game, a `ball_markup.json` mapping a frame index to
/// the ball centre in **pixel** coordinates:
///
/// ```jsonc
/// { "144": { "x": 921, "y": 507 }, "145": { "x": 934, "y": 498 }, ... }
/// ```
///
/// (frames where the ball is absent are either omitted or carry a negative
/// coordinate). These are *ground-truth* ball positions, so they populate a
/// fixture's [ClipFixture.groundTruthFrames]; feed your model's per-frame
/// predictions in as `predictedFrames` to score detection precision/recall
/// against them, or leave them off for a perfect-detector baseline.
library;

import '../scoring/scoring_engine.dart';
import '../vision/detection.dart';
import 'clip_fixture.dart';

/// Default normalized side length of the synthesized ball box. OpenTTGames only
/// labels the ball *centre*, so the benchmark's IoU needs a nominal box; a real
/// 40 mm ball is a handful of pixels, i.e. a couple of percent of the frame.
const double kOpenTtGamesBallSize = 0.03;

/// Builds the ground-truth [FrameResult] stream from an OpenTTGames
/// `ball_markup.json` map.
///
/// [ballMarkup] keys are frame-index strings and values are `{x, y}` maps in
/// pixel space; [frameWidth]/[frameHeight] are the source video's pixel
/// dimensions (used to normalize to `[0, 1]`); [fps] converts a frame index into
/// the millisecond timestamp the tracker uses for velocity. Frames are returned
/// in ascending frame-index order with a synthesized [kOpenTtGamesBallSize] box.
/// A frame whose coordinate is missing or negative yields a ball-less
/// [FrameResult] (a true "ball absent" ground truth).
List<FrameResult> openTtGamesGroundTruthFrames({
  required Map<String, dynamic> ballMarkup,
  required int frameWidth,
  required int frameHeight,
  required double fps,
  double ballSize = kOpenTtGamesBallSize,
}) {
  assert(frameWidth > 0, 'frameWidth must be positive');
  assert(frameHeight > 0, 'frameHeight must be positive');
  assert(fps > 0, 'fps must be positive');
  assert(ballSize > 0 && ballSize < 1, 'ballSize must be in (0, 1)');

  final indices = ballMarkup.keys
      .map(int.tryParse)
      .whereType<int>()
      .toList(growable: false)
    ..sort();

  final frames = <FrameResult>[];
  for (final idx in indices) {
    final raw = ballMarkup['$idx'];
    Detection? ball;
    if (raw is Map) {
      final px = (raw['x'] as num?)?.toDouble();
      final py = (raw['y'] as num?)?.toDouble();
      if (px != null && py != null && px >= 0 && py >= 0) {
        ball = _ballAt(px / frameWidth, py / frameHeight, ballSize);
      }
    }
    frames.add(
      FrameResult(
        timestampMs: (idx * 1000 / fps).round(),
        ball: ball,
      ),
    );
  }
  return frames;
}

/// Builds a full [ClipFixture] from an OpenTTGames `ball_markup.json` map.
///
/// The labeled ball positions become [ClipFixture.groundTruthFrames]. Pass your
/// model's per-frame output as [predictedFrames] to score perception against
/// that ground truth; when omitted, the ground-truth frames double as the
/// predictions (a perfect-detector baseline that still exercises the
/// tracker/scoring pipeline). OpenTTGames carries no scoreboard, so supply the
/// rally outcome via [groundTruth] if you scored the clip by hand.
ClipFixture clipFixtureFromOpenTtGames({
  required String name,
  required Map<String, dynamic> ballMarkup,
  required int frameWidth,
  required int frameHeight,
  required double fps,
  List<FrameResult>? predictedFrames,
  ClipGroundTruth groundTruth = const ClipGroundTruth(pointsA: 0, pointsB: 0),
  String source = 'OpenTTGames',
  double netX = 0.5,
  Player leftPlayer = Player.a,
  Player firstServer = Player.a,
  int pointsPerGame = 11,
  int bestOf = 5,
  double ballSize = kOpenTtGamesBallSize,
}) {
  final gtFrames = openTtGamesGroundTruthFrames(
    ballMarkup: ballMarkup,
    frameWidth: frameWidth,
    frameHeight: frameHeight,
    fps: fps,
    ballSize: ballSize,
  );
  return ClipFixture(
    name: name,
    source: source,
    fps: fps.round(),
    netX: netX,
    leftPlayer: leftPlayer,
    firstServer: firstServer,
    pointsPerGame: pointsPerGame,
    bestOf: bestOf,
    frames: predictedFrames ?? gtFrames,
    groundTruthFrames: gtFrames,
    groundTruth: groundTruth,
  );
}

/// A normalized ball [Detection] centred on ([cx], [cy]) with a [size]-wide box,
/// clamped so the box stays inside the frame.
Detection _ballAt(double cx, double cy, double size) {
  final left = (cx - size / 2).clamp(0.0, 1.0 - size).toDouble();
  final top = (cy - size / 2).clamp(0.0, 1.0 - size).toDouble();
  return Detection(
    label: 'ball',
    confidence: 1.0,
    box: BBox(left, top, size, size),
  );
}
