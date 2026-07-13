/// A labeled evaluation clip for the offline benchmark harness.
///
/// The objective asks us to "use various resources to find videos to benchmark
/// on." A [ClipFixture] is the portable, on-disk representation of one such
/// clip: a sequence of per-frame vision detections (exactly what the live
/// `ultralytics_yolo` runtime emits, in the runtime-agnostic [FrameResult]
/// form) plus the *ground truth* of how the rally(s) actually scored.
///
/// Fixtures are JSON so annotations can be authored externally — e.g. converted
/// from OpenTTGames ball-position + event labels, or hand-labeled from a
/// side-angle YouTube clip — and dropped into `benchmark/clips/` without any
/// code change. Feeding a fixture through [BenchmarkRunner] runs the *same*
/// tracker → referee → scoring pipeline the live camera uses, so scoring
/// accuracy can be measured objectively and compared across model swaps.
library;

import '../analysis/ball_tracker.dart';
import '../scoring/scoring_engine.dart';
import '../vision/detection.dart';
import 'event_metrics.dart';

/// The verified scoring outcome of a clip, used to score the pipeline against.
class ClipGroundTruth {
  const ClipGroundTruth({
    required this.pointsA,
    required this.pointsB,
    this.pointWinners,
  });

  /// Points player A / B actually won over the clip.
  final int pointsA;
  final int pointsB;

  /// Optional ordered list of the true point winners, when the annotation
  /// records rally-by-rally outcomes. Enables ordered point-level accuracy on
  /// top of the aggregate totals.
  final List<Player>? pointWinners;

  factory ClipGroundTruth.fromJson(Map<String, dynamic> json) {
    final winners = json['pointWinners'] as List<dynamic>?;
    return ClipGroundTruth(
      pointsA: (json['pointsA'] as num).toInt(),
      pointsB: (json['pointsB'] as num).toInt(),
      pointWinners: winners
          ?.map((w) => _playerFromString(w as String))
          .toList(growable: false),
    );
  }

  Map<String, dynamic> toJson() => {
        'pointsA': pointsA,
        'pointsB': pointsB,
        if (pointWinners != null)
          'pointWinners': pointWinners!.map((p) => p.name).toList(),
      };
}

/// A benchmark clip: how to configure the pipeline, the frames to replay, and
/// the ground-truth outcome to score against.
class ClipFixture {
  const ClipFixture({
    required this.name,
    required this.frames,
    required this.groundTruth,
    this.groundTruthFrames,
    this.groundTruthEvents,
    this.source = 'unknown',
    this.fps = 30,
    this.netX = 0.5,
    this.tableLeft,
    this.tableRight,
    this.tableTop,
    this.tableBottom,
    this.leftPlayer = Player.a,
    this.firstServer = Player.a,
    this.pointsPerGame = 11,
    this.bestOf = 5,
  });

  /// Short identifier for the clip (used in reports).
  final String name;

  /// Where the clip came from (dataset name, URL, "synthetic", ...).
  final String source;

  /// Nominal capture frame rate (documentation only; the tracker uses each
  /// frame's own timestamp for its velocity maths).
  final int fps;

  /// Normalized x of the net line — the table geometry for this clip's camera
  /// placement.
  final double netX;

  /// Optional normalized bounds of the table *surface* region in the frame.
  /// When present, the pipeline should gate bounces to this band (a direction
  /// change outside it — a paddle hit, or the ball beyond the table's edge —
  /// is not a table bounce). Null means unknown: the full frame is assumed,
  /// preserving the pre-existing fixture behaviour.
  final double? tableLeft;
  final double? tableRight;
  final double? tableTop;
  final double? tableBottom;

  /// The [TableGeometry] this clip's camera placement implies: the calibrated
  /// net line plus the surface band when annotated (full frame otherwise).
  TableGeometry get geometry => TableGeometry(
        netX: netX,
        left: tableLeft ?? 0.0,
        right: tableRight ?? 1.0,
        top: tableTop ?? 0.0,
        bottom: tableBottom ?? 1.0,
      );

  /// Which player occupies the left half of the frame.
  final Player leftPlayer;

  /// Match configuration to reproduce the clip's scoring context.
  final Player firstServer;
  final int pointsPerGame;
  final int bestOf;

  /// The replayed vision frames, in timestamp order. These are the pipeline's
  /// *predicted* detections (what the model output / what we replay).
  final List<FrameResult> frames;

  /// Optional per-frame *ground-truth* detections, index-aligned with [frames],
  /// used by the perception benchmark ([DetectionBenchmark]) to score ball/pose
  /// accuracy. Null when the clip only carries a scoring outcome.
  final List<FrameResult>? groundTruthFrames;

  /// Optional ground-truth *event* timings (table bounces / net crossings) used
  /// by the event-detection benchmark ([EventDetectionBenchmark]) to score the
  /// tracker's bounce/net-cross timing. Null when the clip carries no event
  /// labels. Populated from OpenTTGames `events_markup.json` via
  /// `openTtGamesBounceEvents`.
  final List<GroundTruthEvent>? groundTruthEvents;

  final ClipGroundTruth groundTruth;

  factory ClipFixture.fromJson(Map<String, dynamic> json) {
    final rawFrames = (json['frames'] as List<dynamic>? ?? const []);
    final rawGtFrames = json['groundTruthFrames'] as List<dynamic>?;
    final rawGtEvents = json['groundTruthEvents'] as List<dynamic>?;
    return ClipFixture(
      name: json['name'] as String? ?? 'unnamed',
      source: json['source'] as String? ?? 'unknown',
      fps: (json['fps'] as num?)?.toInt() ?? 30,
      netX: (json['netX'] as num?)?.toDouble() ?? 0.5,
      tableLeft: (json['tableLeft'] as num?)?.toDouble(),
      tableRight: (json['tableRight'] as num?)?.toDouble(),
      tableTop: (json['tableTop'] as num?)?.toDouble(),
      tableBottom: (json['tableBottom'] as num?)?.toDouble(),
      leftPlayer: _playerFromString(json['leftPlayer'] as String? ?? 'a'),
      firstServer: _playerFromString(json['firstServer'] as String? ?? 'a'),
      pointsPerGame: (json['pointsPerGame'] as num?)?.toInt() ?? 11,
      bestOf: (json['bestOf'] as num?)?.toInt() ?? 5,
      frames: rawFrames
          .map((f) => _frameFromJson(f as Map<String, dynamic>))
          .toList(growable: false),
      groundTruthFrames: rawGtFrames
          ?.map((f) => _frameFromJson(f as Map<String, dynamic>))
          .toList(growable: false),
      groundTruthEvents: rawGtEvents
          ?.map((e) => GroundTruthEvent.fromJson(e as Map<String, dynamic>))
          .toList(growable: false),
      groundTruth: ClipGroundTruth.fromJson(
        json['groundTruth'] as Map<String, dynamic>,
      ),
    );
  }

  Map<String, dynamic> toJson() => {
        'name': name,
        'source': source,
        'fps': fps,
        'netX': netX,
        if (tableLeft != null) 'tableLeft': tableLeft,
        if (tableRight != null) 'tableRight': tableRight,
        if (tableTop != null) 'tableTop': tableTop,
        if (tableBottom != null) 'tableBottom': tableBottom,
        'leftPlayer': leftPlayer.name,
        'firstServer': firstServer.name,
        'pointsPerGame': pointsPerGame,
        'bestOf': bestOf,
        'groundTruth': groundTruth.toJson(),
        'frames': frames.map(_frameToJson).toList(),
        if (groundTruthFrames != null)
          'groundTruthFrames':
              groundTruthFrames!.map(_frameToJson).toList(),
        if (groundTruthEvents != null)
          'groundTruthEvents':
              groundTruthEvents!.map((e) => e.toJson()).toList(),
      };
}

Player _playerFromString(String s) =>
    s.toLowerCase() == 'b' ? Player.b : Player.a;

FrameResult _frameFromJson(Map<String, dynamic> json) {
  final ballJson = json['ball'] as Map<String, dynamic>?;
  final peopleJson = json['people'] as List<dynamic>? ?? const [];
  return FrameResult(
    timestampMs: (json['t'] as num).toInt(),
    fps: (json['fps'] as num?)?.toDouble(),
    ball: ballJson == null ? null : _detectionFromJson(ballJson),
    people: peopleJson
        .map((p) => _poseFromJson(p as Map<String, dynamic>))
        .toList(growable: false),
  );
}

Map<String, dynamic> _frameToJson(FrameResult f) => {
      't': f.timestampMs,
      if (f.fps != null) 'fps': f.fps,
      if (f.ball != null) 'ball': _detectionToJson(f.ball!),
      if (f.people.isNotEmpty)
        'people': f.people.map(_poseToJson).toList(),
    };

BBox _boxFromJson(List<dynamic> b) => BBox(
      (b[0] as num).toDouble(),
      (b[1] as num).toDouble(),
      (b[2] as num).toDouble(),
      (b[3] as num).toDouble(),
    );

List<double> _boxToJson(BBox b) => [b.left, b.top, b.width, b.height];

Detection _detectionFromJson(Map<String, dynamic> json) => Detection(
      label: json['label'] as String? ?? 'ball',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 1.0,
      box: _boxFromJson(json['box'] as List<dynamic>),
    );

Map<String, dynamic> _detectionToJson(Detection d) => {
      'label': d.label,
      'confidence': d.confidence,
      'box': _boxToJson(d.box),
    };

PersonPose _poseFromJson(Map<String, dynamic> json) {
  final kps = json['keypoints'] as List<dynamic>? ?? const [];
  return PersonPose(
    box: _boxFromJson(json['box'] as List<dynamic>),
    trackId: (json['trackId'] as num?)?.toInt(),
    keypoints: kps
        .map((k) => _keypointFromJson(k as List<dynamic>))
        .toList(growable: false),
  );
}

Map<String, dynamic> _poseToJson(PersonPose p) => {
      'box': _boxToJson(p.box),
      if (p.trackId != null) 'trackId': p.trackId,
      'keypoints': p.keypoints.map(_keypointToJson).toList(),
    };

Keypoint _keypointFromJson(List<dynamic> k) => Keypoint(
      (k[0] as num).toDouble(),
      (k[1] as num).toDouble(),
      (k[2] as num).toDouble(),
    );

List<double> _keypointToJson(Keypoint k) => [k.x, k.y, k.confidence];
