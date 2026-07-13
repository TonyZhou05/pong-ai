import 'dart:ui' show Rect;

import 'package:ultralytics_yolo/ultralytics_yolo.dart' as yolo;

import 'detection.dart';

/// Tuning for how raw YOLO detections are turned into a [FrameResult].
///
/// The default label sets match the COCO class names emitted by the stock
/// YOLO detection/pose models the [ultralytics_yolo] plugin ships (the ball is
/// COCO class 32, "sports ball"); a fine-tuned ping-pong ball model can be
/// slotted in by extending [ballLabels].
class YoloFrameConfig {
  const YoloFrameConfig({
    this.ballLabels = const {'sports ball', 'ball', 'ping pong ball'},
    this.personLabels = const {'person'},
    this.minBallConfidence = 0.20,
    this.minPersonConfidence = 0.30,
    this.maxBallRelativeSize,
    this.maxPeople = 2,
  }) : assert(
          maxBallRelativeSize == null ||
              (maxBallRelativeSize > 0 && maxBallRelativeSize <= 1),
          'maxBallRelativeSize must be in (0, 1]',
        );

  /// Lower-cased class names that count as the ball.
  final Set<String> ballLabels;

  /// Lower-cased class names that count as a player.
  final Set<String> personLabels;

  /// Drop ball detections below this confidence.
  final double minBallConfidence;

  /// Reject a "ball" detection whose *smaller* box dimension exceeds this
  /// fraction of the frame — a physical size sanity cap for a ping-pong ball.
  ///
  /// A regulation ball is 40mm on a 2.74m table that spans the frame, i.e. under
  /// ~2% of the frame; even accounting for the near-camera perspective and
  /// motion blur it never approaches a quarter of the frame. So a generic
  /// COCO "sports ball" box that is large in *both* axes (a person's head/torso,
  /// a bright round logo, or an actual basketball/volleyball in a gym) cannot be
  /// the ping-pong ball and would otherwise seed a bad trajectory — before the
  /// [BallTracker]'s post-trajectory `maxJump` gate can ever engage. Gating on
  /// the smaller dimension keeps a legitimately motion-blurred ball (elongated
  /// along one axis only) while catching genuinely large false positives.
  /// `null` disables the gate (the default, preserving prior behaviour).
  final double? maxBallRelativeSize;

  /// Drop person detections below this confidence.
  final double minPersonConfidence;

  /// Keep at most this many players (the largest by box area — the two people
  /// nearest/most prominent at the table), since only two people ever matter at
  /// a table. Non-positive means "no limit".
  final int maxPeople;
}

/// Converts the [ultralytics_yolo] plugin's per-frame output into the app's
/// runtime-agnostic [FrameResult].
///
/// This is the seam between live on-device inference and the pure-Dart
/// tracker → referee → scoring pipeline: any camera-backed [VisionService]
/// funnels the plugin's `onStreamingData` payload (or its parsed
/// `List<YOLOResult>`) through here. Keeping the mapping pure makes it unit
/// testable without a device, camera, or platform channel.
class YoloFrameAdapter {
  const YoloFrameAdapter({this.config = const YoloFrameConfig()});

  final YoloFrameConfig config;

  /// Map a raw `onStreamingData` payload to a [FrameResult].
  ///
  /// Recognized keys (see the plugin's `YOLOView.onStreamingData` docs):
  /// `detections`, `fps`, `timestamp`, `imageWidth`, `imageHeight`. Missing
  /// keys degrade gracefully — an absent timestamp falls back to
  /// [fallbackTimestampMs], and absent image dimensions drop keypoints (the
  /// player box, which is already normalized, is still kept).
  FrameResult fromStreamingData(
    Map<dynamic, dynamic> data, {
    int fallbackTimestampMs = 0,
  }) {
    final rawDetections = (data['detections'] as List<dynamic>?) ?? const [];
    final results = <yolo.YOLOResult>[];
    for (final entry in rawDetections) {
      if (entry is Map) {
        results.add(yolo.YOLOResult.fromMap(entry));
      }
    }

    final timestamp = _asDouble(data['timestamp']);
    return fromResults(
      results,
      timestampMs: timestamp?.round() ?? fallbackTimestampMs,
      imageWidth: _asDouble(data['imageWidth']),
      imageHeight: _asDouble(data['imageHeight']),
      fps: _asDouble(data['fps']),
    );
  }

  /// Map already-parsed detections to a [FrameResult].
  ///
  /// [imageWidth]/[imageHeight] are the upright frame dimensions used to
  /// normalize pose keypoints (which the plugin reports in pixel space); when
  /// omitted, keypoints are dropped but bounding boxes are still produced.
  FrameResult fromResults(
    List<yolo.YOLOResult> results, {
    required int timestampMs,
    double? imageWidth,
    double? imageHeight,
    double? fps,
  }) {
    yolo.YOLOResult? bestBall;
    final people = <PersonPose>[];

    for (final r in results) {
      final label = r.className.toLowerCase();
      if (config.ballLabels.contains(label)) {
        if (r.confidence < config.minBallConfidence) continue;
        if (_ballTooLarge(r.normalizedBox)) continue;
        if (bestBall == null || r.confidence > bestBall.confidence) {
          bestBall = r;
        }
      } else if (config.personLabels.contains(label)) {
        if (r.confidence < config.minPersonConfidence) continue;
        people.add(
          PersonPose(
            box: _bbox(r.normalizedBox),
            keypoints: _keypoints(r, imageWidth, imageHeight),
          ),
        );
      }
    }

    // Keep only the two most-prominent people (largest box *area*), since only
    // two players ever matter at a table. Area, not width: with the phone at the
    // side of the table the players are seen side-on (narrow but tall boxes),
    // while a spectator facing the camera is wide but short — sorting on width
    // alone would systematically drop the real players for a bystander.
    if (config.maxPeople > 0 && people.length > config.maxPeople) {
      people.sort((a, b) => _boxArea(b.box).compareTo(_boxArea(a.box)));
      people.removeRange(config.maxPeople, people.length);
    }

    return FrameResult(
      timestampMs: timestampMs,
      ball: bestBall == null
          ? null
          : Detection(
              label: 'ball',
              confidence: bestBall.confidence,
              box: _bbox(bestBall.normalizedBox),
            ),
      people: people,
      fps: fps,
    );
  }

  BBox _bbox(Rect box) => BBox(box.left, box.top, box.width, box.height);

  /// Whether a ball candidate's box is implausibly large for a ping-pong ball.
  /// Uses the *smaller* dimension so a motion-blurred ball (elongated along one
  /// axis) survives while a head/torso/large-ball false positive (large in both
  /// axes) is rejected. Always false when the gate is disabled.
  bool _ballTooLarge(Rect box) {
    final limit = config.maxBallRelativeSize;
    if (limit == null) return false;
    final smaller = box.width < box.height ? box.width : box.height;
    return smaller > limit;
  }

  static double _boxArea(BBox box) => box.width * box.height;

  List<Keypoint> _keypoints(
    yolo.YOLOResult r,
    double? imageWidth,
    double? imageHeight,
  ) {
    final points = r.keypoints;
    final confs = r.keypointConfidences;
    if (points == null ||
        confs == null ||
        imageWidth == null ||
        imageHeight == null ||
        imageWidth <= 0 ||
        imageHeight <= 0) {
      return const [];
    }
    final out = <Keypoint>[];
    for (var i = 0; i < points.length && i < confs.length; i++) {
      out.add(
        Keypoint(
          (points[i].x / imageWidth).clamp(0.0, 1.0),
          (points[i].y / imageHeight).clamp(0.0, 1.0),
          confs[i],
        ),
      );
    }
    return out;
  }

  static double? _asDouble(Object? value) =>
      value is num ? value.toDouble() : null;
}
