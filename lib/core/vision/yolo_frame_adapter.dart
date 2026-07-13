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
    this.maxPeople = 2,
  });

  /// Lower-cased class names that count as the ball.
  final Set<String> ballLabels;

  /// Lower-cased class names that count as a player.
  final Set<String> personLabels;

  /// Drop ball detections below this confidence.
  final double minBallConfidence;

  /// Drop person detections below this confidence.
  final double minPersonConfidence;

  /// Keep at most this many players (the most confident ones), since only two
  /// people ever matter at a table. Non-positive means "no limit".
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

    // Keep only the most-confident players; two ever matter at a table.
    if (config.maxPeople > 0 && people.length > config.maxPeople) {
      people.sort((a, b) => b.box.width.compareTo(a.box.width));
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
