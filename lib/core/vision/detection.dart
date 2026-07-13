/// Pure-Dart data models for the vision pipeline output.
///
/// These are deliberately runtime-agnostic: they can be produced by the live
/// `ultralytics_yolo` plugin or replayed from the benchmark harness.
library;

/// A normalized [0,1] rectangle relative to the frame.
class BBox {
  const BBox(this.left, this.top, this.width, this.height);

  final double left;
  final double top;
  final double width;
  final double height;

  double get centerX => left + width / 2;
  double get centerY => top + height / 2;
}

/// A single detected object (e.g. the ball) with confidence.
class Detection {
  const Detection({
    required this.label,
    required this.confidence,
    required this.box,
  });

  final String label;
  final double confidence;
  final BBox box;
}

/// A normalized keypoint from a pose model with a per-point confidence.
class Keypoint {
  const Keypoint(this.x, this.y, this.confidence);

  final double x;
  final double y;
  final double confidence;
}

/// COCO 17-keypoint pose for a single detected person.
class PersonPose {
  const PersonPose({
    required this.box,
    required this.keypoints,
    this.trackId,
  });

  final BBox box;

  /// 17 COCO keypoints in the canonical order (nose, eyes, ..., ankles).
  final List<Keypoint> keypoints;

  /// Stable track id across frames, when the tracker can assign one.
  final int? trackId;
}

/// Everything the vision pipeline extracted from one camera frame.
class FrameResult {
  const FrameResult({
    required this.timestampMs,
    this.ball,
    this.ballCandidates = const [],
    this.people = const [],
    this.fps,
  });

  final int timestampMs;

  /// Best ball detection this frame, if any.
  final Detection? ball;

  /// The *other* ball detections this frame beyond [ball] (i.e. lower-confidence
  /// candidates that still passed the vision-layer filters), best-first. Usually
  /// empty — the detector reports at most one "ball" — but when it also latches
  /// onto a round object elsewhere in the frame these alternatives let a
  /// trajectory-aware consumer (the [BallTracker]'s `maxJump` gate) fall back to
  /// the candidate consistent with the predicted path instead of losing the ball.
  final List<Detection> ballCandidates;

  final List<PersonPose> people;

  /// Native inference FPS reported by the runtime, if available.
  final double? fps;
}
