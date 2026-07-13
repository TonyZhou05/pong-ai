import 'dart:async';

import 'detection.dart';

/// Abstraction over the on-device model runtime.
///
/// Implemented for real inference by a `ultralytics_yolo`-backed service
/// (added in a later iteration), and by a fake/replay service that feeds
/// recorded [FrameResult]s for tests and benchmarking. Keeping the rest of the
/// app behind this interface lets us swap models without touching UI or rules.
abstract class VisionService {
  /// Load the player-pose and ball-detection models into memory.
  Future<void> load();

  /// A stream of per-frame results while the camera is running.
  Stream<FrameResult> get frames;

  /// Start/stop consuming the camera.
  Future<void> start();
  Future<void> stop();

  Future<void> dispose();
}
