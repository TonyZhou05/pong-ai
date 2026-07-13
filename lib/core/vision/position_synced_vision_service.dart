/// A [VisionService] that replays recorded [FrameResult]s in lock-step with an
/// external playback clock — the seam that keeps detection overlays glued to
/// real match *footage*.
///
/// [ReplayVisionService] paces frames on its own wall-clock timer, which is
/// right for a headless replay but drifts against a real video element (whose
/// playback can stall, buffer, or be paused by the user). This service instead
/// polls a caller-supplied `positionMs` (e.g. the video player's current
/// position) and emits every not-yet-emitted frame whose timestamp has been
/// reached, so a paused video pauses the pipeline and a seek-to-start replays
/// it — the overlay can never desynchronize from the pixels underneath it.
///
/// Pure Dart (a [Timer] plus a callback), so it is unit-testable without any
/// video plugin.
library;

import 'dart:async';

import 'detection.dart';
import 'vision_service.dart';

class PositionSyncedVisionService implements VisionService {
  PositionSyncedVisionService(
    this._frames, {
    required this.positionMs,
    this.pollInterval = const Duration(milliseconds: 16),
  });

  /// The frames to replay, in timestamp order. Timestamps are on the same
  /// clock as [positionMs] (milliseconds into the footage).
  final List<FrameResult> _frames;

  /// The external playback clock: current position into the footage in ms.
  final int Function() positionMs;

  /// How often the playback position is sampled.
  final Duration pollInterval;

  final StreamController<FrameResult> _controller =
      StreamController<FrameResult>.broadcast();
  Timer? _timer;
  int _index = 0;

  /// The playback position observed on the previous poll, to detect the clock
  /// jumping *backwards* (a seek) and re-sync the frame cursor.
  int _lastPos = 0;

  /// Whether every frame has been emitted (the footage played to the end).
  bool get isFinished => _index >= _frames.length;

  @override
  Stream<FrameResult> get frames => _controller.stream;

  @override
  Future<void> load() async {}

  /// Starts (or restarts, after a seek-to-start) polling the playback clock.
  /// Emission is driven purely by [positionMs], so frames only flow while the
  /// underlying footage actually advances.
  @override
  Future<void> start() async {
    _timer?.cancel();
    _index = 0;
    _lastPos = 0;
    _timer = Timer.periodic(pollInterval, (_) => _tick());
  }

  void _tick() {
    final pos = positionMs();
    // The clock moved backwards: the footage was seeked (e.g. rewound for a
    // replay). Re-sync the cursor so emission resumes from the new position
    // instead of staying exhausted at the old one — otherwise the overlay
    // would freeze for the rest of the (re-)viewing.
    if (pos < _lastPos) {
      _index = 0;
      while (_index < _frames.length && _frames[_index].timestampMs < pos) {
        _index++;
      }
    }
    _lastPos = pos;
    while (_index < _frames.length &&
        _frames[_index].timestampMs <= pos &&
        !_controller.isClosed) {
      _controller.add(_frames[_index]);
      _index++;
    }
    if (_index >= _frames.length) {
      _timer?.cancel();
      _timer = null;
    }
  }

  @override
  Future<void> stop() async {
    _timer?.cancel();
    _timer = null;
  }

  @override
  Future<void> dispose() async {
    await stop();
    if (!_controller.isClosed) await _controller.close();
  }
}
