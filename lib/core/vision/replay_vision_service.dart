/// A [VisionService] that replays a fixed list of [FrameResult]s on a timer.
///
/// This is the non-camera implementation of the vision seam. It serves two
/// purposes:
///
/// * **In-app demo / development** — drive the live [MatchController] pipeline
///   and its UI without a phone camera or the `ultralytics_yolo` runtime.
/// * **Benchmarking** — feed frames decoded from a recorded match clip through
///   the exact same pipeline the live camera will use, so tracker/referee
///   accuracy can be measured offline (see docs/ARCHITECTURE.md).
///
/// It has no Flutter dependency, so it can be exercised in pure-Dart tests.
library;

import 'dart:async';

import 'detection.dart';
import 'vision_service.dart';

class ReplayVisionService implements VisionService {
  ReplayVisionService(
    this._frames, {
    this.interval = const Duration(milliseconds: 33),
    this.loop = false,
  });

  /// The frames to replay, in timestamp order.
  final List<FrameResult> _frames;

  /// Wall-clock spacing between emitted frames (the replay cadence). This is
  /// independent of the frames' own [FrameResult.timestampMs], which the
  /// downstream tracker uses for its velocity maths.
  final Duration interval;

  /// When true, restart from the first frame after the last one is emitted.
  final bool loop;

  final StreamController<FrameResult> _controller =
      StreamController<FrameResult>.broadcast();
  Timer? _timer;
  int _index = 0;
  bool _loaded = false;

  /// Whether the replay has emitted its final frame (never true when [loop]).
  bool get isFinished => !loop && _index >= _frames.length && _timer == null;

  @override
  Stream<FrameResult> get frames => _controller.stream;

  @override
  Future<void> load() async {
    _loaded = true;
  }

  @override
  Future<void> start() async {
    if (!_loaded) await load();
    _timer?.cancel();
    _index = 0;
    _timer = Timer.periodic(interval, (_) => _tick());
  }

  void _tick() {
    if (_index >= _frames.length) {
      if (loop) {
        _index = 0;
      } else {
        _timer?.cancel();
        _timer = null;
        return;
      }
    }
    if (!_controller.isClosed) {
      _controller.add(_frames[_index]);
    }
    _index++;
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
