/// The camera-backed [VisionService]: routes live `ultralytics_yolo` output
/// into the app's [FrameResult] stream.
///
/// The plugin's on-device inference is surfaced through a `YOLOView` widget
/// whose `onStreamingData` callback fires once per processed frame. That widget
/// lives in the UI layer (it needs a platform view / camera), but the *routing*
/// — converting each raw payload via [YoloFrameAdapter], enforcing a monotonic
/// clock, and multiplexing onto a stream the [MatchController] pipeline reads —
/// is pure Dart and lives here so it can be unit tested without a device.
///
/// Wiring, from the screen that owns the camera preview:
///
/// ```dart
/// final vision = YoloVisionService();
/// await vision.load();
/// await vision.start();
/// // ...in build():
/// YOLOView(
///   task: YOLOTask.detect,
///   modelPath: 'yolo11n',
///   onStreamingData: vision.onStreamingData,
/// );
/// // vision.frames now drives MatchController / MatchScreen.
/// ```
library;

import 'dart:async';

import 'detection.dart';
import 'vision_service.dart';
import 'yolo_frame_adapter.dart';

class YoloVisionService implements VisionService {
  YoloVisionService({
    YoloFrameAdapter? adapter,
    Duration? defaultFrameInterval,
  })  : adapter = adapter ?? const YoloFrameAdapter(),
        _defaultStepMs = (defaultFrameInterval ?? const Duration(milliseconds: 33))
            .inMilliseconds
            .clamp(1, 1000);

  /// Converts raw plugin detections into runtime-agnostic [FrameResult]s.
  final YoloFrameAdapter adapter;

  /// Fallback frame spacing (ms) used to synthesize a monotonic timestamp when
  /// the plugin reports no/duplicate/decreasing timestamps and no usable fps.
  final int _defaultStepMs;

  final StreamController<FrameResult> _controller =
      StreamController<FrameResult>.broadcast();

  bool _loaded = false;
  bool _running = false;
  int _frameCount = 0;

  /// Timestamp of the last frame we emitted, so the [BallTracker] downstream
  /// (which drops non-increasing timestamps) always sees a strictly rising
  /// clock even if the device reports a wobbly or absent one.
  int? _lastTs;

  @override
  Stream<FrameResult> get frames => _controller.stream;

  /// Whether frames are currently being forwarded (between [start] and [stop]).
  bool get isRunning => _running;

  /// Number of frames forwarded onto [frames] since the last [start].
  int get frameCount => _frameCount;

  @override
  Future<void> load() async {
    _loaded = true;
  }

  @override
  Future<void> start() async {
    if (!_loaded) await load();
    _running = true;
    _frameCount = 0;
    _lastTs = null;
  }

  @override
  Future<void> stop() async {
    _running = false;
  }

  /// Callback wired to `YOLOView.onStreamingData`. Ignored unless [start] has
  /// been called; drops frames after [stop] or [dispose] so a lingering plugin
  /// callback can't leak past the service lifecycle.
  void onStreamingData(Map<dynamic, dynamic> data) {
    if (!_running || _controller.isClosed) return;
    final raw = adapter.fromStreamingData(
      data,
      fallbackTimestampMs: _syntheticNext(),
    );
    _emit(raw);
  }

  /// Forward an already-parsed [FrameResult] (e.g. from a service that parses
  /// the payload itself). Also monotonic-clamped so all entry points share one
  /// clock.
  void onFrame(FrameResult frame) {
    if (!_running || _controller.isClosed) return;
    _emit(frame);
  }

  void _emit(FrameResult frame) {
    final ts = _monotonic(frame.timestampMs, frame.fps);
    _lastTs = ts;
    _frameCount++;
    _controller.add(
      FrameResult(
        timestampMs: ts,
        ball: frame.ball,
        people: frame.people,
        fps: frame.fps,
      ),
    );
  }

  /// Force [reported] to be strictly greater than the previously emitted
  /// timestamp, stepping by the frame interval (derived from [fps] when
  /// available) when the device clock stalls or goes backwards.
  int _monotonic(int reported, double? fps) {
    final last = _lastTs;
    if (last == null) return reported;
    if (reported > last) return reported;
    return last + _stepMs(fps);
  }

  /// A plausible timestamp for a payload that carried none, so the adapter's
  /// fallback is already monotonic before [_emit] re-checks it.
  int _syntheticNext() => (_lastTs ?? 0) + _defaultStepMs;

  int _stepMs(double? fps) {
    if (fps != null && fps > 0) {
      return (1000 / fps).round().clamp(1, 1000);
    }
    return _defaultStepMs;
  }

  @override
  Future<void> dispose() async {
    _running = false;
    if (!_controller.isClosed) await _controller.close();
  }
}
