import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/replay_vision_service.dart';

FrameResult _frame(int t) => FrameResult(timestampMs: t);

void main() {
  group('ReplayVisionService', () {
    test('emits the supplied frames in order and stops when exhausted',
        () async {
      final frames = [_frame(0), _frame(33), _frame(66)];
      final service = ReplayVisionService(
        frames,
        interval: const Duration(milliseconds: 1),
      );

      final collected = service.frames.take(3).toList();
      await service.start();
      final result = await collected;

      expect(result.map((f) => f.timestampMs), [0, 33, 66]);
      // Let the next tick run, which detects exhaustion and stops the timer.
      await Future<void>.delayed(const Duration(milliseconds: 5));
      expect(service.isFinished, isTrue);
      await service.dispose();
    });

    test('stop() halts emission before the clip finishes', () async {
      final frames = List.generate(100, (i) => _frame(i));
      final service = ReplayVisionService(
        frames,
        interval: const Duration(milliseconds: 5),
      );

      final received = <FrameResult>[];
      service.frames.listen(received.add);
      await service.start();
      await Future<void>.delayed(const Duration(milliseconds: 12));
      await service.stop();
      final countAfterStop = received.length;

      await Future<void>.delayed(const Duration(milliseconds: 20));
      expect(received.length, countAfterStop);
      expect(received.length, lessThan(frames.length));
      await service.dispose();
    });
  });
}
