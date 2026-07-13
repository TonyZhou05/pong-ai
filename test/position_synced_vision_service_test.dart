import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/vision/detection.dart';
import 'package:pong_ai/core/vision/position_synced_vision_service.dart';

FrameResult _frame(int t) => FrameResult(timestampMs: t);

void main() {
  group('PositionSyncedVisionService', () {
    test('emits exactly the frames the playback clock has reached, in order',
        () {
      fakeAsync((async) {
        var positionMs = 0;
        final service = PositionSyncedVisionService(
          [_frame(0), _frame(33), _frame(66), _frame(100)],
          positionMs: () => positionMs,
        );
        final emitted = <int>[];
        service.frames.listen((f) => emitted.add(f.timestampMs));

        service.start();
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0], reason: 'only t=0 has been reached at position 0');

        positionMs = 70;
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0, 33, 66]);

        service.dispose();
        async.flushTimers();
      });
    });

    test('a stalled clock (paused video) emits nothing until it advances', () {
      fakeAsync((async) {
        var positionMs = 40;
        final service = PositionSyncedVisionService(
          [_frame(0), _frame(33), _frame(66)],
          positionMs: () => positionMs,
        );
        final emitted = <int>[];
        service.frames.listen((f) => emitted.add(f.timestampMs));

        service.start();
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0, 33]);

        // Paused: the clock stays put, so no matter how much wall time passes
        // nothing further is emitted.
        async.elapse(const Duration(seconds: 2));
        expect(emitted, [0, 33]);

        positionMs = 66;
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0, 33, 66]);

        service.dispose();
        async.flushTimers();
      });
    });

    test('start() after a seek-to-start replays from the first frame', () {
      fakeAsync((async) {
        var positionMs = 100;
        final service = PositionSyncedVisionService(
          [_frame(0), _frame(50)],
          positionMs: () => positionMs,
        );
        final emitted = <int>[];
        service.frames.listen((f) => emitted.add(f.timestampMs));

        service.start();
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0, 50]);
        expect(service.isFinished, isTrue);

        positionMs = 0; // seeked back
        service.start();
        expect(service.isFinished, isFalse);
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0, 50, 0]);

        positionMs = 60;
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0, 50, 0, 50]);

        service.dispose();
        async.flushTimers();
      });
    });

    test('a clock jumping backwards (a seek) re-syncs and re-emits', () {
      fakeAsync((async) {
        var positionMs = 60;
        final service = PositionSyncedVisionService(
          [_frame(0), _frame(50), _frame(100)],
          positionMs: () => positionMs,
        );
        final emitted = <int>[];
        service.frames.listen((f) => emitted.add(f.timestampMs));

        service.start();
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0, 50]);

        // Seek back mid-stream: the cursor re-syncs to the new position so
        // the overlay resumes from there instead of freezing.
        positionMs = 10;
        async.elapse(const Duration(milliseconds: 20));
        positionMs = 55;
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0, 50, 50]);

        service.dispose();
        async.flushTimers();
      });
    });

    test('stops polling once every frame has been emitted', () {
      fakeAsync((async) {
        final service = PositionSyncedVisionService(
          [_frame(0)],
          positionMs: () => 10,
        );
        final emitted = <int>[];
        service.frames.listen((f) => emitted.add(f.timestampMs));

        service.start();
        async.elapse(const Duration(milliseconds: 20));
        expect(emitted, [0]);
        expect(service.isFinished, isTrue);

        // The periodic poll timer has self-cancelled; nothing is pending.
        expect(async.periodicTimerCount, 0);

        service.dispose();
        async.flushTimers();
      });
    });
  });
}
