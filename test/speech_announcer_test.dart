import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/audio/speech_announcer.dart';

/// Records every engine call so the announcer's setup/queue/guard behaviour is
/// verifiable without the real `flutter_tts` platform channel.
class _FakeTtsEngine implements TtsEngine {
  final List<String> spoken = [];
  int setupCount = 0;
  int stopCount = 0;

  @override
  Future<void> setup() async => setupCount++;

  @override
  Future<void> speak(String text) async => spoken.add(text);

  @override
  Future<void> stop() async => stopCount++;
}

/// Throws from every method to prove announce/stop swallow engine failures
/// (e.g. a device with no TTS engine, or a headless test).
class _ThrowingTtsEngine implements TtsEngine {
  @override
  Future<void> setup() async => throw StateError('no engine');

  @override
  Future<void> speak(String text) async => throw StateError('no engine');

  @override
  Future<void> stop() async => throw StateError('no engine');
}

void main() {
  group('SpeechAnnouncer', () {
    test('speaks the announcement text through the engine after setup',
        () async {
      final engine = _FakeTtsEngine();
      final announcer = SpeechAnnouncer(engine);

      announcer.announce('Player A, 1-0.');
      // Let the fire-and-forget setup + speak futures complete.
      await Future<void>.delayed(Duration.zero);

      expect(engine.setupCount, 1);
      expect(engine.spoken, ['Player A, 1-0.']);
    });

    test('queues multiple calls in order (setup runs only once)', () async {
      final engine = _FakeTtsEngine();
      final announcer = SpeechAnnouncer(engine);

      announcer.announce('Player A, 1-0.');
      announcer.announce('Change ends.');
      await Future<void>.delayed(Duration.zero);

      expect(engine.setupCount, 1);
      expect(engine.spoken, ['Player A, 1-0.', 'Change ends.']);
    });

    test('ignores empty strings', () async {
      final engine = _FakeTtsEngine();
      final announcer = SpeechAnnouncer(engine);

      announcer.announce('');
      await Future<void>.delayed(Duration.zero);

      expect(engine.spoken, isEmpty);
    });

    test('swallows engine failures instead of throwing', () async {
      final announcer = SpeechAnnouncer(_ThrowingTtsEngine());

      // Neither the fire-and-forget setup, nor announce, nor stop may throw.
      expect(() => announcer.announce('anything'), returnsNormally);
      await expectLater(announcer.stop(), completes);
      await Future<void>.delayed(Duration.zero);
    });

    test('stop() forwards to the engine', () async {
      final engine = _FakeTtsEngine();
      final announcer = SpeechAnnouncer(engine);

      await announcer.stop();

      expect(engine.stopCount, 1);
    });
  });
}
