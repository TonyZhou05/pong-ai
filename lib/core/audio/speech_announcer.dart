import 'package:flutter_tts/flutter_tts.dart';

/// Minimal abstraction over a text-to-speech engine so the announcer logic
/// (one-time setup, error guarding) is unit-testable without the `flutter_tts`
/// platform channel.
abstract class TtsEngine {
  /// Configure the voice once (language, rate, pitch, queue behaviour).
  Future<void> setup();

  /// Speak [text] aloud, queued behind any in-progress utterance.
  Future<void> speak(String text);

  /// Stop any in-progress/queued speech (e.g. on dispose).
  Future<void> stop();
}

/// [TtsEngine] backed by the real `flutter_tts` plugin.
///
/// Uses a moderate speech rate and `QueueMode.add` (queue mode `1`) so a burst
/// of calls — e.g. a point call immediately followed by a "change ends" cue —
/// is spoken in order rather than cutting each other off, matching how a human
/// umpire delivers back-to-back calls.
class FlutterTtsEngine implements TtsEngine {
  FlutterTtsEngine([FlutterTts? tts]) : _tts = tts ?? FlutterTts();

  final FlutterTts _tts;

  @override
  Future<void> setup() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.5);
    await _tts.setPitch(1.0);
    await _tts.awaitSpeakCompletion(true);
    await _tts.setQueueMode(1);
  }

  @override
  Future<void> speak(String text) => _tts.speak(text);

  @override
  Future<void> stop() => _tts.stop();
}

/// Speaks umpire/coach announcement strings aloud through the device's
/// text-to-speech engine, so a phone propped table-side actually *voices* the
/// score/shot calls for a player standing across the table — not just a haptic
/// click plus an on-screen caption.
///
/// This is the concrete realisation of the injectable `onAnnounce` sink the
/// live match/training screens have carried since the announcer was introduced:
/// the default sink previously only pulsed a haptic + system-click cue, so the
/// carefully composed spoken phrases (point/game/match calls, serve rotation,
/// pressure cues, shot grades) were captioned but never actually heard.
///
/// Every engine call is guarded so a device without a TTS engine — or a
/// headless widget test where the platform channel is absent — degrades to a
/// silent no-op instead of throwing.
class SpeechAnnouncer {
  SpeechAnnouncer(this._engine) {
    // Fire-and-forget one-time setup; [_speak] awaits it before speaking so an
    // announcement that arrives before setup completes is still voiced (with
    // the configured voice) rather than dropped.
    _setup = _engine.setup().catchError((_) {});
  }

  /// Builds a [SpeechAnnouncer] on the real on-device `flutter_tts` engine.
  factory SpeechAnnouncer.device() => SpeechAnnouncer(FlutterTtsEngine());

  final TtsEngine _engine;
  late final Future<void> _setup;

  /// Speak [text] aloud. Empty strings are ignored; engine failures are
  /// swallowed so a missing/unsupported TTS engine never crashes a live match.
  void announce(String text) {
    if (text.isEmpty) return;
    _speak(text);
  }

  Future<void> _speak(String text) async {
    try {
      await _setup;
      await _engine.speak(text);
    } catch (_) {
      // No TTS engine available (unsupported device or headless test) — the
      // caption still shows the call, so silence here is an acceptable
      // degradation rather than a crash.
    }
  }

  /// Stop any in-progress/queued speech; call when the screen is disposed so a
  /// half-spoken call doesn't linger after the user leaves.
  Future<void> stop() async {
    try {
      await _engine.stop();
    } catch (_) {}
  }
}
