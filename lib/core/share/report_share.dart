import 'package:share_plus/share_plus.dart';

/// Signature of the sink the summary panels call to hand a composed report off
/// to the operating system's share sheet ("send to a coach").
///
/// It is injectable so the whole share wiring stays widget-testable headlessly:
/// a test passes a fake that records the text/subject, while the app uses
/// [defaultShareReport] which is backed by the real `share_plus` plugin.
typedef ShareReportSink = Future<void> Function(
  String text, {
  String? subject,
});

/// The on-device [ShareReportSink]: opens the platform share sheet so the user
/// can send the summary to Messages, email, a coach, etc.
///
/// The `share_plus` platform-channel call is guarded so a device without a
/// share provider — or a headless widget test where the channel is absent —
/// degrades to a silent no-op instead of throwing, mirroring how
/// [SpeechAnnouncer] guards the TTS channel. The clipboard "Copy report" /
/// "Export JSON" actions remain the always-available fallback.
Future<void> defaultShareReport(String text, {String? subject}) async {
  if (text.isEmpty) return;
  try {
    await SharePlus.instance.share(
      ShareParams(text: text, subject: subject),
    );
  } catch (_) {
    // No share provider available (unsupported device or headless test) — the
    // clipboard actions still let the user get the report out, so silence here
    // is an acceptable degradation rather than a crash.
  }
}
