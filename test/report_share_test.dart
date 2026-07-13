import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/share/report_share.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('defaultShareReport ignores an empty report without touching the '
      'platform channel', () async {
    // Returns early (never calls SharePlus), so this completes trivially.
    await defaultShareReport('');
  });

  test('defaultShareReport swallows the missing-plugin failure headlessly '
      'instead of throwing', () async {
    // No share provider is registered in a headless test, so the underlying
    // platform-channel call throws MissingPluginException; the sink must
    // degrade to a silent no-op (the clipboard actions are the fallback).
    await expectLater(
      defaultShareReport('Match summary', subject: 'Test'),
      completes,
    );
  });
}
