import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/app.dart';

void main() {
  testWidgets('Home screen shows Match and Training modes', (tester) async {
    await tester.pumpWidget(const PongAiApp());

    expect(find.text('pong-ai'), findsOneWidget);
    expect(find.text('Match'), findsOneWidget);
    expect(find.text('Training'), findsOneWidget);
    expect(find.byIcon(Icons.sports_tennis), findsOneWidget);
  });
}
