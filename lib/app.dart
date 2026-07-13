import 'package:flutter/material.dart';

import 'features/home/home_screen.dart';

class PongAiApp extends StatelessWidget {
  const PongAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'pong-ai',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1B5E20),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
