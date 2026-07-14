import 'package:flutter/material.dart';

import '../history/session_history_screen.dart';
import '../match/camera_match_screen.dart';
import '../match/footage_demo.dart';
import '../match/match_screen.dart';
import '../labeling/labeling_screen.dart';
import '../matches/matches_screen.dart';
import '../training/camera_training_screen.dart';
import '../training/training_screen.dart';

/// Landing screen: pick between refereeing a live match and training mode.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 24),
              Text(
                'pong-ai',
                style: Theme.of(context).textTheme.displaySmall,
              ),
              const SizedBox(height: 8),
              Text(
                'Place your phone on the side of the table.',
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 32),
              _ModeCard(
                icon: Icons.videocam,
                title: 'Live Match',
                subtitle: 'Point the camera at the table for live auto-scoring.',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const CameraMatchScreen(),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              _ModeCard(
                icon: Icons.sports_tennis,
                title: 'Match',
                subtitle:
                    'Real footage demo: watch the AI identify players and ball.',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) =>
                        const MatchScreen(footage: defaultFootageDemo),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              _ModeCard(
                icon: Icons.video_library,
                title: 'Matches',
                subtitle:
                    'Real recorded rallies with AI tracking, one tap each.',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const MatchesScreen(),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              _ModeCard(
                icon: Icons.videocam,
                title: 'Live Training',
                subtitle: 'Point the camera at the net and grade shots live.',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const CameraTrainingScreen(),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              _ModeCard(
                icon: Icons.fitness_center,
                title: 'Training',
                subtitle: 'Demo replay: grade a scripted drill.',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const TrainingScreen(),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              _ModeCard(
                icon: Icons.fact_check,
                title: 'Label rallies',
                subtitle:
                    'Record who won each rally and why — trains the referee.',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const LabelingScreen(),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              _ModeCard(
                icon: Icons.history,
                title: 'History',
                subtitle: 'Review saved match and training summaries.',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => const SessionHistoryScreen(),
                  ),
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}

class _ModeCard extends StatelessWidget {
  const _ModeCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        leading: Icon(icon, size: 36),
        title: Text(title, style: Theme.of(context).textTheme.titleLarge),
        subtitle: Text(subtitle),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}
