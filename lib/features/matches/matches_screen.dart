import 'package:flutter/material.dart';

import '../match/footage_demo.dart';
import '../match/match_screen.dart';

/// The real-footage match corpus: a list of side-recorded rally clips
/// (OpenTTGames), each carrying a recorded YOLO detection track. Tapping one
/// opens the footage Match screen — the actual video with the app's
/// player/ball identification overlaid and the scoring pipeline running on
/// the recorded detections.
///
/// The manifest loader is injectable so tests supply an in-memory corpus
/// (asset I/O hangs `testWidgets`).
class MatchesScreen extends StatefulWidget {
  const MatchesScreen({super.key, this.manifestLoader = loadFootageManifest});

  /// Resolves the corpus entries. Defaults to the bundled manifest asset.
  final Future<List<FootageMatch>> Function() manifestLoader;

  @override
  State<MatchesScreen> createState() => _MatchesScreenState();
}

class _MatchesScreenState extends State<MatchesScreen> {
  List<FootageMatch>? _matches;
  Object? _error;

  @override
  void initState() {
    super.initState();
    widget.manifestLoader().then((matches) {
      if (mounted) setState(() => _matches = matches);
    }).catchError((Object e) {
      if (mounted) setState(() => _error = e);
    });
  }

  void _open(FootageMatch match) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => MatchScreen(footage: match.demo),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final matches = _matches;
    return Scaffold(
      appBar: AppBar(title: const Text('Matches')),
      body: _error != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('Could not load the match list: $_error'),
              ),
            )
          : matches == null
              ? const Center(child: CircularProgressIndicator())
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: matches.length,
                  itemBuilder: (context, i) =>
                      _MatchCard(match: matches[i], onTap: () => _open(matches[i])),
                ),
    );
  }
}

class _MatchCard extends StatelessWidget {
  const _MatchCard({required this.match, required this.onTap});

  final FootageMatch match;
  final VoidCallback onTap;

  String get _subtitle {
    final secs = (match.durationMs / 1000).toStringAsFixed(0);
    final parts = [
      '${secs}s',
      if (match.bounces > 0) '${match.bounces} bounces',
      if (match.source.isNotEmpty) match.source,
    ];
    return parts.join(' · ');
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: const Icon(Icons.play_circle_outline, size: 32),
        title: Text(match.title),
        subtitle: Text(_subtitle),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}
