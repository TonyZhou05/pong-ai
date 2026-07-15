import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/labels/rally_label_store.dart';
import '../match/footage_demo.dart';
import '../match/match_screen.dart';

/// Human-labeling workbench for the rally corpus: every rally clip with its
/// actual footage playable inline, a live playback timer, and controls to
/// record who won, why (the losing action, in ITTF terms), and the rally's
/// end time (one tap stamps the current playback position). The pipeline's
/// current call is shown alongside so confirming or correcting is
/// unambiguous. Labels persist locally and export as JSON for the training
/// tooling.
class LabelingScreen extends StatefulWidget {
  const LabelingScreen({
    super.key,
    this.manifestLoader = loadFootageManifest,
    this.labelStore,
    this.playerBuilder = VideoFootagePlayer.new,
  });

  /// Resolves the corpus entries. Defaults to the bundled manifest asset.
  final Future<List<FootageMatch>> Function() manifestLoader;

  /// Label persistence; defaults to the shared_preferences-backed store.
  final RallyLabelStore? labelStore;

  /// Builds the inline footage player for a video asset. Defaults to the
  /// real `video_player`-backed implementation; tests inject a fake whose
  /// position the test drives (no platform channel headlessly).
  final FootagePlayer Function(String videoAsset) playerBuilder;

  @override
  State<LabelingScreen> createState() => _LabelingScreenState();
}

class _LabelingScreenState extends State<LabelingScreen> {
  late final RallyLabelStore _store =
      widget.labelStore ?? PrefsRallyLabelStore();
  List<FootageMatch>? _clips;
  Map<String, RallyLabel> _labels = {};
  Object? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final manifest = await widget.manifestLoader();
      final labels = await _store.load();
      if (!mounted) return;
      setState(() {
        // Per-rally clips only: the combined "_full" sets are derived from
        // the rally labels, not labeled directly.
        _clips = manifest
            .where((m) => !m.id.endsWith('_full'))
            .toList(growable: false);
        _labels = labels;
      });
    } catch (e) {
      if (mounted) setState(() => _error = e);
    }
  }

  Future<void> _update(
    FootageMatch clip, {
    String? winner,
    RallyLabelReason? reason,
    double? endSeconds,
  }) async {
    final existing = _labels[clip.id];
    final label = RallyLabel(
      clipId: clip.id,
      winner: winner ?? existing?.winner ?? 'unclear',
      reason: reason ?? existing?.reason,
      endSeconds: endSeconds ?? existing?.endSeconds,
      labeledAt: DateTime.now().toIso8601String(),
    );
    await _store.save(label);
    if (mounted) setState(() => _labels[clip.id] = label);
  }

  Future<void> _copyExport() async {
    final messenger = ScaffoldMessenger.of(context);
    final json = await _store.exportJson();
    await Clipboard.setData(ClipboardData(text: json));
    messenger.showSnackBar(
      const SnackBar(content: Text('Labels JSON copied to clipboard')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final clips = _clips;
    final labeled = _labels.length;
    return Scaffold(
      appBar: AppBar(
        title: Text(
          clips == null
              ? 'Label rallies'
              : 'Label rallies ($labeled/${clips.length})',
        ),
        actions: [
          IconButton(
            tooltip: 'Copy labels JSON (for training)',
            icon: const Icon(Icons.copy_all),
            onPressed: _copyExport,
          ),
        ],
      ),
      body: _error != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('Could not load the rally list: $_error'),
              ),
            )
          : clips == null
              ? const Center(child: CircularProgressIndicator())
              : ListView.builder(
                  padding: const EdgeInsets.all(12),
                  itemCount: clips.length,
                  itemBuilder: (context, i) => _RallyLabelCard(
                    key: ValueKey(clips[i].id),
                    clip: clips[i],
                    label: _labels[clips[i].id],
                    playerBuilder: widget.playerBuilder,
                    onWinner: (w) => _update(clips[i], winner: w),
                    onReason: (r) => _update(clips[i], reason: r),
                    onEndSeconds: (s) => _update(clips[i], endSeconds: s),
                    onWatchFull: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => MatchScreen(footage: clips[i].demo),
                      ),
                    ),
                  ),
                ),
    );
  }
}

class _RallyLabelCard extends StatefulWidget {
  const _RallyLabelCard({
    super.key,
    required this.clip,
    required this.label,
    required this.playerBuilder,
    required this.onWinner,
    required this.onReason,
    required this.onEndSeconds,
    required this.onWatchFull,
  });

  final FootageMatch clip;
  final RallyLabel? label;
  final FootagePlayer Function(String videoAsset) playerBuilder;
  final ValueChanged<String> onWinner;
  final ValueChanged<RallyLabelReason> onReason;
  final ValueChanged<double> onEndSeconds;
  final VoidCallback onWatchFull;

  @override
  State<_RallyLabelCard> createState() => _RallyLabelCardState();
}

class _RallyLabelCardState extends State<_RallyLabelCard> {
  /// The inline footage player, live while the footage section is open.
  FootagePlayer? _player;
  bool _playerReady = false;

  /// Polls the playback position so the timer readout stays live.
  Timer? _ticker;

  late final TextEditingController _endController = TextEditingController(
    text: widget.label?.endSeconds?.toStringAsFixed(1) ?? '',
  );

  @override
  void dispose() {
    _ticker?.cancel();
    _player?.dispose();
    _endController.dispose();
    super.dispose();
  }

  Future<void> _toggleFootage() async {
    if (_player != null) {
      _ticker?.cancel();
      _ticker = null;
      final p = _player;
      setState(() {
        _player = null;
        _playerReady = false;
      });
      await p?.dispose();
      return;
    }
    final player = widget.playerBuilder(widget.clip.demo.videoAsset);
    setState(() => _player = player);
    await player.initialize();
    if (!mounted || _player != player) return;
    await player.play();
    _ticker = Timer.periodic(const Duration(milliseconds: 100), (_) {
      if (mounted) setState(() {});
    });
    setState(() => _playerReady = true);
  }

  double get _positionSeconds =>
      (_player?.position.inMilliseconds ?? 0) / 1000.0;

  void _useCurrentTime() {
    final s = double.parse(_positionSeconds.toStringAsFixed(1));
    _endController.text = s.toStringAsFixed(1);
    widget.onEndSeconds(s);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final clip = widget.clip;
    final label = widget.label;
    final secs = (clip.durationMs / 1000).toStringAsFixed(0);
    final current = clip.truthSummary;
    final player = _player;
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(clip.title, style: theme.textTheme.titleMedium),
                ),
                if (label != null)
                  Icon(
                    Icons.check_circle,
                    size: 18,
                    color: theme.colorScheme.primary,
                  ),
                IconButton(
                  tooltip:
                      player == null ? 'Show footage' : 'Hide footage',
                  icon: Icon(
                    player == null
                        ? Icons.play_circle_outline
                        : Icons.expand_less,
                  ),
                  onPressed: _toggleFootage,
                ),
                IconButton(
                  tooltip: 'Open with AI overlays',
                  icon: const Icon(Icons.open_in_full),
                  onPressed: widget.onWatchFull,
                ),
              ],
            ),
            Text(
              '${secs}s · ${clip.bounces} bounces · pipeline says: '
              '${current ?? 'no call / unlabeled'}',
              style: theme.textTheme.bodySmall
                  ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
            ),
            if (player != null) ...[
              const SizedBox(height: 8),
              if (!_playerReady)
                const Center(
                  child: Padding(
                    padding: EdgeInsets.all(24),
                    child: CircularProgressIndicator(),
                  ),
                )
              else ...[
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: AspectRatio(
                    aspectRatio: player.aspectRatio,
                    child: player.view,
                  ),
                ),
                Row(
                  children: [
                    IconButton(
                      tooltip: player.isPlaying ? 'Pause' : 'Play',
                      icon: Icon(
                        player.isPlaying ? Icons.pause : Icons.play_arrow,
                      ),
                      onPressed: () async {
                        if (player.isPlaying) {
                          await player.pause();
                        } else {
                          await player.play();
                        }
                        if (mounted) setState(() {});
                      },
                    ),
                    IconButton(
                      tooltip: 'Restart',
                      icon: const Icon(Icons.replay),
                      onPressed: () async {
                        await player.seekToStart();
                        await player.play();
                        if (mounted) setState(() {});
                      },
                    ),
                    // The live playback timer — read the rally's end off it.
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: theme.colorScheme.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        '⏱ ${_positionSeconds.toStringAsFixed(1)}s',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontFeatures: const [FontFeature.tabularFigures()],
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    FilledButton.tonalIcon(
                      icon: const Icon(Icons.timer_outlined, size: 18),
                      label: const Text('Use as end time'),
                      onPressed: _useCurrentTime,
                    ),
                  ],
                ),
              ],
            ],
            const SizedBox(height: 8),
            SegmentedButton<String>(
              segments: const [
                ButtonSegment(value: 'a', label: Text('Player A (left)')),
                ButtonSegment(value: 'b', label: Text('Player B (right)')),
                ButtonSegment(value: 'unclear', label: Text('Unclear')),
              ],
              selected: {if (label != null) label.winner},
              emptySelectionAllowed: true,
              onSelectionChanged: (sel) {
                if (sel.isNotEmpty) widget.onWinner(sel.first);
              },
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  flex: 3,
                  child: DropdownButtonFormField<RallyLabelReason>(
                    initialValue: label?.reason,
                    isExpanded: true,
                    decoration: const InputDecoration(
                      labelText: 'Why (what the loser did)',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    items: [
                      for (final r in RallyLabelReason.values)
                        DropdownMenuItem(
                          value: r,
                          child: Text(
                            r.description,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                    ],
                    onChanged: (r) {
                      if (r != null) widget.onReason(r);
                    },
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: TextFormField(
                    controller: _endController,
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                    ),
                    decoration: const InputDecoration(
                      labelText: 'Ends at (s)',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    // Persist as the user types — a value that was only
                    // typed (never submitted) must survive moving on to the
                    // next card.
                    onChanged: (v) {
                      final s = double.tryParse(v);
                      if (s != null) widget.onEndSeconds(s);
                    },
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
