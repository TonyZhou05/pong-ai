import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/labels/rally_label_store.dart';
import '../match/footage_demo.dart';
import '../match/match_screen.dart';

/// Human-labeling workbench for the rally corpus: for every rally clip,
/// record who won, why (the losing action, in ITTF terms), and a rough end
/// time — alongside the pipeline's current call so confirming or correcting
/// is unambiguous. Labels persist locally and export as JSON for the
/// training tooling (they become benchmark ground truth and, at volume,
/// training data for a learned referee).
class LabelingScreen extends StatefulWidget {
  const LabelingScreen({
    super.key,
    this.manifestLoader = loadFootageManifest,
    this.labelStore,
  });

  /// Resolves the corpus entries. Defaults to the bundled manifest asset.
  final Future<List<FootageMatch>> Function() manifestLoader;

  /// Label persistence; defaults to the shared_preferences-backed store.
  final RallyLabelStore? labelStore;

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
                    clip: clips[i],
                    label: _labels[clips[i].id],
                    onWinner: (w) => _update(clips[i], winner: w),
                    onReason: (r) => _update(clips[i], reason: r),
                    onEndSeconds: (s) => _update(clips[i], endSeconds: s),
                    onWatch: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => MatchScreen(footage: clips[i].demo),
                      ),
                    ),
                  ),
                ),
    );
  }
}

class _RallyLabelCard extends StatelessWidget {
  const _RallyLabelCard({
    required this.clip,
    required this.label,
    required this.onWinner,
    required this.onReason,
    required this.onEndSeconds,
    required this.onWatch,
  });

  final FootageMatch clip;
  final RallyLabel? label;
  final ValueChanged<String> onWinner;
  final ValueChanged<RallyLabelReason> onReason;
  final ValueChanged<double> onEndSeconds;
  final VoidCallback onWatch;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final secs = (clip.durationMs / 1000).toStringAsFixed(0);
    final current = clip.truthSummary;
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
                  tooltip: 'Watch this rally',
                  icon: const Icon(Icons.play_circle_outline),
                  onPressed: onWatch,
                ),
              ],
            ),
            Text(
              '${secs}s · ${clip.bounces} bounces · pipeline says: '
              '${current ?? 'no call / unlabeled'}',
              style: theme.textTheme.bodySmall
                  ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
            ),
            const SizedBox(height: 8),
            SegmentedButton<String>(
              segments: const [
                ButtonSegment(value: 'a', label: Text('Player A (left)')),
                ButtonSegment(value: 'b', label: Text('Player B (right)')),
                ButtonSegment(value: 'unclear', label: Text('Unclear')),
              ],
              selected: {if (label != null) label!.winner},
              emptySelectionAllowed: true,
              onSelectionChanged: (sel) {
                if (sel.isNotEmpty) onWinner(sel.first);
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
                      if (r != null) onReason(r);
                    },
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: TextFormField(
                    initialValue: label?.endSeconds?.toStringAsFixed(1),
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                    ),
                    decoration: const InputDecoration(
                      labelText: 'Ends at (s)',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    onFieldSubmitted: (v) {
                      final s = double.tryParse(v);
                      if (s != null) onEndSeconds(s);
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
