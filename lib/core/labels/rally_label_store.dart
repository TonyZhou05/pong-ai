/// Human ground-truth labels for the real-footage rally corpus.
///
/// The labeling screen collects, per rally clip: who won, *why* (the losing
/// action, in ITTF terms), and a rough end time. These labels are the raw
/// material the pipeline learns from — they verify/correct the referee's
/// calls, become benchmark ground truth, and accumulate toward training a
/// learned referee. Stored via `shared_preferences` (works on web, where the
/// demo runs) as one JSON blob, and exportable as JSON for the training
/// tooling.
library;

import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

/// Why the rally ended — the losing player's action, in the vocabulary a
/// human labeler naturally uses. Mostly parallels the pipeline's
/// `PointReason`, deliberately richer where the pipeline can't distinguish
/// (into-net and serve faults both surface as unreturned balls in the
/// trajectory).
enum RallyLabelReason {
  notReturned('Missed / could not return the ball'),
  doubleBounce('Let it bounce twice on their side'),
  outOfBounds('Hit it out — long or wide'),
  intoNet('Hit it into the net'),
  serveFault('Faulted the serve (net or missed table)'),
  other('Other / unclear');

  const RallyLabelReason(this.description);

  /// Human description of what the LOSER did.
  final String description;
}

/// One rally's human label.
class RallyLabel {
  const RallyLabel({
    required this.clipId,
    required this.winner,
    this.reason,
    this.endSeconds,
    this.labeledAt,
  });

  /// The corpus clip id (e.g. `test_6_r3`).
  final String clipId;

  /// `a` (left player), `b` (right player), or `unclear`.
  final String winner;

  /// What the loser did, when known.
  final RallyLabelReason? reason;

  /// Rough time the rally ends within the clip, in seconds.
  final double? endSeconds;

  /// When the label was recorded (ISO-8601).
  final String? labeledAt;

  factory RallyLabel.fromJson(Map<String, dynamic> json) => RallyLabel(
        clipId: json['clipId'] as String,
        winner: json['winner'] as String,
        reason: RallyLabelReason.values
            .where((r) => r.name == json['reason'])
            .firstOrNull,
        endSeconds: (json['endSeconds'] as num?)?.toDouble(),
        labeledAt: json['labeledAt'] as String?,
      );

  Map<String, dynamic> toJson() => {
        'clipId': clipId,
        'winner': winner,
        if (reason != null) 'reason': reason!.name,
        if (endSeconds != null) 'endSeconds': endSeconds,
        if (labeledAt != null) 'labeledAt': labeledAt,
      };
}

/// Persistence seam for rally labels; tests inject the in-memory fake.
abstract class RallyLabelStore {
  Future<Map<String, RallyLabel>> load();
  Future<void> save(RallyLabel label);
  Future<void> remove(String clipId);

  /// The full label set as pretty JSON, for export to the training tooling.
  Future<String> exportJson() async {
    final labels = await load();
    final list = labels.values.map((l) => l.toJson()).toList()
      ..sort(
        (x, y) => (x['clipId'] as String).compareTo(y['clipId'] as String),
      );
    return const JsonEncoder.withIndent(' ').convert(list);
  }
}

/// `shared_preferences`-backed store (localStorage on web).
class PrefsRallyLabelStore extends RallyLabelStore {
  static const _key = 'rally_labels_v1';

  Future<Map<String, RallyLabel>> _read(SharedPreferences prefs) async {
    final raw = prefs.getString(_key);
    if (raw == null) return {};
    final list = jsonDecode(raw) as List<dynamic>;
    return {
      for (final e in list)
        (e as Map<String, dynamic>)['clipId'] as String: RallyLabel.fromJson(e),
    };
  }

  Future<void> _write(
    SharedPreferences prefs,
    Map<String, RallyLabel> labels,
  ) async {
    await prefs.setString(
      _key,
      jsonEncode(labels.values.map((l) => l.toJson()).toList()),
    );
  }

  @override
  Future<Map<String, RallyLabel>> load() async =>
      _read(await SharedPreferences.getInstance());

  @override
  Future<void> save(RallyLabel label) async {
    final prefs = await SharedPreferences.getInstance();
    final labels = await _read(prefs);
    labels[label.clipId] = label;
    await _write(prefs, labels);
  }

  @override
  Future<void> remove(String clipId) async {
    final prefs = await SharedPreferences.getInstance();
    final labels = await _read(prefs);
    labels.remove(clipId);
    await _write(prefs, labels);
  }
}

/// In-memory store for tests.
class InMemoryRallyLabelStore extends RallyLabelStore {
  final Map<String, RallyLabel> _labels = {};

  @override
  Future<Map<String, RallyLabel>> load() async => Map.of(_labels);

  @override
  Future<void> save(RallyLabel label) async => _labels[label.clipId] = label;

  @override
  Future<void> remove(String clipId) async => _labels.remove(clipId);
}
