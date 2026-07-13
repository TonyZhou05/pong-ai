import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/history/history_store_provider.dart';
import '../../core/history/session_history_store.dart';
import '../../core/history/session_trends.dart';
import '../../core/share/report_share.dart';
import 'progress_chart.dart';

/// Browse, view and delete previously-saved match / training sessions.
///
/// The analytics pipeline has produced structured JSON reports since
/// iterations 38/39 and [SessionHistoryStore] (iteration 45) persisted them —
/// but nothing ever *read* them back, so the across-session history the exports
/// were built for had no surface. This screen is that consumer: it lists the
/// stored sessions newest-first, opens any one to view its full report, and can
/// delete it.
///
/// The [store] is injectable (tests pass a temp-dir store); by default the
/// screen resolves the on-device store lazily via [defaultSessionHistoryStore]
/// so `flutter test` never touches the `path_provider` plugin.
class SessionHistoryScreen extends StatefulWidget {
  const SessionHistoryScreen({
    super.key,
    this.store,
    this.storeLoader,
    this.shareReport = defaultShareReport,
  });

  /// A ready store to use directly (tests). When null, [storeLoader] resolves
  /// one lazily.
  final SessionHistoryStore? store;

  /// Resolves the store when [store] is null. Defaults to the on-device
  /// documents-directory store.
  final Future<SessionHistoryStore> Function()? storeLoader;

  /// Sink for the OS share sheet, injectable so the aggregate-report Share
  /// action stays widget-testable headlessly. Defaults to the `share_plus`
  /// backed [defaultShareReport].
  final ShareReportSink shareReport;

  @override
  State<SessionHistoryScreen> createState() => _SessionHistoryScreenState();
}

class _SessionHistoryScreenState extends State<SessionHistoryScreen> {
  SessionHistoryStore? _store;
  List<StoredSession>? _sessions;
  Object? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _sessions = null;
      _error = null;
    });
    try {
      final store = widget.store ??
          _store ??
          await (widget.storeLoader ?? defaultSessionHistoryStore)();
      final sessions = await store.list();
      if (!mounted) return;
      setState(() {
        _store = store;
        _sessions = sessions;
      });
    } on Object catch (e) {
      if (!mounted) return;
      setState(() => _error = e);
    }
  }

  Future<void> _delete(StoredSession session) async {
    final store = _store;
    if (store == null) return;
    await store.delete(session.id);
    await _load();
  }

  /// The aggregate cross-session progress + career summary as shareable text,
  /// or null when nothing is saved yet. Composed from the same [SessionTrends]
  /// that backs the on-screen Training-progress / Match-record cards.
  String? _progressReport() {
    final sessions = _sessions;
    if (sessions == null || sessions.isEmpty) return null;
    return SessionTrends.fromSessions(sessions).report();
  }

  Future<void> _copyProgress(BuildContext context) async {
    final report = _progressReport();
    if (report == null) return;
    final messenger = ScaffoldMessenger.of(context);
    await Clipboard.setData(ClipboardData(text: report));
    messenger.showSnackBar(
      const SnackBar(content: Text('Progress copied to clipboard')),
    );
  }

  Future<void> _shareProgress() async {
    final report = _progressReport();
    if (report == null) return;
    await widget.shareReport(report, subject: 'Table tennis progress');
  }

  @override
  Widget build(BuildContext context) {
    final hasSessions = (_sessions?.isNotEmpty ?? false);
    return Scaffold(
      appBar: AppBar(
        title: const Text('History'),
        actions: [
          if (hasSessions) ...[
            IconButton(
              icon: const Icon(Icons.copy),
              tooltip: 'Copy progress',
              onPressed: () => _copyProgress(context),
            ),
            IconButton(
              icon: const Icon(Icons.ios_share),
              tooltip: 'Share progress',
              onPressed: _shareProgress,
            ),
          ],
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Reload',
            onPressed: _load,
          ),
        ],
      ),
      body: SafeArea(child: _body(context)),
    );
  }

  Widget _body(BuildContext context) {
    final theme = Theme.of(context);
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            "Couldn't load history.\n$_error",
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium,
          ),
        ),
      );
    }
    final sessions = _sessions;
    if (sessions == null) {
      return const Center(child: CircularProgressIndicator());
    }
    if (sessions.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            'No saved sessions yet.\nFinish a match or training drill and tap '
            '"Save to history".',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium,
          ),
        ),
      );
    }
    final trends = SessionTrends.fromSessions(sessions);
    // The header cards scroll together with the session list (rather than
    // sitting fixed above an Expanded list) so a tall stack — records banner +
    // trends header + match card — never overflows on a short screen.
    return CustomScrollView(
      slivers: [
        if (trends.latestSessionSetRecord)
          SliverToBoxAdapter(child: _RecordsBanner(trends: trends)),
        if (trends.hasTrainingTrend)
          SliverToBoxAdapter(child: _TrendsHeader(trends: trends)),
        if (trends.hasMatchData)
          SliverToBoxAdapter(child: _MatchSummaryCard(trends: trends)),
        _sessionSliver(sessions),
      ],
    );
  }

  Widget _sessionSliver(List<StoredSession> sessions) {
    return SliverList.separated(
      itemCount: sessions.length,
      separatorBuilder: (_, __) => const Divider(height: 1),
      itemBuilder: (context, i) {
        final session = sessions[i];
        return ListTile(
          leading: Icon(
            session.kind == SessionKind.match
                ? Icons.sports_tennis
                : Icons.fitness_center,
          ),
          title: Text(sessionHeadline(session)),
          subtitle: Text(formatSessionTime(session.savedAt)),
          trailing: IconButton(
            icon: const Icon(Icons.delete_outline),
            tooltip: 'Delete',
            onPressed: () => _delete(session),
          ),
          onTap: () => Navigator.of(context).push(
            MaterialPageRoute<void>(
              builder: (_) => _SessionDetailScreen(session: session),
            ),
          ),
        );
      },
    );
  }
}

/// Celebratory banner shown above the trend cards when the most-recently-saved
/// session set at least one new career best (see
/// [SessionTrends.latestSessionRecords]).
class _RecordsBanner extends StatelessWidget {
  const _RecordsBanner({required this.trends});

  final SessionTrends trends;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final records = trends.latestSessionRecords;
    return Card(
      margin: const EdgeInsets.fromLTRB(12, 12, 12, 4),
      color: theme.colorScheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.emoji_events,
                  color: theme.colorScheme.onPrimaryContainer,
                ),
                const SizedBox(width: 8),
                Text(
                  'New personal record${records.length > 1 ? 's' : ''}!',
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: theme.colorScheme.onPrimaryContainer,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            for (final r in records)
              Padding(
                padding: const EdgeInsets.only(top: 2),
                child: Text(
                  '• $r',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: theme.colorScheme.onPrimaryContainer,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// Compact across-session training-progression card shown above the list once
/// there are at least two saved drills to trend between.
class _TrendsHeader extends StatelessWidget {
  const _TrendsHeader({required this.trends});

  final SessionTrends trends;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final improvement = trends.scoreImprovement;
    final first = trends.firstSession!;
    final latest = trends.latestSession!;
    String pct(double v) => '${(v * 100).round()}%';
    final delta = improvement ?? 0;
    final improving = delta > 0.0005;
    final declining = delta < -0.0005;
    final color = improving
        ? Colors.green
        : (declining ? theme.colorScheme.error : theme.colorScheme.onSurface);
    return Card(
      margin: const EdgeInsets.fromLTRB(12, 12, 12, 4),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Training progress', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            Row(
              children: [
                Icon(
                  improving
                      ? Icons.trending_up
                      : (declining ? Icons.trending_down : Icons.trending_flat),
                  color: color,
                ),
                const SizedBox(width: 8),
                Text(
                  '${pct(first.averageScore)} → ${pct(latest.averageScore)}',
                  style: theme.textTheme.titleLarge?.copyWith(color: color),
                ),
                const SizedBox(width: 8),
                Text(
                  '(${delta >= 0 ? '+' : ''}${(delta * 100).round()}%)',
                  style: theme.textTheme.bodyMedium?.copyWith(color: color),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Best: grade ${trends.bestSession!.overallGrade} · '
              '${trends.trainingCount} drills · '
              '${trends.totalShotsPracticed} shots'
              '${trends.bestMaxSpeedKmh != null ? ' · fastest ${trends.bestMaxSpeedKmh!.toStringAsFixed(1)} km/h' : ''}'
              '${trends.averageShotSpeedKmh != null ? ' · typical ${trends.averageShotSpeedKmh!.toStringAsFixed(1)} km/h' : ''}'
              '${(trends.bestOnTargetStreak ?? 0) >= 2 ? ' · streak ${trends.bestOnTargetStreak}' : ''}',
              style: theme.textTheme.bodyMedium,
            ),
            if (_consistencyLine(trends) case final line?) ...[
              const SizedBox(height: 4),
              Text(line, style: theme.textTheme.bodyMedium),
            ],
            if (trends.hasRecurringFocus) ...[
              const SizedBox(height: 4),
              Row(
                children: [
                  Icon(
                    Icons.center_focus_strong,
                    size: 18,
                    color: theme.colorScheme.primary,
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      'Keep working on ${trends.recurringFocus!.toLowerCase()} '
                      '(${trends.recurringFocusCount}/${trends.trainingCount} drills)',
                      style: theme.textTheme.bodyMedium,
                    ),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 12),
            ProgressChartView(sessions: trends.trainingSessions),
          ],
        ),
      ),
    );
  }

  /// A compact "Placement tighter · Rhythm up 12%" style line summarising the
  /// consistency progression, or null if neither metric has enough sessions.
  static String? _consistencyLine(SessionTrends trends) {
    final parts = <String>[];
    final depth = trends.depthConsistencyImprovement;
    if (depth != null && depth.abs() > 0.0005) {
      parts.add('Placement ${depth > 0 ? 'tighter' : 'looser'}');
    }
    final lateral = trends.lateralConsistencyImprovement;
    if (lateral != null && lateral.abs() > 0.0005) {
      parts.add('Lateral ${lateral > 0 ? 'tighter' : 'looser'}');
    }
    final rhythm = trends.rhythmConsistencyImprovement;
    if (rhythm != null && rhythm.abs() > 0.0005) {
      parts.add(
        'Rhythm ${rhythm > 0 ? 'up' : 'down'} '
        '${(rhythm.abs() * 100).round()}%',
      );
    }
    final speed = trends.speedImprovement;
    if (speed != null && speed.abs() > 0.05) {
      parts.add(
        'Speed ${speed > 0 ? 'up' : 'down'} '
        '${speed.abs().toStringAsFixed(1)} km/h',
      );
    }
    final accuracy = trends.accuracyImprovement;
    if (accuracy != null && accuracy.abs() > 0.005) {
      parts.add(
        'Accuracy ${accuracy > 0 ? 'up' : 'down'} '
        '${(accuracy.abs() * 100).round()}%',
      );
    }
    return parts.isEmpty ? null : parts.join(' · ');
  }
}

/// Compact cumulative match "career" totals card, shown above the list once at
/// least one match has been saved. Matches pit Player A vs B (no single tracked
/// user to trend), so this surfaces cumulative bests instead of a progression.
class _MatchSummaryCard extends StatelessWidget {
  const _MatchSummaryCard({required this.trends});

  final SessionTrends trends;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final chips = <String>[];
    final points = trends.totalMatchPoints;
    if (points != null) chips.add('$points points');
    final playtime = trends.totalMatchDurationMs;
    if (playtime != null) {
      chips.add('${playtime ~/ 60000}m played');
    }
    final rally = trends.longestMatchRallyStrokes;
    if (rally != null) chips.add('longest rally $rally');
    final avgRally = trends.averageMatchRallyStrokes;
    if (avgRally != null) {
      chips.add('avg rally ${avgRally.toStringAsFixed(1)}');
    }
    final rallyTrend = trends.matchRallyLengthImprovement;
    if (rallyTrend != null && rallyTrend.abs() > 0.05) {
      chips.add(
        'rally ${rallyTrend > 0 ? '↑' : '↓'} ${rallyTrend.abs().toStringAsFixed(1)}',
      );
    }
    final speed = trends.fastestMatchBallSpeedKmh;
    if (speed != null) chips.add('fastest ${speed.toStringAsFixed(1)} km/h');
    final avgSpeed = trends.averageMatchBallSpeedKmh;
    if (avgSpeed != null) chips.add('avg ${avgSpeed.toStringAsFixed(1)} km/h');
    final paceTrend = trends.matchBallSpeedImprovement;
    if (paceTrend != null && paceTrend.abs() > 0.05) {
      chips.add(
        'pace ${paceTrend > 0 ? '↑' : '↓'} ${paceTrend.abs().toStringAsFixed(1)} km/h',
      );
    }
    return Card(
      margin: const EdgeInsets.fromLTRB(12, 4, 12, 4),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.sports_tennis, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text('Match record', style: theme.textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              '${trends.matchCount} '
              '${trends.matchCount == 1 ? 'match' : 'matches'} played',
              style: theme.textTheme.titleLarge,
            ),
            if (trends.hasMatchWinRecord) ...[
              const SizedBox(height: 4),
              Text(
                "Head-to-head · A ${trends.matchWinsBy('A')}–"
                "${trends.matchWinsBy('B')} B"
                "${trends.longestMatchWinStreak >= 2 ? ' · streak '
                    '${trends.longestMatchWinStreakSeat} '
                    '${trends.longestMatchWinStreak}' : ''}"
                "${trends.currentMatchWinStreak >= 2 ? ' · now '
                    '${trends.currentMatchWinStreakSeat} '
                    '${trends.currentMatchWinStreak}' : ''}",
                style: theme.textTheme.bodyMedium,
              ),
            ],
            if (chips.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(chips.join(' · '), style: theme.textTheme.bodyMedium),
            ],
            if (trends.hasRecurringMatchFocus) ...[
              const SizedBox(height: 4),
              Row(
                children: [
                  Icon(
                    Icons.center_focus_strong,
                    size: 18,
                    color: theme.colorScheme.primary,
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      'Match focus: ${trends.recurringMatchFocus!.toLowerCase()} '
                      '(${trends.recurringMatchFocusCount}/${trends.matchCount} matches)',
                      style: theme.textTheme.bodyMedium,
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Read-only view of one stored session's full structured report.
class _SessionDetailScreen extends StatelessWidget {
  const _SessionDetailScreen({required this.session});

  final StoredSession session;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final pretty =
        const JsonEncoder.withIndent('  ').convert(session.report);
    return Scaffold(
      appBar: AppBar(title: Text(sessionHeadline(session))),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                formatSessionTime(session.savedAt),
                style: theme.textTheme.labelLarge,
              ),
              const SizedBox(height: 12),
              SelectableText(
                pretty,
                style: theme.textTheme.bodySmall
                    ?.copyWith(fontFamily: 'monospace'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A one-line headline for a stored session, derived from the report contents
/// (final score for a match, grade + shot count for a drill). Pure so it is
/// unit-testable without rendering.
String sessionHeadline(StoredSession session) {
  final report = session.report;
  switch (session.kind) {
    case SessionKind.match:
      final score = report['score'];
      if (score is Map) {
        final gamesA = score['gamesA'];
        final gamesB = score['gamesB'];
        if (gamesA is num && gamesB is num && (gamesA > 0 || gamesB > 0)) {
          return 'Match · $gamesA–$gamesB games';
        }
        final pointsA = score['pointsA'];
        final pointsB = score['pointsB'];
        if (pointsA is num && pointsB is num) {
          return 'Match · $pointsA–$pointsB';
        }
      }
      return 'Match';
    case SessionKind.training:
      final s = report['session'];
      if (s is Map) {
        final grade = s['overallGrade'];
        final shots = s['shotCount'];
        if (grade is String && shots is num) {
          return 'Training · grade $grade · $shots shots';
        }
      }
      return 'Training';
  }
}

/// Format a session's save time as a compact local timestamp.
String formatSessionTime(DateTime at) {
  final local = at.toLocal();
  String two(int n) => n.toString().padLeft(2, '0');
  return '${local.year}-${two(local.month)}-${two(local.day)} '
      '${two(local.hour)}:${two(local.minute)}';
}
