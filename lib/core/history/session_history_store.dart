/// On-disk history of completed match / training sessions.
///
/// The analytics pipeline already produces structured, JSON-encodable reports
/// (`buildMatchReportJson`, `buildTrainingReportJson`) — but until now the only
/// thing the app could do with one was copy it to the clipboard. Nothing was
/// *kept*: close the screen and the session is gone, so the objective's "keep
/// track of the scores … and produce summary" goal had no across-session memory.
/// Both JSON exporters even name this in their own docs ("stored as match
/// history, diffed across sessions") — the store was the missing piece.
///
/// [SessionHistoryStore] persists each report as a small wrapped JSON file in a
/// directory and lists / loads / deletes them back. It takes the target
/// [Directory] as a constructor argument (the app passes the platform documents
/// directory; a test passes a temp dir), so — like the rest of `core/` — it is
/// pure Dart (`dart:io` + `dart:convert`, no Flutter/plugin) and unit-testable
/// end-to-end without a device.
library;

import 'dart:convert';
import 'dart:io';

/// The kind of session a stored report came from.
enum SessionKind {
  match,
  training;

  String get key => name;

  static SessionKind? fromKey(Object? key) {
    for (final k in SessionKind.values) {
      if (k.key == key) return k;
    }
    return null;
  }
}

/// A single persisted session: its file [id], [kind], save time, and the
/// structured [report] map (exactly what `buildMatchReportJson` /
/// `buildTrainingReportJson` produced).
class StoredSession {
  const StoredSession({
    required this.id,
    required this.kind,
    required this.savedAt,
    required this.report,
  });

  /// The file stem (no `.json`), unique within the store and used by
  /// [SessionHistoryStore.load] / [SessionHistoryStore.delete].
  final String id;
  final SessionKind kind;
  final DateTime savedAt;
  final Map<String, dynamic> report;

  /// The wrapper written to disk: metadata around the raw report so a listing
  /// can show kind/time without parsing the whole (possibly large) report.
  Map<String, Object?> toWrapper() => {
        'kind': kind.key,
        'savedAt': savedAt.toIso8601String(),
        'report': report,
      };

  /// Parse a wrapper produced by [toWrapper]. Returns null if the shape is
  /// unrecognizable, so a corrupt / foreign file in the history directory is
  /// skipped rather than crashing a listing.
  static StoredSession? fromWrapper(String id, Object? decoded) {
    if (decoded is! Map) return null;
    final kind = SessionKind.fromKey(decoded['kind']);
    final savedAt = DateTime.tryParse('${decoded['savedAt']}');
    final report = decoded['report'];
    if (kind == null || savedAt == null || report is! Map) return null;
    return StoredSession(
      id: id,
      kind: kind,
      savedAt: savedAt,
      report: Map<String, dynamic>.from(report),
    );
  }
}

/// Persists [StoredSession]s as `<kind>-<millis>.json` files under [directory].
class SessionHistoryStore {
  SessionHistoryStore(this.directory);

  /// The directory history files live in. Created on first [save] if absent.
  final Directory directory;

  /// Save [report] as a new session, returning the persisted record.
  ///
  /// The id is derived from [at] (defaults to now); if a file with that id
  /// already exists (two saves in the same millisecond) a `-2`, `-3`, … suffix
  /// is appended so an earlier session is never overwritten.
  Future<StoredSession> save({
    required SessionKind kind,
    required Map<String, dynamic> report,
    DateTime? at,
  }) async {
    if (!directory.existsSync()) {
      directory.createSync(recursive: true);
    }
    final savedAt = at ?? DateTime.now();
    final base = '${kind.key}-${savedAt.millisecondsSinceEpoch}';
    var id = base;
    var attempt = 2;
    while (File(_pathFor(id)).existsSync()) {
      id = '$base-$attempt';
      attempt++;
    }
    final session = StoredSession(
      id: id,
      kind: kind,
      savedAt: savedAt,
      report: report,
    );
    await File(_pathFor(id)).writeAsString(
      const JsonEncoder.withIndent('  ').convert(session.toWrapper()),
    );
    return session;
  }

  /// All stored sessions, newest first (ties broken by id). Unparseable or
  /// foreign `*.json` files in [directory] are skipped, not surfaced.
  Future<List<StoredSession>> list() async {
    if (!directory.existsSync()) return const [];
    final sessions = <StoredSession>[];
    for (final entity in directory.listSync().whereType<File>()) {
      final path = entity.path;
      if (!path.toLowerCase().endsWith('.json')) continue;
      final id = _idFromPath(path);
      final StoredSession? session;
      try {
        session = StoredSession.fromWrapper(id, jsonDecode(await entity.readAsString()));
      } on FormatException {
        continue;
      }
      if (session != null) sessions.add(session);
    }
    sessions.sort((a, b) {
      final byTime = b.savedAt.compareTo(a.savedAt);
      return byTime != 0 ? byTime : b.id.compareTo(a.id);
    });
    return sessions;
  }

  /// Load a single session by [id], or null if it does not exist / is corrupt.
  Future<StoredSession?> load(String id) async {
    final file = File(_pathFor(id));
    if (!file.existsSync()) return null;
    try {
      return StoredSession.fromWrapper(id, jsonDecode(await file.readAsString()));
    } on FormatException {
      return null;
    }
  }

  /// Delete the session with [id]. Returns true if a file was removed.
  Future<bool> delete(String id) async {
    final file = File(_pathFor(id));
    if (!file.existsSync()) return false;
    await file.delete();
    return true;
  }

  String _pathFor(String id) => '${directory.path}${Platform.pathSeparator}$id.json';

  String _idFromPath(String path) {
    final name = path.split(Platform.pathSeparator).last;
    return name.endsWith('.json') ? name.substring(0, name.length - 5) : name;
  }
}
