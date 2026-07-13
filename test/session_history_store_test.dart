import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/history/session_history_store.dart';

void main() {
  late Directory dir;
  late SessionHistoryStore store;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('pong_history_test');
    store = SessionHistoryStore(dir);
  });

  tearDown(() {
    if (dir.existsSync()) dir.deleteSync(recursive: true);
  });

  Map<String, dynamic> report(int a, int b) => {
        'schemaVersion': 1,
        'score': {'pointsA': a, 'pointsB': b},
      };

  group('save', () {
    test('creates the directory if it does not exist and writes a file',
        () async {
      final nested = Directory('${dir.path}${Platform.pathSeparator}nested');
      final s = SessionHistoryStore(nested);
      expect(nested.existsSync(), isFalse);

      final saved = await s.save(kind: SessionKind.match, report: report(11, 9));

      expect(nested.existsSync(), isTrue);
      expect(
        File('${nested.path}${Platform.pathSeparator}${saved.id}.json')
            .existsSync(),
        isTrue,
      );
    });

    test('derives the id from kind and save time', () async {
      final at = DateTime.fromMillisecondsSinceEpoch(1720000000000);
      final saved = await store.save(
        kind: SessionKind.training,
        report: report(0, 0),
        at: at,
      );
      expect(saved.id, 'training-1720000000000');
      expect(saved.kind, SessionKind.training);
      expect(saved.savedAt, at);
    });

    test('never overwrites an earlier same-millisecond session', () async {
      final at = DateTime.fromMillisecondsSinceEpoch(1720000000000);
      final first = await store.save(kind: SessionKind.match, report: report(1, 0), at: at);
      final second = await store.save(kind: SessionKind.match, report: report(2, 0), at: at);

      expect(first.id, 'match-1720000000000');
      expect(second.id, 'match-1720000000000-2');
      final all = await store.list();
      expect(all, hasLength(2));
    });
  });

  group('list', () {
    test('returns an empty list for a never-written store', () async {
      expect(await store.list(), isEmpty);
    });

    test('returns saved sessions newest first', () async {
      await store.save(
        kind: SessionKind.match,
        report: report(11, 5),
        at: DateTime.fromMillisecondsSinceEpoch(1000),
      );
      await store.save(
        kind: SessionKind.training,
        report: report(0, 0),
        at: DateTime.fromMillisecondsSinceEpoch(3000),
      );
      await store.save(
        kind: SessionKind.match,
        report: report(11, 9),
        at: DateTime.fromMillisecondsSinceEpoch(2000),
      );

      final all = await store.list();
      expect(all.map((s) => s.savedAt.millisecondsSinceEpoch), [3000, 2000, 1000]);
      expect(all.first.kind, SessionKind.training);
    });

    test('round-trips the report map', () async {
      await store.save(kind: SessionKind.match, report: report(11, 9));
      final loaded = (await store.list()).single;
      expect(loaded.report['score'], {'pointsA': 11, 'pointsB': 9});
    });

    test('skips corrupt / foreign json files in the directory', () async {
      await store.save(kind: SessionKind.match, report: report(11, 0));
      // A malformed json file and a well-formed but non-session file.
      File('${dir.path}${Platform.pathSeparator}garbage.json')
          .writeAsStringSync('{not json');
      File('${dir.path}${Platform.pathSeparator}foreign.json')
          .writeAsStringSync(jsonEncode({'hello': 'world'}));

      final all = await store.list();
      expect(all, hasLength(1));
      expect(all.single.kind, SessionKind.match);
    });
  });

  group('load / delete', () {
    test('load returns the stored session and null for a missing id', () async {
      final saved = await store.save(kind: SessionKind.match, report: report(11, 3));
      final loaded = await store.load(saved.id);
      expect(loaded, isNotNull);
      expect(loaded!.report['score'], {'pointsA': 11, 'pointsB': 3});
      expect(await store.load('does-not-exist'), isNull);
    });

    test('delete removes the file and reports whether one existed', () async {
      final saved = await store.save(kind: SessionKind.match, report: report(0, 0));
      expect(await store.delete(saved.id), isTrue);
      expect(await store.load(saved.id), isNull);
      expect(await store.list(), isEmpty);
      expect(await store.delete(saved.id), isFalse);
    });
  });
}
