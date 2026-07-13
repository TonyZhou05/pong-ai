/// Builds the app's default [SessionHistoryStore] rooted in the platform
/// documents directory.
///
/// [SessionHistoryStore] itself is pure Dart (it takes a [Directory]) so it can
/// be unit-tested against a temp dir; this file is the thin, plugin-touching
/// seam that resolves the *real* persistent location on device via
/// `path_provider`. Keeping it separate means every test injects its own store
/// and never has to touch the platform channel — importing this file is safe in
/// `flutter test`, only calling [defaultSessionHistoryStore] hits the plugin.
library;

import 'dart:io';

import 'package:path_provider/path_provider.dart';

import 'session_history_store.dart';

/// The sub-directory of the app documents directory that history files live in.
const String historyDirName = 'session_history';

/// Resolve the on-device [SessionHistoryStore] backed by
/// `<app documents>/session_history`. The directory is created lazily by the
/// store's first save, so this never touches the filesystem beyond asking
/// `path_provider` for the documents path.
Future<SessionHistoryStore> defaultSessionHistoryStore() async {
  final docs = await getApplicationDocumentsDirectory();
  final dir = Directory('${docs.path}${Platform.pathSeparator}$historyDirName');
  return SessionHistoryStore(dir);
}
