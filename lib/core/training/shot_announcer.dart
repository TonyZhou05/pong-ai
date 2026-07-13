/// Pure-Dart spoken shot-grade announcer for training mode.
///
/// The match path gained a spoken umpire-style score call in iteration 108
/// ([MatchAnnouncer]) for the same headline reason the app exists: once the
/// phone is propped table-side the player is standing across the table and
/// cannot read the on-screen feed. Training mode had the mirror-image gap —
/// each stroke was graded and shown only *visually* in the recent-shots feed,
/// so a lone player drilling against a rebound net had no way to hear whether a
/// shot landed well without walking over to look. A coach standing court-side
/// calls the shot ("Good!", "Too short") out loud on each attempt; an audible
/// cue is what closes that loop for a table-side phone.
///
/// [ShotAnnouncer] is the pure, testable half of that: fed each freshly graded
/// [Shot], it returns the spoken call the stroke warrants — a grade call
/// ("Excellent shot!", "Good.", "Fair — a bit off.", "Off target.") plus an
/// encouraging streak call once several on-target shots land in a row
/// ("Good. 3 in a row!"). It is deliberately Flutter- and audio-free: the
/// actual speaking/haptic cue lives behind an injectable sink in the UI layer,
/// so a text-to-speech engine can be dropped in later without touching this
/// deterministic call logic, matching the [MatchAnnouncer] seam.
library;

import 'shot_analyzer.dart';

/// Turns a stream of graded [Shot]s into coach-style spoken calls.
///
/// Feed every completed shot via [onShot]; it returns the call to speak for that
/// stroke, tracking a running streak of on-target ([ShotGrade.good] or better)
/// shots so a hot streak earns an encouraging call-out. Unlike [MatchAnnouncer]
/// every shot is announce-worthy, so [onShot] never returns `null`.
class ShotAnnouncer {
  /// Number of consecutive on-target shots ending at the most recent one.
  int _streak = 0;

  /// Minimum consecutive on-target shots before the streak is called out.
  final int streakThreshold;

  ShotAnnouncer({this.streakThreshold = 3})
      : assert(streakThreshold >= 2, 'a streak needs at least two shots');

  /// The current on-target streak length (for the UI / tests).
  int get streak => _streak;

  /// The spoken call for [shot]: its grade phrase, suffixed with a streak
  /// call-out once [streakThreshold] on-target shots have landed in a row. A
  /// below-target shot ([ShotGrade.fair] or [ShotGrade.poor]) breaks the streak.
  String onShot(Shot shot) {
    final onTarget = shot.grade.index >= ShotGrade.good.index;
    _streak = onTarget ? _streak + 1 : 0;
    final call = _gradeCall(shot.grade);
    if (onTarget && _streak >= streakThreshold) {
      return '$call $_streak in a row!';
    }
    return call;
  }

  /// Forget the running streak so the next [onShot] starts fresh (used when a
  /// new drill session starts on the same screen).
  void reset() => _streak = 0;

  static String _gradeCall(ShotGrade grade) {
    switch (grade) {
      case ShotGrade.excellent:
        return 'Excellent shot!';
      case ShotGrade.good:
        return 'Good.';
      case ShotGrade.fair:
        return 'Fair — a bit off.';
      case ShotGrade.poor:
        return 'Off target.';
    }
  }
}
