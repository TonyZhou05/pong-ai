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
/// ("Excellent shot!", "Good.", "Fair — a bit off.", "Off target."), an
/// encouraging streak call once several on-target shots land in a row
/// ("Good. 3 in a row!"), and a session-best pace milestone when a shot beats
/// the fastest one so far ("Good. New top speed, 42 km/h!") — the training-mode
/// parity of the match announcer's climactic pressure cues. It is deliberately
/// Flutter- and audio-free: the
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

  /// Fastest shot pace (km/h) seen so far this session, 0 until a shot carries a
  /// physical speed (see [Shot.speedKmh]).
  double _bestSpeedKmh = 0;

  /// Minimum consecutive on-target shots before the streak is called out.
  final int streakThreshold;

  ShotAnnouncer({this.streakThreshold = 3})
      : assert(streakThreshold >= 2, 'a streak needs at least two shots');

  /// The current on-target streak length (for the UI / tests).
  int get streak => _streak;

  /// The fastest shot pace (km/h) called out so far this session, 0 if no shot
  /// has carried a physical speed yet.
  double get topSpeedKmh => _bestSpeedKmh;

  /// The spoken call for [shot]: its grade phrase, suffixed with a session-best
  /// pace milestone when the shot beats the fastest so far, then a streak
  /// call-out once [streakThreshold] on-target shots have landed in a row. A
  /// below-target shot ([ShotGrade.fair] or [ShotGrade.poor]) breaks the streak.
  String onShot(Shot shot) {
    final onTarget = shot.grade.index >= ShotGrade.good.index;
    _streak = onTarget ? _streak + 1 : 0;
    final parts = <String>[_gradeCall(shot.grade)];

    // Session-best pace milestone. Only fires once a baseline exists (the first
    // speed-bearing shot silently sets it, so the milestone isn't trivially true
    // on shot one) and stays silent when no physical scale is available
    // (speedKmh == 0, e.g. before table calibration provides a ruler).
    if (shot.speedKmh > 0) {
      if (_bestSpeedKmh > 0 && shot.speedKmh > _bestSpeedKmh) {
        parts.add('New top speed, ${shot.speedKmh.round()} km/h!');
      }
      if (shot.speedKmh > _bestSpeedKmh) _bestSpeedKmh = shot.speedKmh;
    }

    if (onTarget && _streak >= streakThreshold) {
      parts.add('$_streak in a row!');
    }
    return parts.join(' ');
  }

  /// Forget the running streak and session-best pace so the next [onShot] starts
  /// fresh (used when a new drill session starts on the same screen).
  void reset() {
    _streak = 0;
    _bestSpeedKmh = 0;
  }

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

/// A short spoken end-of-session summary, voiced when the player taps Finish so
/// a table-side phone reads back the drill result hands-free.
///
/// The per-shot [ShotAnnouncer] closes the "did that shot land well?" loop
/// during a drill, but when the session ends the player is typically walking
/// over to collect balls — across the table from the phone, unable to read the
/// end-of-session report. A concise spoken wrap-up (how many shots, the overall
/// grade, and the top pace when a physical scale is available) gives that final
/// feedback without a trip to the screen — the training-mode parity of the
/// match announcer's climactic "Match to Player A" call.
///
/// Returns a "no shots recorded" note for an empty session. Kept Flutter- and
/// audio-free like [ShotAnnouncer] so it is unit-testable and the actual
/// speaking stays behind the UI layer's injectable sink.
String spokenSessionSummary(TrainingSummary summary) {
  if (summary.shotCount == 0) return 'Session complete. No shots recorded.';
  final shotWord = summary.shotCount == 1 ? 'shot' : 'shots';
  final parts = <String>[
    'Session complete. ${summary.shotCount} $shotWord, '
        'grade ${summary.overallGrade}.',
  ];
  if (summary.maxSpeedKmh > 0) {
    parts.add('Top speed ${summary.maxSpeedKmh.round()} kilometres per hour.');
  }
  return parts.join(' ');
}
