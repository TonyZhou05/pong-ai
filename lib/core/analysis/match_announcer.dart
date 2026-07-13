/// Pure-Dart umpire-style score announcer.
///
/// The objective's headline deployment story is "place the phone on the side of
/// the table and let the app keep score" — but once the phone is propped
/// table-side, the players are standing across the table and cannot read the
/// on-screen scoreboard. Every prior iteration surfaced the score only
/// *visually* (the scoreboard, banners, overlays), so a player has no way to
/// know the app registered a point without walking over to look. Real apps in
/// this space (and every human umpire) *call the score out loud* after each
/// rally — an audible cue is what actually closes the loop for a table-side
/// phone.
///
/// [MatchAnnouncer] is the pure, testable half of that: fed each new
/// [MatchState] as the score advances, it emits the spoken call for whatever
/// just happened — a point call ("Player A, 5–3."), a game call
/// ("Game to Player A. 1 game all."), or a match call ("Match to Player A, 3
/// games to 2.") — or `null` when nothing announce-worthy changed (an unchanged
/// frame, or an undo that walked the score *back*). When a point leaves a side
/// one point from the game or match, the point call is suffixed with a spoken
/// pressure cue ("Player A, 10–8. Game point Player A.") — the audible parity of
/// the visual game-point/match-point banner, so a player who can't read the
/// scoreboard still hears the climax coming.
///
/// It is deliberately Flutter- and audio-free: the actual speaking/haptic cue
/// lives behind an injectable sink in the UI layer, so a text-to-speech engine
/// can be dropped in later as a one-line change (the same injectable-seam
/// pattern as `VisionModelProfile`) without touching this deterministic call
/// logic, which is unit-tested against synthetic score sequences.
library;

import '../scoring/match_situation.dart';
import '../scoring/scoring_engine.dart';

/// Turns a stream of [MatchState] snapshots into umpire-style spoken calls.
///
/// Feed every new score state via [onState]; it compares against the previously
/// seen state and returns the call the *forward* transition warrants (or `null`
/// for no change / a backward step such as an undo). The first call just seeds
/// the baseline and returns `null`.
class MatchAnnouncer {
  MatchState? _last;

  /// The spoken call for the transition from the previously seen state to
  /// [state], or `null` when nothing announce-worthy advanced (an unchanged
  /// state, or a score that stepped *backwards* via undo). Always `null` on the
  /// very first observation, which just records the baseline.
  String? onState(MatchState state) {
    final prev = _last;
    _last = state;
    if (prev == null) return null;

    // Match just ended — the headline call takes precedence over the game call
    // that shares the same transition.
    if (state.isMatchOver && !prev.isMatchOver) {
      final winner = state.gamesA > state.gamesB ? Player.a : Player.b;
      final hi = state.gamesFor(winner);
      final lo = state.gamesFor(winner.other);
      return 'Match to ${_name(winner)}, $hi games to $lo.';
    }

    // A game completed (games total went up) without ending the match.
    if (state.gamesA + state.gamesB > prev.gamesA + prev.gamesB) {
      final winner = state.gamesA > prev.gamesA ? Player.a : Player.b;
      return 'Game to ${_name(winner)}. ${_gamesStanding(state)}.';
    }

    // A point was scored within the current game (points total went up).
    if (state.pointsA + state.pointsB > prev.pointsA + prev.pointsB) {
      return _pointCall(state);
    }

    // No forward change (unchanged frame, or an undo stepped the score back).
    return null;
  }

  /// Forget the baseline so the next [onState] just re-seeds (used when a fresh
  /// match starts on the same screen, so the reset to 0–0 isn't announced).
  void reset() => _last = null;

  /// The within-game point call: the leading player named first, or "N all" on
  /// a tie — friendlier for a spoken cue than the strict server-first umpire
  /// numeric, since the app tracks anonymous seats A/B rather than named
  /// players. When the new score leaves a side one point from the game or the
  /// match, the call is suffixed with a spoken pressure cue ("Game point Player
  /// A." / "Match point Player B.") — the audible parity of the visual
  /// game-point/match-point banner, which is exactly what a player standing
  /// across the table (who can't read the scoreboard) needs to hear.
  String _pointCall(MatchState s) {
    final String base;
    if (s.pointsA == s.pointsB) {
      base = '${s.pointsA} all.';
    } else {
      final leader = s.pointsA > s.pointsB ? Player.a : Player.b;
      final hi = s.pointsFor(leader);
      final lo = s.pointsFor(leader.other);
      base = '${_name(leader)}, $hi–$lo.';
    }
    final pressure = _pressureCue(s);
    return pressure == null ? base : '$base $pressure';
  }

  /// The spoken game-point / match-point cue for [s], or `null` when neither
  /// side is one point away. Mirrors [MatchSituation] (which mirrors the
  /// [ScoringEngine] win rule) so the cue can never disagree with the score, and
  /// voices the count ("Double game point Player A.", "Triple match point Player
  /// B.") the same way the visual banner does.
  String? _pressureCue(MatchState s) {
    final situation = MatchSituation(s);
    final candidate = situation.candidate;
    if (candidate == null) return null;
    final kind = situation.isMatchPoint ? 'match point' : 'game point';
    final n = situation.pointCount;
    final String phrase;
    switch (n) {
      case 1:
        phrase = kind;
      case 2:
        phrase = 'double $kind';
      case 3:
        phrase = 'triple $kind';
      default:
        phrase = '$n ${kind}s';
    }
    final capitalized = phrase[0].toUpperCase() + phrase.substring(1);
    return '$capitalized ${_name(candidate)}.';
  }

  /// The games-standing clause of a game call ("1 game all", "Player A leads 2
  /// games to 1").
  String _gamesStanding(MatchState s) {
    if (s.gamesA == s.gamesB) {
      return '${s.gamesA} ${s.gamesA == 1 ? 'game' : 'games'} all';
    }
    final leader = s.gamesA > s.gamesB ? Player.a : Player.b;
    final hi = s.gamesFor(leader);
    final lo = s.gamesFor(leader.other);
    return '${_name(leader)} leads $hi games to $lo';
  }

  static String _name(Player p) => p == Player.a ? 'Player A' : 'Player B';
}
