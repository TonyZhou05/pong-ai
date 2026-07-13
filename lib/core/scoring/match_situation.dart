/// Pure-Dart derivation of the *live* game-point / match-point situation from a
/// [MatchState] — the headline "you're one away" scoreboard cue that apps like
/// Ball AI flash during play.
///
/// This has **no Flutter or vision dependencies** so it can be unit-tested in
/// isolation and reused by any scoreboard surface. It reads only the current
/// snapshot; it does not advance or mutate the score.
library;

import 'scoring_engine.dart';

/// The pressure level of the point about to be played.
enum PointPressure {
  /// Neither side is one point from winning the current game.
  none,

  /// A player would win the *current game* (but not the match) with the next
  /// point.
  gamePoint,

  /// A player would win the *whole match* with the next point.
  matchPoint,
}

/// Interprets a [MatchState] into whether a side is at game/match point, who,
/// and how many consecutive chances they hold (e.g. 3 at 10–7 → "triple game
/// point"). Mirrors the [ScoringEngine] win rule (reach [MatchState.pointsPerGame]
/// with a two-point lead) so the cue can never disagree with the scoring.
class MatchSituation {
  const MatchSituation(this.state);

  final MatchState state;

  int get _gamesToWinMatch => (state.bestOf ~/ 2) + 1;

  /// Whether awarding [p] a single point would win the current game, per the
  /// same rule [ScoringEngine] applies.
  bool _wouldWinGame(Player p) {
    final leader = state.pointsFor(p) + 1;
    final trailer = state.pointsFor(p.other);
    return leader >= state.pointsPerGame && (leader - trailer) >= 2;
  }

  /// The player one point away from winning the current game, or `null` if
  /// neither side is at game point (or the match is already over). At most one
  /// side can qualify, since winning requires a two-point lead.
  Player? get candidate {
    if (state.isMatchOver) return null;
    for (final p in Player.values) {
      if (_wouldWinGame(p)) return p;
    }
    return null;
  }

  /// Whether some side is one point from winning the current game.
  bool get isGamePoint => candidate != null;

  /// Whether the [candidate]'s next-point game win would also decide the match.
  bool get isMatchPoint {
    final p = candidate;
    if (p == null) return false;
    return state.gamesFor(p) + 1 >= _gamesToWinMatch;
  }

  /// Whether the current game is the *deciding game* — the final possible game
  /// of the match, with both players one game short of winning (e.g. 2–2 in a
  /// best-of-5, 1–1 in a best-of-3). Unlike [isMatchPoint] (one *point* away),
  /// this holds for the whole final game, so the scoreboard can flag "we're into
  /// the decider" from the game's first point. A best-of-1 has no decider
  /// context (its sole game is trivially last), so this is always false there.
  bool get isDecidingGame {
    if (state.isMatchOver) return false;
    final needed = _gamesToWinMatch;
    if (needed < 2) return false;
    return state.gamesA == needed - 1 && state.gamesB == needed - 1;
  }

  /// The pressure level of the next point.
  PointPressure get pressure {
    if (candidate == null) return PointPressure.none;
    return isMatchPoint ? PointPressure.matchPoint : PointPressure.gamePoint;
  }

  /// How many consecutive chances the [candidate] holds — e.g. 3 at 10–7
  /// ("triple game point"), 1 at 11–10. Zero when no side is at game point.
  int get pointCount {
    final p = candidate;
    if (p == null) return 0;
    return state.pointsFor(p) - state.pointsFor(p.other);
  }

  /// A short scoreboard banner, e.g. "Match point A", "Double game point B",
  /// or `null` when neither side is at game/match point.
  String? get label {
    final p = candidate;
    if (p == null) return null;
    final who = p == Player.a ? 'A' : 'B';
    final kind = isMatchPoint ? 'match point' : 'game point';
    final n = pointCount;
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
    return '$capitalized $who';
  }

  /// The scoreboard banner to display: the game/match-point [label] when a side
  /// is one point away, otherwise "Deciding game" for the whole final game, or
  /// `null` when there is no notable context. Game/match point takes precedence
  /// so the climax of the decider still reads "Match point", not just the
  /// persistent decider context.
  String? get bannerLabel {
    return label ?? (isDecidingGame ? 'Deciding game' : null);
  }
}
