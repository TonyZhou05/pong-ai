/// Pure-Dart post-match performance analysis.
///
/// The [RallyReferee] already explains *why* every point ended (a double
/// bounce, a ball that was never returned, or an open-play resolution). On its
/// own each verdict is transient — applied to the score and forgotten. This
/// layer retains the sequence of awarded points ([ScoredPoint]) and derives the
/// performance story from it ([MatchSummary]): who won, how many points came
/// from forcing the opponent into errors versus open play, each player's
/// longest run, and a human-readable report.
///
/// Like the rest of `core/`, it has no Flutter or vision dependencies, so the
/// analytics can be unit-tested against a synthetic point log.
library;

import '../scoring/scoring_engine.dart';
import 'rally_referee.dart';

/// One point awarded during a match, tagged with why the rally ended.
///
/// This is the durable record the referee's transient [PointDecision] becomes
/// once a winner is known (either auto-attributed or resolved by the user).
class ScoredPoint {
  const ScoredPoint({
    required this.winner,
    required this.reason,
    required this.timestampMs,
    this.server,
    this.gameIndex,
  });

  final Player winner;
  final PointReason reason;
  final int timestampMs;

  /// Who served this rally (the server *before* the point was awarded, since
  /// serving the point precedes winning it). Null when the server was not
  /// recorded — e.g. a [ScoredPoint] built by an older caller — so serve/receive
  /// analytics count only points where it is known.
  final Player? server;

  /// The 0-based index of the game this point belonged to (games completed
  /// *before* this point was awarded). Null when it was not recorded — e.g. a
  /// [ScoredPoint] built by an older caller — so per-game breakdowns only cover
  /// points where it is known.
  final int? gameIndex;

  @override
  String toString() =>
      'ScoredPoint($winner, $reason, @$timestampMs'
      '${server == null ? '' : ', serve $server'}'
      '${gameIndex == null ? '' : ', game $gameIndex'})';
}

/// The final point score of a single completed-or-in-progress game.
class GameScore {
  const GameScore({required this.pointsA, required this.pointsB});

  final int pointsA;
  final int pointsB;

  int pointsFor(Player p) => p == Player.a ? pointsA : pointsB;

  /// The game winner if it reached a decided margin (target reached with a
  /// 2-point lead), or null while it is still in progress.
  Player? winnerAt(int pointsPerGame) {
    final leader = pointsA >= pointsB ? pointsA : pointsB;
    final trailer = pointsA >= pointsB ? pointsB : pointsA;
    if (leader >= pointsPerGame && (leader - trailer) >= 2) {
      return pointsA > pointsB ? Player.a : Player.b;
    }
    return null;
  }

  @override
  String toString() => '$pointsA–$pointsB';
}

/// Aggregated performance analysis over a match's [ScoredPoint] log.
class MatchSummary {
  const MatchSummary({required this.points, required this.finalState});

  /// The ordered log of every awarded point.
  final List<ScoredPoint> points;

  /// The score at the moment the summary was taken.
  final MatchState finalState;

  /// Total points played (rallies with a decided winner).
  int get totalPoints => points.length;

  /// Points won by [p] across the whole match.
  int pointsWonBy(Player p) =>
      points.where((point) => point.winner == p).length;

  /// Points [p] won grouped by why the rally ended.
  Map<PointReason, int> reasonBreakdownFor(Player p) {
    final counts = <PointReason, int>{
      for (final reason in PointReason.values) reason: 0,
    };
    for (final point in points) {
      if (point.winner == p) counts[point.reason] = counts[point.reason]! + 1;
    }
    return counts;
  }

  /// Points [p] won because the opponent failed to keep the ball in play — a
  /// double bounce or a shot that was never returned. These reflect [p]
  /// forcing errors (pressure, placement, pace).
  int forcedErrorsWonBy(Player p) {
    final breakdown = reasonBreakdownFor(p);
    return breakdown[PointReason.doubleBounce]! +
        breakdown[PointReason.notReturned]!;
  }

  /// Points [p] won that were settled in open play (lost-in-flight rallies the
  /// user resolved manually, e.g. a smash winner or an out-of-play call).
  int openPlayPointsWonBy(Player p) =>
      reasonBreakdownFor(p)[PointReason.outOfPlay]!;

  /// Points played on [p]'s own serve (rallies [p] served), among the points
  /// whose server was recorded.
  int servePointsPlayedBy(Player p) =>
      points.where((point) => point.server == p).length;

  /// Points [p] served *and* won — the serve-hold count. In table tennis the
  /// server has the initiative, so this rate is a headline effectiveness stat.
  int servePointsWonBy(Player p) =>
      points.where((point) => point.server == p && point.winner == p).length;

  /// Points [p] won while *receiving* (the opponent served). These are the
  /// return-of-serve breaks.
  int receivePointsWonBy(Player p) => points
      .where((point) => point.server == p.other && point.winner == p)
      .length;

  /// Fraction of [p]'s own service points that [p] won, in `[0, 1]`, or null
  /// when [p] served no recorded points.
  double? serveWinRateFor(Player p) {
    final served = servePointsPlayedBy(p);
    if (served == 0) return null;
    return servePointsWonBy(p) / served;
  }

  /// Whether any point carried a recorded server, i.e. serve analytics are
  /// meaningful for this match.
  bool get hasServeData => points.any((point) => point.server != null);

  /// Whether any point carried a recorded game index, i.e. the per-game
  /// breakdown is meaningful for this match.
  bool get hasGameData => points.any((point) => point.gameIndex != null);

  /// The per-game point score line, reconstructed from the point log by
  /// counting each game's points won by each player. Games appear in play
  /// order; a trailing in-progress game (no winner yet) is included so the
  /// current game's running score shows too. Points with no recorded
  /// [ScoredPoint.gameIndex] are ignored, so this is empty when [hasGameData]
  /// is false.
  List<GameScore> get gameScores {
    final byGame = <int, List<int>>{}; // gameIndex -> [pointsA, pointsB]
    for (final point in points) {
      final g = point.gameIndex;
      if (g == null) continue;
      final tally = byGame.putIfAbsent(g, () => [0, 0]);
      if (point.winner == Player.a) {
        tally[0] += 1;
      } else {
        tally[1] += 1;
      }
    }
    final indices = byGame.keys.toList()..sort();
    return [
      for (final g in indices)
        GameScore(pointsA: byGame[g]![0], pointsB: byGame[g]![1]),
    ];
  }

  /// Whether a player scoring the next point from a pre-point score of
  /// ([forPoints], [againstPoints]) would win the game at [target] under ITTF
  /// rules (reach the target with a 2-point lead). Used to detect game-point
  /// situations.
  static bool _scoringWinsGame(int forPoints, int againstPoints, int target) {
    final after = forPoints + 1;
    return after >= target && (after - againstPoints) >= 2;
  }

  /// Walks the point log in order, reconstructing each game's running score, and
  /// tags every point with which player (if any) was at *game point* going into
  /// it — i.e. would have won the game by winning that point. At most one player
  /// can be at game point at a time (only the leader, at/after deuce), so a
  /// single nullable field captures it. Points with no recorded
  /// [ScoredPoint.gameIndex] can't be placed in a game, so they carry no game
  /// point.
  Iterable<({Player winner, Player? gamePointFor})> _gamePointOutcomes() sync* {
    final target = finalState.pointsPerGame;
    final runningByGame = <int, List<int>>{}; // gameIndex -> [pointsA, pointsB]
    for (final point in points) {
      final g = point.gameIndex;
      if (g == null) {
        yield (winner: point.winner, gamePointFor: null);
        continue;
      }
      final tally = runningByGame.putIfAbsent(g, () => [0, 0]);
      final pa = tally[0];
      final pb = tally[1];
      Player? gamePointFor;
      if (_scoringWinsGame(pa, pb, target)) {
        gamePointFor = Player.a;
      } else if (_scoringWinsGame(pb, pa, target)) {
        gamePointFor = Player.b;
      }
      yield (winner: point.winner, gamePointFor: gamePointFor);
      if (point.winner == Player.a) {
        tally[0] += 1;
      } else {
        tally[1] += 1;
      }
    }
  }

  /// Points on which [p] held a game point (a chance to close out the game by
  /// winning the rally).
  int gamePointsHeldBy(Player p) =>
      _gamePointOutcomes().where((e) => e.gamePointFor == p).length;

  /// Game points [p] held *and* converted (won to close the game). The
  /// game-point conversion rate is a headline table-tennis clutch stat.
  int gamePointsConvertedBy(Player p) => _gamePointOutcomes()
      .where((e) => e.gamePointFor == p && e.winner == p)
      .length;

  /// Points on which [p] *faced* a game point (the opponent could have closed
  /// the game by winning the rally).
  int gamePointsFacedBy(Player p) =>
      _gamePointOutcomes().where((e) => e.gamePointFor == p.other).length;

  /// Game points [p] faced *and* saved (won the rally to deny the opponent the
  /// game). The complement to [gamePointsConvertedBy].
  int gamePointsSavedBy(Player p) => _gamePointOutcomes()
      .where((e) => e.gamePointFor == p.other && e.winner == p)
      .length;

  /// Fraction of [p]'s own game points that [p] converted, in `[0, 1]`, or null
  /// when [p] held no game point.
  double? gamePointConversionRateFor(Player p) {
    final held = gamePointsHeldBy(p);
    if (held == 0) return null;
    return gamePointsConvertedBy(p) / held;
  }

  /// Whether any game-point situation occurred, i.e. pressure/clutch analytics
  /// are meaningful for this match. Requires per-game indexing to reconstruct
  /// the within-game running score.
  bool get hasPressureData =>
      hasGameData && _gamePointOutcomes().any((e) => e.gamePointFor != null);

  /// The longest run of consecutive points won by [p].
  int longestStreakFor(Player p) {
    var longest = 0;
    var current = 0;
    for (final point in points) {
      if (point.winner == p) {
        current += 1;
        if (current > longest) longest = current;
      } else {
        current = 0;
      }
    }
    return longest;
  }

  /// Wall-clock span between the first and last recorded point, in ms.
  int get durationMs => points.length < 2
      ? 0
      : points.last.timestampMs - points.first.timestampMs;

  /// The match winner, or null if the match is not over yet.
  Player? get matchWinner {
    if (!finalState.isMatchOver) return null;
    return finalState.gamesA > finalState.gamesB ? Player.a : Player.b;
  }

  static String _fmtDuration(int ms) {
    final totalSeconds = ms ~/ 1000;
    final minutes = totalSeconds ~/ 60;
    final seconds = totalSeconds % 60;
    return '$minutes:${seconds.toString().padLeft(2, '0')}';
  }

  static String _name(Player p) => p == Player.a ? 'Player A' : 'Player B';

  /// A deterministic, human-readable performance report.
  String report() {
    final lines = <String>['Match summary'];

    final winner = matchWinner;
    if (winner != null) {
      lines.add(
        '${_name(winner)} wins '
        '${finalState.gamesA}–${finalState.gamesB} in games.',
      );
    } else {
      lines.add(
        'In progress — games ${finalState.gamesA}–${finalState.gamesB}, '
        'points ${finalState.pointsA}–${finalState.pointsB}.',
      );
    }
    lines.add('$totalPoints points played over ${_fmtDuration(durationMs)}.');

    final games = gameScores;
    if (games.isNotEmpty) {
      lines.add('Games: ${games.join(', ')}.');
    }

    for (final player in Player.values) {
      lines
        ..add('')
        ..add('${_name(player)} — ${pointsWonBy(player)} pts')
        ..add('  • ${forcedErrorsWonBy(player)} won on forced errors')
        ..add('  • ${openPlayPointsWonBy(player)} won in open play')
        ..add('  • longest run: ${longestStreakFor(player)}');
      final serveRate = serveWinRateFor(player);
      if (serveRate != null) {
        lines.add(
          '  • serve points won: ${servePointsWonBy(player)}/'
          '${servePointsPlayedBy(player)} (${(serveRate * 100).round()}%)',
        );
      }
      if (hasPressureData) {
        lines.add(
          '  • game points: converted ${gamePointsConvertedBy(player)}/'
          '${gamePointsHeldBy(player)}, saved ${gamePointsSavedBy(player)}/'
          '${gamePointsFacedBy(player)}',
        );
      }
    }

    return lines.join('\n');
  }
}
