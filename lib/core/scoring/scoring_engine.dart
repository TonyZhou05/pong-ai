/// Pure-Dart implementation of the official table-tennis (ITTF) scoring rules.
///
/// This has **no Flutter or vision dependencies** so it can be unit-tested in
/// isolation and driven either by the live vision pipeline or by recorded
/// events from the benchmark harness.
library;

/// The two ends of the table. [a] is, by convention, the near side to the phone.
enum Player { a, b }

extension PlayerX on Player {
  Player get other => this == Player.a ? Player.b : Player.a;
}

/// Immutable snapshot of a match at a point in time.
class MatchState {
  const MatchState({
    required this.pointsA,
    required this.pointsB,
    required this.gamesA,
    required this.gamesB,
    required this.server,
    required this.initialServer,
    required this.pointsPerGame,
    required this.bestOf,
    required this.isMatchOver,
  });

  /// Points in the current game.
  final int pointsA;
  final int pointsB;

  /// Games won in the match.
  final int gamesA;
  final int gamesB;

  /// Who serves the next point.
  final Player server;

  /// Who served the very first point of the current game (used to derive the
  /// serve rotation deterministically).
  final Player initialServer;

  final int pointsPerGame;
  final int bestOf;
  final bool isMatchOver;

  int pointsFor(Player p) => p == Player.a ? pointsA : pointsB;
  int gamesFor(Player p) => p == Player.a ? gamesA : gamesB;

  MatchState copyWith({
    int? pointsA,
    int? pointsB,
    int? gamesA,
    int? gamesB,
    Player? server,
    Player? initialServer,
    bool? isMatchOver,
  }) {
    return MatchState(
      pointsA: pointsA ?? this.pointsA,
      pointsB: pointsB ?? this.pointsB,
      gamesA: gamesA ?? this.gamesA,
      gamesB: gamesB ?? this.gamesB,
      server: server ?? this.server,
      initialServer: initialServer ?? this.initialServer,
      pointsPerGame: pointsPerGame,
      bestOf: bestOf,
      isMatchOver: isMatchOver ?? this.isMatchOver,
    );
  }

  @override
  String toString() =>
      'MatchState(games $gamesA-$gamesB, points $pointsA-$pointsB, '
      'server $server${isMatchOver ? ', OVER' : ''})';
}

/// Applies ITTF scoring rules as points are awarded.
///
/// Rules implemented:
/// * A game is won at [pointsPerGame] (default 11) points with a 2-point lead.
/// * At 10-10 ("deuce") play continues until a player leads by 2.
/// * A match is won by first to `(bestOf / 2) + 1` games.
/// * Serve alternates every 2 points; during deuce it alternates every point.
class ScoringEngine {
  ScoringEngine({
    Player firstServer = Player.a,
    int pointsPerGame = 11,
    int bestOf = 5,
  })  : assert(pointsPerGame >= 1, 'pointsPerGame must be positive'),
        assert(bestOf.isOdd && bestOf >= 1, 'bestOf must be a positive odd number'),
        _state = MatchState(
          pointsA: 0,
          pointsB: 0,
          gamesA: 0,
          gamesB: 0,
          server: firstServer,
          initialServer: firstServer,
          pointsPerGame: pointsPerGame,
          bestOf: bestOf,
          isMatchOver: false,
        );

  MatchState _state;
  MatchState get state => _state;

  final List<MatchState> _history = [];

  int get _gamesToWinMatch => (_state.bestOf ~/ 2) + 1;

  /// Awards one point to [scorer] and advances match/game/serve state.
  /// Returns the new [MatchState]. No-op once the match is over.
  MatchState awardPoint(Player scorer) {
    if (_state.isMatchOver) return _state;

    _history.add(_state);

    var next = scorer == Player.a
        ? _state.copyWith(pointsA: _state.pointsA + 1)
        : _state.copyWith(pointsB: _state.pointsB + 1);

    if (_isGameWon(next.pointsA, next.pointsB, next.pointsPerGame)) {
      next = _completeGame(next, scorer);
    } else {
      next = next.copyWith(server: _serverFor(next));
    }

    _state = next;
    return _state;
  }

  /// Records who serves the very first point of the match.
  ///
  /// Valid only before the match has begun (no point awarded yet); returns
  /// `false` and changes nothing once play has started. The app defaults to
  /// [Player.a] serving, but the user (or a fine-tuned serve detector) can set
  /// the real first server so the serve rotation, the "who's serving" indicator
  /// and the serve/receive analytics are correct instead of always assuming A.
  bool setFirstServer(Player p) {
    if (_history.isNotEmpty ||
        _state.pointsA != 0 ||
        _state.pointsB != 0 ||
        _state.gamesA != 0 ||
        _state.gamesB != 0) {
      return false;
    }
    _state = _state.copyWith(server: p, initialServer: p);
    return true;
  }

  /// Reconfigures the match format (game length and/or best-of series length).
  ///
  /// Valid only before the match has begun (no point awarded yet); returns
  /// `false` and changes nothing once play has started or if the requested
  /// format is invalid ([pointsPerGame] must be positive, [bestOf] a positive
  /// odd number). The app defaults to 11-point games, best-of-5, but the user
  /// can pick a shorter/longer match (e.g. best-of-3 for casual play,
  /// best-of-7 for a full match) before scoring starts. Omitted arguments keep
  /// the current value.
  bool setMatchFormat({int? pointsPerGame, int? bestOf}) {
    if (_history.isNotEmpty ||
        _state.pointsA != 0 ||
        _state.pointsB != 0 ||
        _state.gamesA != 0 ||
        _state.gamesB != 0) {
      return false;
    }
    final ppg = pointsPerGame ?? _state.pointsPerGame;
    final bo = bestOf ?? _state.bestOf;
    if (ppg < 1 || bo < 1 || bo.isEven) return false;
    _state = MatchState(
      pointsA: 0,
      pointsB: 0,
      gamesA: 0,
      gamesB: 0,
      server: _state.server,
      initialServer: _state.initialServer,
      pointsPerGame: ppg,
      bestOf: bo,
      isMatchOver: false,
    );
    return true;
  }

  /// Undo the last [awardPoint]. Returns true if something was undone.
  bool undo() {
    if (_history.isEmpty) return false;
    _state = _history.removeLast();
    return true;
  }

  bool _isGameWon(int pa, int pb, int target) {
    final leader = pa >= pb ? pa : pb;
    final trailer = pa >= pb ? pb : pa;
    return leader >= target && (leader - trailer) >= 2;
  }

  MatchState _completeGame(MatchState afterPoint, Player gameWinner) {
    final gamesA = afterPoint.gamesA + (gameWinner == Player.a ? 1 : 0);
    final gamesB = afterPoint.gamesB + (gameWinner == Player.b ? 1 : 0);
    final matchOver =
        gamesA >= _gamesToWinMatch || gamesB >= _gamesToWinMatch;

    // Next game: server alternates from who started the previous game.
    final nextInitialServer = afterPoint.initialServer.other;

    return afterPoint.copyWith(
      pointsA: 0,
      pointsB: 0,
      gamesA: gamesA,
      gamesB: gamesB,
      server: nextInitialServer,
      initialServer: nextInitialServer,
      isMatchOver: matchOver,
    );
  }

  /// Deterministically derive who should serve given the current point count.
  ///
  /// Before deuce, serve switches every 2 points. At/after deuce (both players
  /// at [pointsPerGame] - 1), serve switches every single point.
  Player _serverFor(MatchState s) {
    final total = s.pointsA + s.pointsB;
    final deuce =
        s.pointsA >= s.pointsPerGame - 1 && s.pointsB >= s.pointsPerGame - 1;

    final switches = deuce
        // 2 serves each up to (target-1)*2 points, then 1 each.
        ? (s.pointsPerGame - 1) + (total - 2 * (s.pointsPerGame - 1))
        : total ~/ 2;

    return switches.isEven ? s.initialServer : s.initialServer.other;
  }

  /// Human-readable score for display / debugging, e.g. "1-0 (11-9, 5-3*)".
  String get scoreLine {
    final serveMark = _state.server == Player.a ? '*' : '';
    final serveMarkB = _state.server == Player.b ? '*' : '';
    return '${_state.gamesA}-${_state.gamesB} '
        '(${_state.pointsA}$serveMark-${_state.pointsB}$serveMarkB)';
  }
}
