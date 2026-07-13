/// Pure-Dart bounce-placement / shot-map analytics.
///
/// Match-analysis apps (e.g. "Ball AI") show a *placement map*: where on the
/// table each shot landed — short by the net vs. deep to the baseline, and how
/// spread the landings are across the table. That information is already in the
/// pipeline: [BallTracker] emits a [BounceEvent] (with the table-relative x/y and
/// [TableSide]) every time the ball touches the surface, but until now nothing
/// consumed it — [RallyAnalyzer] only counts net crossings, and the summary
/// layers ignore bounce locations entirely. This layer folds those bounces into
/// a per-side placement distribution: a depth-from-net profile (short / middle /
/// deep) plus lateral spread and a coarse grid heatmap.
///
/// Like the rest of `core/analysis`, it has no Flutter or vision-plugin
/// dependencies — it is driven purely by [TrackerEvent]s and a [TableGeometry],
/// so it is unit-testable against synthetic bounces and replayed benchmark clips.
///
/// Coordinate convention matches the pipeline: the frame is normalized to
/// `[0,1] x [0,1]` with the net a vertical line at [TableGeometry.netX]. With the
/// phone placed on the side of the table, the frame's **x** axis runs along the
/// table length (the net splits it) and the frame's **y** axis runs across the
/// table's near/far depth, so a bounce carries two independent placement axes:
///
/// * **depth-from-net** — how far past the net the ball landed, 0 at the net and
///   1 at that side's baseline (short serves vs. deep drives);
/// * **lateral** — where across the table's near/far depth it landed, 0 at the
///   surface's near (top) edge and 1 at its far (bottom) edge.
library;

import 'dart:math' as math;

import 'ball_tracker.dart';

/// Coarse depth zones for a bounce, measured from the net.
enum PlacementDepth {
  /// Landed in the third nearest the net.
  short,

  /// Landed in the middle third of the half.
  middle,

  /// Landed in the third nearest the baseline.
  deep,
}

/// One bounce located in table-relative coordinates.
class BouncePlacement {
  const BouncePlacement({
    required this.side,
    required this.depthFromNet,
    required this.lateral,
    required this.timestampMs,
  });

  /// Which half of the table the ball bounced on.
  final TableSide side;

  /// How far past the net the bounce landed, clamped to `[0,1]`: 0 at the net,
  /// 1 at this side's baseline.
  final double depthFromNet;

  /// Where across the table's near/far depth the bounce landed, clamped to
  /// `[0,1]`: 0 at the surface's near (top) edge, 1 at its far (bottom) edge.
  final double lateral;

  final int timestampMs;

  /// The coarse depth zone this bounce falls in (net-relative thirds).
  PlacementDepth get depthZone {
    if (depthFromNet < 1 / 3) return PlacementDepth.short;
    if (depthFromNet < 2 / 3) return PlacementDepth.middle;
    return PlacementDepth.deep;
  }

  @override
  String toString() => 'Bounce($side, depth=${depthFromNet.toStringAsFixed(2)}, '
      'lateral=${lateral.toStringAsFixed(2)})';
}

/// Placement distribution of every bounce on one side of the table.
class SidePlacementStats {
  const SidePlacementStats(this.side, this.bounces);

  final TableSide side;
  final List<BouncePlacement> bounces;

  int get count => bounces.length;

  /// Mean depth-from-net over the bounces (0 net … 1 baseline). Zero when empty.
  double get averageDepth => bounces.isEmpty
      ? 0
      : bounces.fold(0.0, (s, b) => s + b.depthFromNet) / bounces.length;

  /// Population standard deviation of depth-from-net — how consistently the
  /// player placed the ball at the same length. Zero when fewer than 2 bounces.
  double get depthConsistency {
    if (bounces.length < 2) return 0;
    final mean = averageDepth;
    final variance =
        bounces.fold(0.0, (s, b) => s + math.pow(b.depthFromNet - mean, 2)) /
            bounces.length;
    return math.sqrt(variance);
  }

  /// Span of lateral landing positions (max − min), a placement-width proxy.
  /// Zero when empty.
  double get lateralSpread {
    if (bounces.isEmpty) return 0;
    final xs = bounces.map((b) => b.lateral);
    return xs.reduce(math.max) - xs.reduce(math.min);
  }

  int _zoneCount(PlacementDepth z) =>
      bounces.where((b) => b.depthZone == z).length;

  int get shortCount => _zoneCount(PlacementDepth.short);
  int get middleCount => _zoneCount(PlacementDepth.middle);
  int get deepCount => _zoneCount(PlacementDepth.deep);

  /// A `depthBins × lateralBins` landing-count grid (the placement heatmap):
  /// `grid[d][l]` is how many bounces fell in depth band `d` (0 = nearest the
  /// net) and lateral band `l` (0 = near/top edge). Bins must be positive.
  List<List<int>> heatmap({int depthBins = 3, int lateralBins = 3}) {
    assert(depthBins > 0 && lateralBins > 0);
    final grid = List.generate(
      depthBins,
      (_) => List.filled(lateralBins, 0),
    );
    for (final b in bounces) {
      final d = math.min((b.depthFromNet * depthBins).floor(), depthBins - 1);
      final l = math.min((b.lateral * lateralBins).floor(), lateralBins - 1);
      grid[d][l]++;
    }
    return grid;
  }

  /// One-line human-readable placement summary for this side.
  String describe() {
    if (bounces.isEmpty) return '$side: no bounces';
    return '$side: $count bounces, avg depth ${averageDepth.toStringAsFixed(2)} '
        '($shortCount short / $middleCount mid / $deepCount deep), '
        'lateral spread ${lateralSpread.toStringAsFixed(2)}';
  }
}

/// Incrementally folds a match's [BounceEvent]s into per-side placement stats.
///
/// Feed each [TrackerEvent] (as produced by [BallTracker.update]) to [observe];
/// non-bounce events are ignored. Read the accumulated [statsFor] a side at any
/// time. The [geometry] must match the tracker's so net/edge-relative
/// coordinates line up — the controller rebuilds this on the calibrated geometry
/// exactly like it does the movement analytics.
class BouncePlacementAnalyzer {
  BouncePlacementAnalyzer({this.geometry = const TableGeometry()});

  final TableGeometry geometry;

  final Map<TableSide, List<BouncePlacement>> _bounces = {
    TableSide.left: [],
    TableSide.right: [],
  };

  /// Fold one tracker event; only [BounceEvent]s contribute.
  void observe(TrackerEvent event) {
    if (event is! BounceEvent) return;
    _bounces[event.side]!.add(_place(event));
  }

  BouncePlacement _place(BounceEvent b) {
    final depth = b.side == TableSide.left
        ? (geometry.netX - b.x) / (geometry.netX - geometry.left)
        : (b.x - geometry.netX) / (geometry.right - geometry.netX);
    final lateral =
        (b.y - geometry.top) / (geometry.bottom - geometry.top);
    return BouncePlacement(
      side: b.side,
      depthFromNet: depth.clamp(0.0, 1.0),
      lateral: lateral.clamp(0.0, 1.0),
      timestampMs: b.timestampMs,
    );
  }

  /// Placement distribution of every bounce recorded on [side] so far.
  SidePlacementStats statsFor(TableSide side) =>
      SidePlacementStats(side, List.of(_bounces[side]!));

  /// Forget all recorded bounces.
  void reset() {
    _bounces[TableSide.left]!.clear();
    _bounces[TableSide.right]!.clear();
  }
}
