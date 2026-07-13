import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/analysis/ball_tracker.dart';
import 'package:pong_ai/core/analysis/bounce_placement.dart';
import 'package:pong_ai/core/analysis/match_controller.dart';
import 'package:pong_ai/core/vision/detection.dart';

FrameResult _ball(int t, double x, double y) => FrameResult(
      timestampMs: t,
      ball: Detection(label: 'ball', confidence: 0.9, box: BBox(x, y, 0, 0)),
    );

BounceEvent _bounce(double x, double y, TableSide side, {int t = 0}) =>
    BounceEvent(t, x, y, side);

void main() {
  group('BouncePlacement coordinates', () {
    // Default geometry: net at 0.5, full-frame surface [0,1]x[0,1].
    final analyzer = BouncePlacementAnalyzer();

    test('right-side bounce at the net is depth 0, at the baseline is depth 1',
        () {
      analyzer
        ..reset()
        ..observe(_bounce(0.5, 0.5, TableSide.right)) // at the net
        ..observe(_bounce(1.0, 0.5, TableSide.right)); // at the baseline
      final b = analyzer.statsFor(TableSide.right).bounces;
      expect(b[0].depthFromNet, closeTo(0.0, 1e-9));
      expect(b[1].depthFromNet, closeTo(1.0, 1e-9));
    });

    test('left-side depth is measured from the net toward the left baseline',
        () {
      analyzer
        ..reset()
        ..observe(_bounce(0.25, 0.5, TableSide.left)); // halfway to baseline
      final b = analyzer.statsFor(TableSide.left).bounces.single;
      expect(b.depthFromNet, closeTo(0.5, 1e-9));
    });

    test('lateral is the y position within the surface band', () {
      analyzer
        ..reset()
        ..observe(_bounce(0.7, 0.2, TableSide.right));
      expect(
        analyzer.statsFor(TableSide.right).bounces.single.lateral,
        closeTo(0.2, 1e-9),
      );
    });

    test('depth zone classification splits into net-relative thirds', () {
      PlacementDepth zoneOf(double x) {
        final a = BouncePlacementAnalyzer()
          ..observe(_bounce(x, 0.5, TableSide.right));
        return a.statsFor(TableSide.right).bounces.single.depthZone;
      }

      expect(zoneOf(0.55), PlacementDepth.short); // depth 0.1
      expect(zoneOf(0.75), PlacementDepth.middle); // depth 0.5
      expect(zoneOf(0.95), PlacementDepth.deep); // depth 0.9
    });

    test('uses the calibrated net/edges for depth', () {
      // Net at 0.6, right edge at 0.9: a bounce at 0.75 is halfway to baseline.
      final a = BouncePlacementAnalyzer(
        geometry: const TableGeometry(netX: 0.6, left: 0.1, right: 0.9),
      )..observe(_bounce(0.75, 0.5, TableSide.right));
      expect(
        a.statsFor(TableSide.right).bounces.single.depthFromNet,
        closeTo(0.5, 1e-9),
      );
    });

    test('clamps out-of-band coordinates into [0,1]', () {
      // A bounce reported left of the left edge would give negative depth.
      final a = BouncePlacementAnalyzer(
        geometry: const TableGeometry(left: 0.2, right: 0.8),
      )..observe(_bounce(0.1, 1.5, TableSide.left));
      final b = a.statsFor(TableSide.left).bounces.single;
      expect(b.depthFromNet, inInclusiveRange(0.0, 1.0));
      expect(b.lateral, 1.0);
    });
  });

  group('SidePlacementStats aggregates', () {
    test('separates bounces by side and ignores non-bounce events', () {
      final a = BouncePlacementAnalyzer()
        ..observe(_bounce(0.7, 0.5, TableSide.right))
        ..observe(_bounce(0.8, 0.5, TableSide.right))
        ..observe(_bounce(0.3, 0.5, TableSide.left))
        ..observe(const NetCrossEvent(0, TableSide.left, TableSide.right))
        ..observe(const BallLostEvent(0));
      expect(a.statsFor(TableSide.right).count, 2);
      expect(a.statsFor(TableSide.left).count, 1);
    });

    test('averageDepth and zone counts summarise placement', () {
      final a = BouncePlacementAnalyzer()
        ..observe(_bounce(0.55, 0.5, TableSide.right)) // short, depth 0.1
        ..observe(_bounce(0.75, 0.5, TableSide.right)) // middle, depth 0.5
        ..observe(_bounce(0.95, 0.5, TableSide.right)); // deep, depth 0.9
      final s = a.statsFor(TableSide.right);
      expect(s.averageDepth, closeTo(0.5, 1e-9));
      expect(s.shortCount, 1);
      expect(s.middleCount, 1);
      expect(s.deepCount, 1);
    });

    test('lateralSpread is the max-min of landing y', () {
      final a = BouncePlacementAnalyzer()
        ..observe(_bounce(0.7, 0.2, TableSide.right))
        ..observe(_bounce(0.7, 0.9, TableSide.right));
      expect(a.statsFor(TableSide.right).lateralSpread, closeTo(0.7, 1e-9));
    });

    test('depthConsistency is 0 for <2 bounces and the stddev otherwise', () {
      final one = BouncePlacementAnalyzer()
        ..observe(_bounce(0.75, 0.5, TableSide.right));
      expect(one.statsFor(TableSide.right).depthConsistency, 0);

      // Two bounces at depth 0.0 and 1.0: mean 0.5, stddev 0.5.
      final two = BouncePlacementAnalyzer()
        ..observe(_bounce(0.5, 0.5, TableSide.right))
        ..observe(_bounce(1.0, 0.5, TableSide.right));
      expect(two.statsFor(TableSide.right).depthConsistency, closeTo(0.5, 1e-9));
    });

    test('empty side reports zeros and a no-bounces description', () {
      final s = BouncePlacementAnalyzer().statsFor(TableSide.left);
      expect(s.count, 0);
      expect(s.averageDepth, 0);
      expect(s.lateralSpread, 0);
      expect(s.describe(), contains('no bounces'));
    });

    test('heatmap counts bounces into a depth × lateral grid', () {
      final a = BouncePlacementAnalyzer()
        ..observe(_bounce(0.55, 0.1, TableSide.right)) // depth band 0, lat 0
        ..observe(_bounce(0.95, 0.9, TableSide.right)); // depth band 2, lat 2
      final grid = a.statsFor(TableSide.right).heatmap();
      expect(grid, hasLength(3));
      expect(grid[0][0], 1);
      expect(grid[2][2], 1);
      // Every other cell is empty.
      final total = grid.expand((r) => r).fold(0, (s, c) => s + c);
      expect(total, 2);
    });
  });

  group('reset', () {
    test('forgets all bounces on both sides', () {
      final a = BouncePlacementAnalyzer()
        ..observe(_bounce(0.7, 0.5, TableSide.right))
        ..observe(_bounce(0.3, 0.5, TableSide.left))
        ..reset();
      expect(a.statsFor(TableSide.right).count, 0);
      expect(a.statsFor(TableSide.left).count, 0);
    });
  });

  group('MatchController integration', () {
    test('records a real table bounce with the right side and depth', () {
      final controller = MatchController();
      // Ball crosses to the right side, then arcs down-and-up (a bounce) deep on
      // the right, then is lost -> a notReturned point for Player.a.
      final frames = <FrameResult>[
        _ball(0, 0.30, 0.50),
        _ball(33, 0.55, 0.50), // crosses the net onto the right side
        _ball(66, 0.85, 0.45),
        _ball(99, 0.85, 0.60), // descending
        _ball(132, 0.85, 0.55), // ascending -> bounce apex at t=99 (right, x=0.85)
        for (var i = 0; i < 8; i++) FrameResult(timestampMs: 165 + i * 33),
      ];
      for (final f in frames) {
        controller.onFrame(f);
      }
      final right = controller.placementFor(TableSide.right);
      expect(right.count, 1);
      // x=0.85 with net 0.5, right edge 1.0 -> depth 0.7 -> deep.
      expect(right.bounces.single.depthZone, PlacementDepth.deep);
      expect(controller.placementFor(TableSide.left).count, 0);
      expect(controller.score.pointsA, 1);
    });
  });
}
