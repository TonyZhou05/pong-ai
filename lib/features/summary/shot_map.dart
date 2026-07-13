/// Visual shot-map: the "Ball AI"-style placement map that draws where every
/// tracked bounce landed on a schematic top-down table.
///
/// [BouncePlacementAnalyzer] already mines the tracker's [BounceEvent]s into a
/// per-side placement distribution ([SidePlacementStats]), but until now that
/// was only surfaced as text counts (short/mid/deep). This widget renders those
/// bounces as a scatter/heat cloud on a top-down table so a player can *see*
/// their placement pattern — the headline analytics view of match apps.
///
/// Rendering is split from the geometry math: [shotMapPosition] is a pure
/// function mapping a [BouncePlacement] to normalized `[0,1] x [0,1]` table
/// coordinates (net at x = 0.5), so the placement math is unit-testable without
/// pixels, and [_ShotMapPainter] just paints those points.
library;

import 'package:flutter/material.dart';

import '../../core/analysis/ball_tracker.dart';
import '../../core/analysis/bounce_placement.dart';

/// Maps a [BouncePlacement] onto the schematic top-down table used by the shot
/// map, in normalized `[0,1] x [0,1]` coordinates where the net is the vertical
/// line at x = 0.5.
///
/// A bounce's [BouncePlacement.depthFromNet] runs 0 (at the net) → 1 (at that
/// side's baseline), so it fans *outward* from the centre net toward the left or
/// right edge depending on [BouncePlacement.side]. Its
/// [BouncePlacement.lateral] runs 0 (near/top edge) → 1 (far/bottom edge) and
/// maps straight to y.
({double x, double y}) shotMapPosition(BouncePlacement b) {
  final depth = b.depthFromNet.clamp(0.0, 1.0);
  final x = b.side == TableSide.left ? 0.5 - depth * 0.5 : 0.5 + depth * 0.5;
  return (x: x, y: b.lateral.clamp(0.0, 1.0));
}

/// A top-down table with a dot for every recorded bounce on either side.
///
/// Overlapping translucent dots build up into a density/heat cloud, so the
/// player can read their placement pattern at a glance. Renders an empty table
/// (with a hint) when no bounces have been recorded yet.
class ShotMapView extends StatelessWidget {
  const ShotMapView({
    super.key,
    required this.left,
    required this.right,
  });

  /// Placement stats for the left half of the table.
  final SidePlacementStats left;

  /// Placement stats for the right half of the table.
  final SidePlacementStats right;

  int get _totalBounces => left.count + right.count;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AspectRatio(
      aspectRatio: 2, // side-on table is wider than deep
      child: DecoratedBox(
        decoration: BoxDecoration(
          border: Border.all(color: Colors.white24),
          borderRadius: BorderRadius.circular(8),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: Stack(
            children: [
              Positioned.fill(
                child: CustomPaint(
                  painter: _ShotMapPainter(
                    bounces: [...left.bounces, ...right.bounces],
                    dotColor: theme.colorScheme.primary,
                  ),
                ),
              ),
              if (_totalBounces == 0)
                Center(
                  child: Text(
                    'No bounces tracked yet',
                    style: theme.textTheme.labelMedium
                        ?.copyWith(color: Colors.white60),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ShotMapPainter extends CustomPainter {
  _ShotMapPainter({required this.bounces, required this.dotColor});

  final List<BouncePlacement> bounces;
  final Color dotColor;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;

    // Table felt.
    canvas.drawRect(rect, Paint()..color = const Color(0xFF0D3B12));

    // Outer table lines + centre (long) line.
    final line = Paint()
      ..color = Colors.white54
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    final inset = rect.deflate(6);
    canvas.drawRect(inset, line);
    canvas.drawLine(
      Offset(inset.left, inset.center.dy),
      Offset(inset.right, inset.center.dy),
      Paint()
        ..color = Colors.white24
        ..strokeWidth = 1,
    );

    // Net line down the middle.
    canvas.drawLine(
      Offset(inset.center.dx, inset.top),
      Offset(inset.center.dx, inset.bottom),
      Paint()
        ..color = Colors.white
        ..strokeWidth = 2,
    );

    // Bounce dots (translucent so overlaps read as heat).
    final dot = Paint()..color = dotColor.withValues(alpha: 0.55);
    final radius = size.shortestSide * 0.035;
    for (final b in bounces) {
      final p = shotMapPosition(b);
      final center = Offset(
        inset.left + p.x * inset.width,
        inset.top + p.y * inset.height,
      );
      canvas.drawCircle(center, radius, dot);
    }
  }

  @override
  bool shouldRepaint(_ShotMapPainter old) =>
      old.bounces != bounces || old.dotColor != dotColor;
}
