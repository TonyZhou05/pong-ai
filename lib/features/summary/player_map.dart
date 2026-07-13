/// Visual player-positioning map: the "Ball AI"-style court-coverage heatmap
/// that draws where each player actually stood over a match.
///
/// [PlayerMovementAnalyzer] mines every frame's pose into per-player footwork
/// metrics, but the aggregate [PlayerMovementStats] only exposed *summaries*
/// (distance, coverage span, average position). The individual foot-position
/// samples — the raw material for a positioning heatmap — were discarded. The
/// analyzer now retains them ([PlayerMovementAnalyzer.positionsFor]), and this
/// widget renders them as a per-player scatter/heat cloud on a schematic
/// top-down table so a player can *see* their court coverage: how far they
/// ranged, whether they held their ground or drifted, and where their "home"
/// base was.
///
/// Rendering is split from the geometry math: [playerMapPosition] is a pure
/// function mapping a raw frame foot position to normalized `[0,1] x [0,1]`
/// table coordinates with the net re-centred to x = 0.5 (each half scaled
/// independently around the calibrated [netX]), so the mapping is unit-testable
/// without pixels and [_PlayerMapPainter] just paints those points.
library;

import 'package:flutter/material.dart';

import '../../core/analysis/player_movement.dart';
import '../../core/scoring/scoring_engine.dart';

/// Maps a raw frame foot position onto the schematic top-down table used by the
/// positioning map, in normalized `[0,1] x [0,1]` coordinates where the net is
/// the vertical line at x = 0.5.
///
/// With the phone on the side of the table the frame's x runs along the table
/// length (the net splits it at [netX]) and y runs across the near/far depth.
/// Each half is scaled independently around [netX] so the net always renders at
/// the centre regardless of where the calibrator placed it: `[0, netX]` maps to
/// `[0, 0.5]` and `[netX, 1]` maps to `[0.5, 1]`. y passes straight through.
({double x, double y}) playerMapPosition(FramePoint foot, {double netX = 0.5}) {
  final fx = foot.x.clamp(0.0, 1.0);
  final n = netX.clamp(1e-6, 1 - 1e-6);
  final x = fx <= n ? 0.5 * (fx / n) : 0.5 + 0.5 * ((fx - n) / (1 - n));
  return (x: x.clamp(0.0, 1.0), y: foot.y.clamp(0.0, 1.0));
}

/// A top-down table with a heat cloud per player showing where they stood.
///
/// Overlapping translucent dots build up into a density map, tinted one colour
/// per player, so their court coverage reads at a glance. Renders an empty
/// table (with a hint) when neither player was tracked.
class PlayerPositionMapView extends StatelessWidget {
  const PlayerPositionMapView({
    super.key,
    required this.positions,
    this.netX = 0.5,
  });

  /// Foot-position samples per player (frame coordinates), as produced by
  /// [PlayerMovementAnalyzer.positionsFor].
  final Map<Player, List<FramePoint>> positions;

  /// The calibrated net line (frame x), used to re-centre each half on the map.
  final double netX;

  int get _totalSamples =>
      positions.values.fold(0, (s, list) => s + list.length);

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
                  painter: _PlayerMapPainter(
                    positions: positions,
                    netX: netX,
                    colorA: theme.colorScheme.primary,
                    colorB: theme.colorScheme.tertiary,
                  ),
                ),
              ),
              if (_totalSamples == 0)
                Center(
                  child: Text(
                    'No player movement tracked yet',
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

class _PlayerMapPainter extends CustomPainter {
  _PlayerMapPainter({
    required this.positions,
    required this.netX,
    required this.colorA,
    required this.colorB,
  });

  final Map<Player, List<FramePoint>> positions;
  final double netX;
  final Color colorA;
  final Color colorB;

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

    // One translucent heat cloud per player.
    final radius = size.shortestSide * 0.045;
    for (final player in Player.values) {
      final color = player == Player.a ? colorA : colorB;
      final dot = Paint()..color = color.withValues(alpha: 0.4);
      for (final foot in positions[player] ?? const <FramePoint>[]) {
        final p = playerMapPosition(foot, netX: netX);
        final center = Offset(
          inset.left + p.x * inset.width,
          inset.top + p.y * inset.height,
        );
        canvas.drawCircle(center, radius, dot);
      }
    }
  }

  @override
  bool shouldRepaint(_PlayerMapPainter old) =>
      old.positions != positions ||
      old.netX != netX ||
      old.colorA != colorA ||
      old.colorB != colorB;
}
