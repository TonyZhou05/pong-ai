/// Visual **training** shot-map: where every graded stroke landed on the target
/// half, the practice-mode counterpart to the match [ShotMapView].
///
/// Match mode gets a placement map from [BouncePlacementAnalyzer]; training mode
/// only ever surfaced its [Shot]s as a text feed and aggregate numbers. Now that
/// each [Shot] carries both its landing [Shot.depth] (net → baseline) and
/// [Shot.lateral] (across the table), we can draw the same "Ball AI"-style
/// placement cloud for a drill — so the player can *see* how tightly their shots
/// cluster around the target and where they scatter.
///
/// As with [ShotMapView], the geometry math is split out as a pure, unit-testable
/// [trainingShotMapPosition] function; [_TrainingShotMapPainter] just draws the
/// points and the target band.
library;

import 'package:flutter/material.dart';

import '../../core/training/shot_analyzer.dart';

/// Maps a [Shot] onto the schematic target half used by the training shot-map,
/// in normalized `[0,1] x [0,1]` coordinates: **x** is the landing depth (0 at
/// the net on the left edge, 1 at the baseline on the right edge) and **y** is
/// the lateral position across the table (0 near/top edge, 1 far/bottom edge).
({double x, double y}) trainingShotMapPosition(Shot s) =>
    (x: s.depth.clamp(0.0, 1.0), y: s.lateral.clamp(0.0, 1.0));

/// A schematic target-half table with a grade-coloured dot for every recorded
/// [Shot], plus the target landing band highlighted, so a player can read their
/// placement pattern across a drill at a glance.
///
/// Renders an empty table (with a hint) when no shots have been recorded yet.
class TrainingShotMapView extends StatelessWidget {
  const TrainingShotMapView({
    super.key,
    required this.shots,
    this.config = const TrainingConfig(),
  });

  /// Every graded stroke recorded in the session.
  final List<Shot> shots;

  /// The drill's target definition (used to draw the target depth band).
  final TrainingConfig config;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AspectRatio(
      aspectRatio: 2, // target half, drawn wider than deep
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
                  painter: _TrainingShotMapPainter(
                    shots: shots,
                    targetDepth: config.targetDepth,
                    depthTolerance: config.depthTolerance,
                    theme: theme,
                  ),
                ),
              ),
              if (shots.isEmpty)
                Center(
                  child: Text(
                    'No shots tracked yet',
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

class _TrainingShotMapPainter extends CustomPainter {
  _TrainingShotMapPainter({
    required this.shots,
    required this.targetDepth,
    required this.depthTolerance,
    required this.theme,
  });

  final List<Shot> shots;
  final double targetDepth;
  final double depthTolerance;
  final ThemeData theme;

  static Color _gradeColor(ShotGrade g) => switch (g) {
        ShotGrade.excellent => const Color(0xFF2ECC71),
        ShotGrade.good => const Color(0xFF48C9B0),
        ShotGrade.fair => const Color(0xFFF4D03F),
        ShotGrade.poor => const Color(0xFFE74C3C),
      };

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;

    // Table felt.
    canvas.drawRect(rect, Paint()..color = const Color(0xFF0D3B12));

    final inset = rect.deflate(6);

    // Target landing band: a vertical stripe at targetDepth ± tolerance.
    final near = (targetDepth - depthTolerance).clamp(0.0, 1.0);
    final far = (targetDepth + depthTolerance).clamp(0.0, 1.0);
    canvas.drawRect(
      Rect.fromLTRB(
        inset.left + near * inset.width,
        inset.top,
        inset.left + far * inset.width,
        inset.bottom,
      ),
      Paint()..color = const Color(0x3348C9B0),
    );

    // Outer table lines + centre (long) line.
    final line = Paint()
      ..color = Colors.white54
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    canvas.drawRect(inset, line);
    canvas.drawLine(
      Offset(inset.left, inset.center.dy),
      Offset(inset.right, inset.center.dy),
      Paint()
        ..color = Colors.white24
        ..strokeWidth = 1,
    );

    // Net on the near (left) edge; target depth line at its centre.
    canvas.drawLine(
      Offset(inset.left, inset.top),
      Offset(inset.left, inset.bottom),
      Paint()
        ..color = Colors.white
        ..strokeWidth = 2,
    );

    // Shot dots (grade-coloured, translucent so overlaps read as density).
    final radius = size.shortestSide * 0.05;
    for (final s in shots) {
      final p = trainingShotMapPosition(s);
      final center = Offset(
        inset.left + p.x * inset.width,
        inset.top + p.y * inset.height,
      );
      canvas.drawCircle(
        center,
        radius,
        Paint()..color = _gradeColor(s.grade).withValues(alpha: 0.7),
      );
    }
  }

  @override
  bool shouldRepaint(_TrainingShotMapPainter old) =>
      old.shots != shots ||
      old.targetDepth != targetDepth ||
      old.depthTolerance != depthTolerance;
}
