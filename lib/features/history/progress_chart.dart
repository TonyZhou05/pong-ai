/// Visual training-progress chart: a line of average shot-quality score across
/// the saved training sessions, oldest → latest.
///
/// [SessionTrends] (iteration 48/49) already folds the saved history into a
/// first→latest score delta and surfaces it as a compact text card, but the
/// *shape* of the progression — steady climb, plateau, a dip and recovery — was
/// never drawn. This widget plots one point per saved drill so a player can see
/// their trajectory at a glance, the across-session analog of the match-side
/// [MomentumChartView] and the shot-map cloud.
///
/// Rendering is split from the math: [progressChartPoints] is a pure function
/// mapping the per-session score series to normalized `[0,1] x [0,1]` plot
/// coordinates (unit-testable without pixels), and [_ProgressPainter] just draws
/// them.
library;

import 'package:flutter/material.dart';

import '../../core/history/session_trends.dart';

/// Normalized `[0,1] x [0,1]` plot points for a training-score progression line,
/// one per session in the order given (oldest → latest).
///
/// x spreads the sessions evenly left → right; a lone session sits at x = 0.5.
/// y maps each `[0,1]` average score with **0 at the bottom, 1 at the top** (so
/// a rising line reads as improving), clamping out-of-range scores.
List<({double x, double y})> progressChartPoints(List<double> scores) {
  final n = scores.length;
  final out = <({double x, double y})>[];
  for (var i = 0; i < n; i++) {
    final x = n == 1 ? 0.5 : i / (n - 1);
    final y = 1 - scores[i].clamp(0.0, 1.0);
    out.add((x: x, y: y));
  }
  return out;
}

/// A line chart of average shot-quality score across saved training sessions.
///
/// Shows an empty axis (with a hint) until at least one training session exists.
class ProgressChartView extends StatelessWidget {
  const ProgressChartView({super.key, required this.sessions});

  /// Training sessions oldest → latest (from [SessionTrends.trainingSessions]).
  final List<TrainingTrendPoint> sessions;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scores = [for (final s in sessions) s.averageScore];
    return AspectRatio(
      aspectRatio: 2.4,
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
                  painter: _ProgressPainter(
                    points: progressChartPoints(scores),
                    lineColor: theme.colorScheme.primary,
                  ),
                ),
              ),
              if (sessions.isEmpty)
                Center(
                  child: Text(
                    'No training drills saved yet',
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

class _ProgressPainter extends CustomPainter {
  _ProgressPainter({required this.points, required this.lineColor});

  /// Normalized plot points (oldest → latest); y is 0 at the bottom.
  final List<({double x, double y})> points;

  /// Colour of the progression line and its session dots.
  final Color lineColor;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    canvas.drawRect(rect, Paint()..color = const Color(0xFF15171C));

    final inset = rect.deflate(8);

    // Gridlines at 0% / 50% / 100% score so the vertical scale is readable.
    final gridPaint = Paint()
      ..color = Colors.white24
      ..strokeWidth = 1;
    for (final frac in const [0.0, 0.5, 1.0]) {
      final y = inset.top + frac * inset.height;
      canvas.drawLine(
        Offset(inset.left, y),
        Offset(inset.right, y),
        gridPaint,
      );
    }

    if (points.isEmpty) return;

    Offset pixelAt(({double x, double y}) p) => Offset(
          inset.left + p.x * inset.width,
          inset.top + p.y * inset.height,
        );

    // The progression line (skipped for a single session, which is just a dot).
    if (points.length >= 2) {
      final line = Path()..moveTo(pixelAt(points.first).dx, pixelAt(points.first).dy);
      for (var i = 1; i < points.length; i++) {
        final p = pixelAt(points[i]);
        line.lineTo(p.dx, p.dy);
      }
      canvas.drawPath(
        line,
        Paint()
          ..color = lineColor
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }

    // A dot per session, with the latest one emphasized.
    final dotPaint = Paint()..color = lineColor;
    for (var i = 0; i < points.length; i++) {
      final center = pixelAt(points[i]);
      final isLatest = i == points.length - 1;
      canvas.drawCircle(center, isLatest ? 4 : 3, dotPaint);
      if (isLatest) {
        canvas.drawCircle(
          center,
          6,
          Paint()
            ..color = lineColor.withValues(alpha: 0.3)
            ..style = PaintingStyle.fill,
        );
      }
    }
  }

  @override
  bool shouldRepaint(_ProgressPainter old) =>
      old.points != points || old.lineColor != lineColor;
}
