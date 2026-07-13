/// Visual momentum chart: the score-progression timeline that shows how the
/// lead swung between the two players over the course of a match.
///
/// [MatchSummary] already retains the ordered log of every awarded [ScoredPoint]
/// (who won each rally, in order), but until now that sequence was only ever
/// reduced to aggregate counts (points won, longest run). This widget plots the
/// *running lead* — the cumulative point differential after each rally — as a
/// filled area timeline, the headline "momentum" view of match-tracking apps:
/// the curve rides above the centre line while Player A is ahead and dips below
/// it while Player B is ahead, so a viewer can read runs, comebacks, and who
/// controlled which stretch of the match at a glance.
///
/// Rendering is split from the math: [momentumSeries] is a pure function turning
/// the point log into the cumulative-differential series (unit-testable without
/// pixels), and [_MomentumPainter] just draws it.
library;

import 'package:flutter/material.dart';

import '../../core/analysis/match_summary.dart';
import '../../core/scoring/scoring_engine.dart';

/// The cumulative point differential (Player A − Player B) after each rally.
///
/// The returned series is prefixed with a leading `0` (the 0–0 start before any
/// point) so it always has `points.length + 1` entries and the curve begins on
/// the centre line. A positive value means Player A leads by that many points; a
/// negative value means Player B leads. An empty point log yields `[0]`.
List<int> momentumSeries(List<ScoredPoint> points) {
  final series = <int>[0];
  var differential = 0;
  for (final point in points) {
    differential += point.winner == Player.a ? 1 : -1;
    series.add(differential);
  }
  return series;
}

/// A filled-area timeline of the running lead across a match.
///
/// Shows an empty axis (with a hint) until at least one point has been scored.
class MomentumChartView extends StatelessWidget {
  const MomentumChartView({super.key, required this.points});

  /// The ordered log of every awarded point (from [MatchSummary.points]).
  final List<ScoredPoint> points;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final series = momentumSeries(points);
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
                  painter: _MomentumPainter(
                    series: series,
                    leaderColor: theme.colorScheme.primary,
                    trailerColor: theme.colorScheme.tertiary,
                  ),
                ),
              ),
              if (points.isEmpty)
                Center(
                  child: Text(
                    'No points played yet',
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

class _MomentumPainter extends CustomPainter {
  _MomentumPainter({
    required this.series,
    required this.leaderColor,
    required this.trailerColor,
  });

  /// Cumulative A−B differential per point (leading 0 at index 0).
  final List<int> series;

  /// Colour used where Player A is ahead (differential > 0).
  final Color leaderColor;

  /// Colour used where Player B is ahead (differential < 0).
  final Color trailerColor;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    canvas.drawRect(rect, Paint()..color = const Color(0xFF15171C));

    final inset = rect.deflate(6);
    final midY = inset.center.dy;

    // Symmetric vertical scale around the centre line; guard the flat 0-lead
    // case so we never divide by zero.
    var peak = 1;
    for (final v in series) {
      if (v.abs() > peak) peak = v.abs();
    }

    // Centre (even) line.
    canvas.drawLine(
      Offset(inset.left, midY),
      Offset(inset.right, midY),
      Paint()
        ..color = Colors.white38
        ..strokeWidth = 1,
    );

    if (series.length < 2) return;

    Offset pointAt(int i) {
      final x = inset.left + i / (series.length - 1) * inset.width;
      final y = midY - series[i] / peak * (inset.height / 2);
      return Offset(x, y);
    }

    // Build the lead curve.
    final line = Path()..moveTo(pointAt(0).dx, pointAt(0).dy);
    for (var i = 1; i < series.length; i++) {
      final p = pointAt(i);
      line.lineTo(p.dx, p.dy);
    }

    // Fill the band between the curve and the centre line, tinted by lead.
    final finalLead = series.last;
    final fillColor = finalLead >= 0 ? leaderColor : trailerColor;
    final fill = Path.from(line)
      ..lineTo(inset.right, midY)
      ..lineTo(inset.left, midY)
      ..close();
    canvas.drawPath(fill, Paint()..color = fillColor.withValues(alpha: 0.25));

    // The lead line on top.
    canvas.drawPath(
      line,
      Paint()
        ..color = fillColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  @override
  bool shouldRepaint(_MomentumPainter old) =>
      old.series != series ||
      old.leaderColor != leaderColor ||
      old.trailerColor != trailerColor;
}
