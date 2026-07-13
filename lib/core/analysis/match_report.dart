/// Unified, exportable post-match report.
///
/// Every prior analytics layer produces its own slice of the story — the
/// [MatchSummary] scoring breakdown, the [RallyStats] rally-length distribution,
/// per-player [PlayerMovementStats] footwork, and per-side [SidePlacementStats]
/// bounce placement — but until now each was only ever rendered piecemeal into
/// its own widget on the summary panel. There was no single, shareable text
/// artifact that answers "how did this match go?" in one place, which the
/// objective's "produce summary" goal asks for.
///
/// [buildMatchReport] folds all of those live analytics off a [MatchController]
/// into one deterministic, human-readable string that can be copied to the
/// clipboard or shared. Like the rest of `core/`, it has no Flutter or vision
/// dependencies, so the composed report is unit-testable end-to-end from a
/// synthetic frame stream.
library;

import '../scoring/scoring_engine.dart';
import 'ball_tracker.dart';
import 'bounce_placement.dart';
import 'match_controller.dart';
import 'player_movement.dart';
import 'rally_analyzer.dart';

String _playerName(Player p) => p == Player.a ? 'Player A' : 'Player B';

String _sideName(TableSide s) => s == TableSide.left ? 'Left side' : 'Right side';

/// A per-player footwork section, or a "not tracked" note when the pose model
/// never located this player (e.g. a camera-free replay clip with no people).
String _movementSection(Player player, PlayerMovementStats m) {
  final lines = <String>['${_playerName(player)} movement'];
  if (!m.wasTracked) {
    lines.add('  • not tracked');
    return lines.join('\n');
  }
  lines
    ..add('  • distance travelled: ${m.distanceTravelled.toStringAsFixed(2)}')
    ..add('  • court coverage: '
        '${m.coverageWidth.toStringAsFixed(2)} wide × '
        '${m.coverageDepth.toStringAsFixed(2)} deep')
    ..add('  • mobility: ${m.mobilityPerSecond.toStringAsFixed(2)} / s');
  final stance = m.averageStanceWidth;
  if (stance != null) {
    lines.add('  • avg stance width: ${stance.toStringAsFixed(2)}');
  }
  return lines.join('\n');
}

/// A per-side bounce-placement section, or a "no bounces" note when nothing
/// landed on that half of the table.
String _placementSection(TableSide side, SidePlacementStats p) {
  final lines = <String>['${_sideName(side)} placement'];
  if (p.count == 0) {
    lines.add('  • no bounces recorded');
    return lines.join('\n');
  }
  lines
    ..add('  • ${p.count} bounces — '
        'avg depth ${p.averageDepth.toStringAsFixed(2)} '
        '(0 net … 1 baseline)')
    ..add('  • ${p.shortCount} short / ${p.middleCount} mid / ${p.deepCount} deep')
    ..add('  • lateral spread: ${p.lateralSpread.toStringAsFixed(2)}');
  return lines.join('\n');
}

/// A ball-speed section, or a "not estimated" note when no along-table motion
/// was measured (e.g. a camera-free clip with no ball detections).
String _ballSpeedSection(MatchController controller) {
  final lines = <String>['Ball speed'];
  if (!controller.hasBallSpeedData) {
    lines.add('  • not estimated');
    return lines.join('\n');
  }
  lines
    ..add('  • fastest: ${controller.maxBallSpeedKmh.toStringAsFixed(1)} km/h')
    ..add('  • average: ${controller.averageBallSpeedKmh.toStringAsFixed(1)} km/h');
  return lines.join('\n');
}

/// Compose the full, shareable match report from a [controller]'s live
/// analytics. Deterministic and Flutter-free.
String buildMatchReport(MatchController controller) {
  final sections = <String>[
    controller.summary.report(),
    controller.rallyStats.report(),
    _ballSpeedSection(controller),
  ];

  for (final player in Player.values) {
    sections.add(_movementSection(player, controller.movementFor(player)));
  }

  for (final side in TableSide.values) {
    sections.add(_placementSection(side, controller.placementFor(side)));
  }

  // Blank line between sections for readability.
  return sections.join('\n\n');
}
