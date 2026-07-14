/// Real-footage demo wiring for the Match screen: which bundled clip to play,
/// how to load its recorded detection track, and the seam over the video
/// plugin so the screen stays widget-testable without a platform channel.
///
/// The footage is a real side-recorded table-tennis rally clip (OpenTTGames,
/// lab.osai.ai) bundled under `assets/footage/`, and the detection track is a
/// [ClipFixture] — the exact benchmark-harness format — whose frames carry the
/// per-frame player boxes/keypoints (YOLO11n-pose) and ball positions (the
/// dataset's labeled ball fused with a YOLO11n detector), so the Match page
/// can overlay the app's identification of both players and the ball on the
/// actual video.
library;

import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/widgets.dart';
import 'package:video_player/video_player.dart';

import '../../core/benchmark/clip_fixture.dart';

/// A bundled footage demo: the video asset plus its detection-track fixture.
class FootageDemo {
  const FootageDemo({required this.videoAsset, required this.fixtureAsset});

  /// The playable video (H.264 mp4, bundled asset).
  final String videoAsset;

  /// The [ClipFixture] JSON asset whose `frames` carry the recorded per-frame
  /// detections, timestamped in milliseconds into [videoAsset].
  final String fixtureAsset;
}

/// The default bundled clip: OpenTTGames `test_2` (30 s, two rally clusters).
const FootageDemo defaultFootageDemo = FootageDemo(
  videoAsset: 'assets/footage/openttgames_test2.mp4',
  fixtureAsset: 'assets/footage/openttgames_test2.json',
);

/// Loads a footage fixture from the asset bundle. Tests bypass this by
/// injecting an in-memory loader instead (asset I/O hangs `testWidgets`).
Future<ClipFixture> loadFootageFixture(String asset) async {
  final raw = await rootBundle.loadString(asset);
  return ClipFixture.fromJson(jsonDecode(raw) as Map<String, dynamic>);
}

/// One entry of the real-footage match corpus: a titled rally clip the
/// Matches screen lists, opening the footage [FootageDemo] on tap.
class FootageMatch {
  const FootageMatch({
    required this.id,
    required this.title,
    required this.demo,
    this.durationMs = 0,
    this.bounces = 0,
    this.source = '',
    this.pointsA = 0,
    this.pointsB = 0,
    this.pointWinners,
  });

  final String id;
  final String title;
  final FootageDemo demo;

  /// Clip length in milliseconds (for the list subtitle).
  final int durationMs;

  /// Ground-truth labeled table bounces in the clip.
  final int bounces;

  /// Dataset the clip came from.
  final String source;

  /// The fixture's current ground-truth outcome (0-0 with no winners means
  /// unlabeled/provisional). Shown by the labeling screen as the existing
  /// label to confirm or correct.
  final int pointsA;
  final int pointsB;
  final List<String>? pointWinners;

  /// Human summary of the current label, or null when unlabeled.
  String? get truthSummary {
    if (pointsA == 0 && pointsB == 0) return null;
    return '$pointsA–$pointsB'
        '${pointWinners == null ? '' : ' (${pointWinners!.join(', ').toUpperCase()})'}';
  }

  factory FootageMatch.fromJson(Map<String, dynamic> json) => FootageMatch(
        id: json['id'] as String,
        title: json['title'] as String,
        demo: FootageDemo(
          videoAsset: json['video'] as String,
          fixtureAsset: json['fixture'] as String,
        ),
        durationMs: (json['durationMs'] as num?)?.toInt() ?? 0,
        bounces: (json['bounces'] as num?)?.toInt() ?? 0,
        source: json['source'] as String? ?? '',
        pointsA: (json['pointsA'] as num?)?.toInt() ?? 0,
        pointsB: (json['pointsB'] as num?)?.toInt() ?? 0,
        pointWinners: (json['pointWinners'] as List<dynamic>?)
            ?.map((w) => w as String)
            .toList(growable: false),
      );
}

/// Loads the bundled match-corpus manifest (`assets/footage/manifest.json`).
/// Tests inject an in-memory list instead.
Future<List<FootageMatch>> loadFootageManifest() async {
  final raw = await rootBundle.loadString('assets/footage/manifest.json');
  return (jsonDecode(raw) as List<dynamic>)
      .map((e) => FootageMatch.fromJson(e as Map<String, dynamic>))
      .toList(growable: false);
}

/// Minimal playback surface the Match screen needs from a video player —
/// the injectable seam over the `video_player` plugin (the same pattern as
/// `YOLOView`'s `cameraPreviewBuilder`): tests supply a fake whose position
/// the test drives, so no platform channel is ever touched headlessly.
abstract class FootagePlayer {
  /// Prepares the player; [view] and [position] are valid afterwards.
  Future<void> initialize();

  /// Current playback position (the clock the detection replay syncs to).
  Duration get position;

  /// Whether the footage is currently advancing.
  bool get isPlaying;

  /// Width/height of the footage, for layout.
  double get aspectRatio;

  Future<void> play();
  Future<void> pause();

  /// Rewinds to the start (used by the Replay action).
  Future<void> seekToStart();

  /// The video widget itself.
  Widget get view;

  Future<void> dispose();
}

/// The real `video_player`-backed implementation.
class VideoFootagePlayer implements FootagePlayer {
  VideoFootagePlayer(String asset)
      : _controller = VideoPlayerController.asset(asset);

  final VideoPlayerController _controller;

  @override
  Future<void> initialize() async {
    await _controller.initialize();
    // The bundled clip has no audio track; muting also keeps web autoplay
    // policies from blocking playback.
    await _controller.setVolume(0);
  }

  @override
  Duration get position => _controller.value.position;

  @override
  bool get isPlaying => _controller.value.isPlaying;

  @override
  double get aspectRatio {
    final ratio = _controller.value.aspectRatio;
    return ratio > 0 ? ratio : 16 / 9;
  }

  @override
  Future<void> play() => _controller.play();

  @override
  Future<void> pause() => _controller.pause();

  @override
  Future<void> seekToStart() => _controller.seekTo(Duration.zero);

  @override
  Widget get view => VideoPlayer(_controller);

  @override
  Future<void> dispose() => _controller.dispose();
}
