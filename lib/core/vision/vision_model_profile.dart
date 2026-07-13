import 'package:ultralytics_yolo/ultralytics_yolo.dart' show YOLOTask;

import 'yolo_frame_adapter.dart';
import 'yolo_vision_service.dart';

/// A named, self-consistent on-device model choice.
///
/// Picking a vision model is not just picking a `.tflite` file: the model
/// determines the inference [task], the *class labels* it emits (a stock COCO
/// model calls the ball `sports ball`; a fine-tuned model may call it `ball`),
/// and the confidence bar those detections deserve. Before this seam the model
/// path lived in the camera screen while the label/threshold decode config
/// ([YoloFrameConfig]) lived separately in the adapter — and the camera screens
/// built a default [YoloVisionService], so there was no way to feed a custom
/// decode config to the live pipeline at all. Bundling them here makes swapping
/// to a fine-tuned ping-pong model a single coherent choice (see
/// `docs/ARCHITECTURE.md` §1 for the model-selection rationale and export path).
class VisionModelProfile {
  const VisionModelProfile({
    required this.id,
    required this.name,
    required this.modelPath,
    this.task = YOLOTask.detect,
    this.frameConfig = const YoloFrameConfig(),
    this.description = '',
  });

  /// Stable identifier (for persistence / selection UI).
  final String id;

  /// Human-readable model name.
  final String name;

  /// The model to load: a plugin-bundled name (e.g. `yolo11n`) or an asset path
  /// to a fine-tuned model (e.g. `assets/models/pingpong.tflite`).
  final String modelPath;

  /// The inference task. [YOLOTask.detect] yields both player and ball boxes;
  /// [YOLOTask.pose] adds player keypoints (footwork analytics) but the stock
  /// pose model drops the ball, so detection is the default for auto-scoring.
  final YOLOTask task;

  /// How this model's raw output is decoded into a [FrameResult] — the label
  /// sets and confidence thresholds tuned for this specific model.
  final YoloFrameConfig frameConfig;

  /// One-line note on when to use this profile.
  final String description;

  /// A [YoloVisionService] whose adapter decodes *this* model's output, so the
  /// profile's [frameConfig] is actually in force on the live pipeline.
  YoloVisionService createVisionService() =>
      YoloVisionService(adapter: YoloFrameAdapter(config: frameConfig));

  VisionModelProfile copyWith({
    String? id,
    String? name,
    String? modelPath,
    YOLOTask? task,
    YoloFrameConfig? frameConfig,
    String? description,
  }) {
    return VisionModelProfile(
      id: id ?? this.id,
      name: name ?? this.name,
      modelPath: modelPath ?? this.modelPath,
      task: task ?? this.task,
      frameConfig: frameConfig ?? this.frameConfig,
      description: description ?? this.description,
    );
  }
}

/// The stock COCO detector the [ultralytics_yolo] plugin ships. A single pass
/// labels both `person` (players) and `sports ball` (the ball), so it drives
/// the whole pipeline with zero setup — the out-of-the-box default.
const VisionModelProfile cocoDetectProfile = VisionModelProfile(
  id: 'coco-detect',
  name: 'COCO detector (yolo11n)',
  modelPath: 'yolo11n',
  description: 'Works out of the box, but the small, motion-blurred ball is '
      'only the generic COCO "sports ball" class — lower ball recall.',
);

/// Drop-in slot for a **fine-tuned** ping-pong detector bundled under
/// `assets/models/`. It is a multi-class (person + ball) detector so a single
/// [YOLOTask.detect] pass still yields both players and the ball, but the ball
/// class is trained on the small, blurred ping-pong ball for far higher recall.
/// The exported model is not shipped yet (see `docs/ARCHITECTURE.md` model
/// export path); this profile exists so pointing the app at it once bundled is
/// a one-line change rather than edits scattered across the screen and adapter.
const VisionModelProfile pingPongDetectProfile = VisionModelProfile(
  id: 'pingpong-detect',
  name: 'Ping-pong detector (fine-tuned)',
  modelPath: 'assets/models/pingpong.tflite',
  frameConfig: YoloFrameConfig(
    // A dedicated model earns a lower ball-confidence bar; recall on the tiny,
    // fast ball matters more than the odd false positive (the Kalman outlier
    // gate in BallTracker rejects the wild ones anyway).
    minBallConfidence: 0.15,
  ),
  description: 'Fine-tuned multi-class (person + ball) detector for higher '
      'recall on the small, motion-blurred ball. Bundle the exported .tflite / '
      '.mlpackage at modelPath to enable.',
);

/// The profile the app runs by default until a fine-tuned model is bundled.
const VisionModelProfile defaultVisionModel = cocoDetectProfile;

/// All selectable model profiles, in preference order.
const List<VisionModelProfile> visionModelProfiles = <VisionModelProfile>[
  cocoDetectProfile,
  pingPongDetectProfile,
];
