import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/vision/vision_model_profile.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart' show YOLOTask;

void main() {
  group('VisionModelProfile', () {
    test('default profile is the stock COCO detector', () {
      expect(defaultVisionModel, same(cocoDetectProfile));
      expect(cocoDetectProfile.modelPath, 'yolo11n');
      expect(cocoDetectProfile.task, YOLOTask.detect);
      // The stock model still recognizes both the ball and the players.
      expect(cocoDetectProfile.frameConfig.ballLabels, contains('sports ball'));
      expect(cocoDetectProfile.frameConfig.personLabels, contains('person'));
    });

    test('registry exposes both profiles by preference order', () {
      expect(visionModelProfiles.first, same(cocoDetectProfile));
      expect(visionModelProfiles, contains(pingPongDetectProfile));
      final ids = visionModelProfiles.map((p) => p.id).toSet();
      expect(ids.length, visionModelProfiles.length, reason: 'ids are unique');
    });

    test('fine-tuned profile lowers the ball-confidence bar but keeps players',
        () {
      // A dedicated ball model is trusted at a lower confidence than COCO's
      // generic "sports ball", while still detecting people (multi-class).
      expect(
        pingPongDetectProfile.frameConfig.minBallConfidence,
        lessThan(cocoDetectProfile.frameConfig.minBallConfidence),
      );
      expect(
        pingPongDetectProfile.frameConfig.personLabels,
        contains('person'),
      );
      expect(pingPongDetectProfile.modelPath, endsWith('.tflite'));
    });

    test('createVisionService wires the profile frame-config into the adapter',
        () {
      final service = pingPongDetectProfile.createVisionService();
      addTearDown(service.dispose);
      expect(
        service.adapter.config,
        same(pingPongDetectProfile.frameConfig),
      );
      expect(
        service.adapter.config.minBallConfidence,
        pingPongDetectProfile.frameConfig.minBallConfidence,
      );
    });

    test('copyWith overrides only the named fields', () {
      final custom = cocoDetectProfile.copyWith(
        modelPath: 'assets/models/custom.tflite',
        task: YOLOTask.pose,
      );
      expect(custom.modelPath, 'assets/models/custom.tflite');
      expect(custom.task, YOLOTask.pose);
      // Unspecified fields are inherited.
      expect(custom.id, cocoDetectProfile.id);
      expect(custom.frameConfig, same(cocoDetectProfile.frameConfig));
    });
  });
}
