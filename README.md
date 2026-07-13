# pong-ai

An AI table-tennis referee and coach. Place your phone on the side of the table
and pong-ai tracks the players and the ball from the camera, keeps score
automatically, and produces a post-match performance summary. A **training mode**
lets you practise against a return net and grades the quality of your shots.

Built with Flutter, running on-device object detection + pose estimation via the
official [`ultralytics_yolo`](https://pub.dev/packages/ultralytics_yolo) plugin
(YOLO nano pose for players, a fine-tuned detector + Kalman tracker for the ball).

## Status

Early scaffold. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the model
selection rationale, benchmarking plan, and roadmap.

Working so far:

- Flutter app scaffold (Android + iOS) with a home screen (Match / Training).
- Pure-Dart, fully unit-tested **ITTF scoring engine** (`lib/core/scoring/`):
  11-point games, 2-point/deuce logic, serve rotation, best-of-N matches, undo.
- Runtime-agnostic vision data models and a `VisionService` interface
  (`lib/core/vision/`) so models can be swapped or replayed from benchmarks.

## Getting started

```bash
flutter pub get
flutter test        # runs scoring-engine + widget tests
flutter run         # on a connected device / simulator
```

## Layout

```
lib/core/scoring/   ITTF rules engine (no Flutter deps → unit-testable)
lib/core/vision/    Detection/pose data models + VisionService interface
lib/features/       UI: home, match, training, summary
docs/ARCHITECTURE.md  Model selection, benchmarking plan, roadmap
```
