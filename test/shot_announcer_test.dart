import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/training/shot_analyzer.dart';
import 'package:pong_ai/core/training/shot_announcer.dart';

/// Builds a [Shot] whose grade is controlled purely by [score] (the rest of the
/// fields are irrelevant to the announcer).
Shot _shot(double score) =>
    Shot(timestampMs: 0, speed: 1, depth: 0.75, score: score);

void main() {
  group('ShotAnnouncer grade calls', () {
    test('names each grade bucket', () {
      final a = ShotAnnouncer();
      expect(a.onShot(_shot(0.9)), 'Excellent shot!');
      a.reset();
      expect(a.onShot(_shot(0.65)), 'Good.');
      a.reset();
      expect(a.onShot(_shot(0.45)), 'Fair — a bit off.');
      a.reset();
      expect(a.onShot(_shot(0.1)), 'Off target.');
    });
  });

  group('ShotAnnouncer streaks', () {
    test('calls out a streak once enough on-target shots land in a row', () {
      final a = ShotAnnouncer();
      expect(a.onShot(_shot(0.9)), 'Excellent shot!'); // streak 1, no call-out
      expect(a.onShot(_shot(0.65)), 'Good.'); // streak 2, still below threshold
      // Third consecutive on-target shot crosses the default threshold.
      expect(a.onShot(_shot(0.85)), 'Excellent shot! 3 in a row!');
      expect(a.onShot(_shot(0.7)), 'Good. 4 in a row!');
      expect(a.streak, 4);
    });

    test('a below-target shot breaks the streak', () {
      final a = ShotAnnouncer();
      a.onShot(_shot(0.9));
      a.onShot(_shot(0.9));
      a.onShot(_shot(0.9)); // streak 3
      expect(a.streak, 3);
      // A fair shot resets the streak and just names the grade.
      expect(a.onShot(_shot(0.45)), 'Fair — a bit off.');
      expect(a.streak, 0);
      // The next on-target shot starts counting from one again.
      expect(a.onShot(_shot(0.9)), 'Excellent shot!');
      expect(a.streak, 1);
    });

    test('reset() forgets the running streak', () {
      final a = ShotAnnouncer();
      a.onShot(_shot(0.9));
      a.onShot(_shot(0.9));
      a.reset();
      expect(a.streak, 0);
      expect(a.onShot(_shot(0.9)), 'Excellent shot!');
    });

    test('a lower streak threshold calls out sooner', () {
      final a = ShotAnnouncer(streakThreshold: 2);
      expect(a.onShot(_shot(0.9)), 'Excellent shot!'); // streak 1
      expect(a.onShot(_shot(0.9)), 'Excellent shot! 2 in a row!');
    });
  });
}
