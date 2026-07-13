import 'package:flutter_test/flutter_test.dart';
import 'package:pong_ai/core/training/shot_analyzer.dart';
import 'package:pong_ai/core/training/shot_announcer.dart';

/// Builds a [Shot] whose grade is controlled purely by [score] (the rest of the
/// fields are irrelevant to the announcer).
Shot _shot(double score) =>
    Shot(timestampMs: 0, speed: 1, depth: 0.75, score: score);

/// Builds an on-target [Shot] carrying a physical [speedKmh] pace.
Shot _fastShot(double speedKmh) =>
    Shot(timestampMs: 0, speed: 1, depth: 0.75, score: 0.9, speedKmh: speedKmh);

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

  group('ShotAnnouncer top-speed milestone', () {
    test('calls out only when a shot beats the session best', () {
      final a = ShotAnnouncer();
      // First speed-bearing shot silently sets the baseline (no milestone).
      expect(a.onShot(_fastShot(30)), 'Excellent shot!'); // streak 1
      expect(a.topSpeedKmh, 30);
      // A slower shot does not fire the milestone.
      expect(a.onShot(_fastShot(25)), 'Excellent shot!'); // streak 2
      expect(a.topSpeedKmh, 30);
      // A faster shot fires it (rounded), before the streak call-out.
      expect(
        a.onShot(_fastShot(42.4)),
        'Excellent shot! New top speed, 42 km/h! 3 in a row!',
      );
      expect(a.topSpeedKmh, 42.4);
    });

    test('stays silent when shots carry no physical speed', () {
      final a = ShotAnnouncer();
      // speedKmh defaults to 0 (no ruler yet), so no milestone ever fires.
      expect(a.onShot(_shot(0.9)), 'Excellent shot!');
      expect(a.onShot(_shot(0.9)), 'Excellent shot!');
      expect(a.topSpeedKmh, 0);
    });

    test('reset() forgets the session-best pace', () {
      final a = ShotAnnouncer();
      a.onShot(_fastShot(30));
      a.onShot(_fastShot(40)); // milestone
      expect(a.topSpeedKmh, 40);
      a.reset();
      expect(a.topSpeedKmh, 0);
      // After reset the first speed shot is a baseline again, no milestone.
      expect(a.onShot(_fastShot(35)), 'Excellent shot!');
    });
  });

  group('spokenSessionSummary', () {
    test('notes an empty session', () {
      expect(
        spokenSessionSummary(const TrainingSummary([])),
        'Session complete. No shots recorded.',
      );
    });

    test('reads back shot count and grade, pluralising', () {
      // Three A-grade shots (score 0.9 -> grade A), no physical speed scale.
      final summary = TrainingSummary(List.filled(3, _shot(0.9)));
      expect(
        spokenSessionSummary(summary),
        'Session complete. 3 shots, grade A.',
      );
    });

    test('uses the singular for a one-shot session', () {
      final summary = TrainingSummary([_shot(0.75)]); // score 0.75 -> grade B
      expect(
        spokenSessionSummary(summary),
        'Session complete. 1 shot, grade B.',
      );
    });

    test('appends the top pace when a physical scale is available', () {
      final summary = TrainingSummary([_fastShot(41.6), _fastShot(30)]);
      // maxSpeedKmh 41.6 rounds to 42; both shots score 0.9 -> grade A.
      expect(
        spokenSessionSummary(summary),
        'Session complete. 2 shots, grade A. '
        'Top speed 42 kilometres per hour.',
      );
    });
  });
}
