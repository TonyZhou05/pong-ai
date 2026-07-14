---
name: verify-scoring
description: Diagnose and fix a mis-scored rally in the footage pipeline — the empirical loop (events → frames → truth → rule fix → sweep) that every scoring correction in this repo followed.
---

# Diagnosing a mis-scored rally

Scoring is DEDUCED, not learned: YOLO detections → `BallTracker` events
(bounce/cross/loss) → `RallyReferee` (ITTF rules) → `ScoringEngine`. A wrong
score is either bad detections feeding the rules, or a rule mis-inferring.
Never patch a rule from theory alone — every fix in this repo came from this
loop:

1. **Reproduce**: `dart run tool/debug_footage_scoring.dart 30 999 0.004 15
   <fixture.json>` (decisions + final score) and
   `dart run tool/dump_tracker_events.dart <fixture.json>` (raw events).
2. **Establish truth from pixels**: extract frame strips (cv2, ~6 stills
   around the suspect moment) AND trace the labeled ball path
   (`ball_markup.json` — src time = seg time + cluster_start − 1.5s pad).
   Who stands where afterwards / who fetches the ball usually decides it.
3. **Find the divergence** between the event stream and reality, then fix at
   the right layer (tracker gate vs referee rule vs extraction), opt-in
   params so synthetic clips/tests keep historical behavior.
4. **Gate**: full-corpus sweep must keep every verified truth; add unit
   tests mirroring the failure; `flutter test` (700+ tests).

## Rules & failure modes already handled (don't re-break)

- `requireServe` gate: between-point passes arm nothing; serve = bounce-on-S
  → cross-from-S, or cross → landing bounce (tracks miss serve bounces).
- `postPointCooldown`: dying bounces after a point don't seed rallies.
- `netBounceExclusion 0.03`: y-reversal at the net plane = net clip, not a
  bounce. Under-net crossings (y below the whole table band) are suppressed.
- `extendedGapFrames`: lob out the frame top / <2 players visible → patient
  loss budget (off-frame play is not a rally end).
- `finishPlay()`: clip ends mid-rally → flush so the rally resolves.
- Exit evidence at loss: `lostOutside == last crossing's target` decides
  ONLY for off-frame returns (`originOffFrame`); a near-player recross
  conflicting with dead-drift (≥2 crossings) is PROVABLY undecidable
  (test_2_r2 vs test_6_r1: identical signatures, opposite truths) → prompt,
  never guess. Single unanswered crossing + exit → receiver wins.
- Sparse-track tuning (footage mode): maxGapFrames 30, default
  minBounceSpeed (0.008 missed real soft-apex bounces), NO maxJump (rejects
  real re-acquisitions after gaps).
- `PointReason`: doubleBounce / notReturned / outOfPlay(=prompt) /
  outOfBounds / manual. Winner = `playerOn(side)` — A=left, B=right.

## Undecidable endings

Some trajectories cannot determine the winner (ball out vs missed return).
Correct output is the undetermined prompt — a prompt is never a wrong score.
A learned referee needs ~200–500 human-labeled rallies (Label-rallies tab
exports them) before it can beat these rules statistically.
