---
name: footage-corpus
description: Regenerate, extend, or validate the real-footage rally corpus (assets/footage) — extraction, labeling, combining, manifest, and the verification sweep that gates every change.
---

# Real-footage rally corpus

The Matches/Match/Label-rallies tabs run on `assets/footage/`: per-rally clips
(`<video>_rN.mp4/.json`), combined sets (`<video>_full.*`), and
`manifest.json`. Fixtures are `ClipFixture` JSON (the benchmark format):
frames = recorded YOLO detections; `groundTruth` = verified or provisional
outcome; netX/table bounds derived from the dataset labels.

## Pipeline (order matters)

1. **Source data**: OpenTTGames videos + markup (`ball_markup.json`,
   `events_markup.json`) in a working dir `openttgames/` (1080p@120fps;
   test_1..7 downloadable from lab.osai.ai — ~0.2–2.3 GB each, ask before
   downloading). Markup `net` events ≈ ball at the net; labels cover only
   SOME rallies and may stop mid-rally; junk coords (−0.001) exist — filter
   to (0.02, 0.98).
2. **Extract**: `python3 tool/extract_footage_corpus.py` (run from the dir
   holding `openttgames/`). It clusters labels into rallies (>1.5s gaps),
   drops handovers (rally ⇔ ≥2 net crossings OR both-side bounces over ≥1s),
   extends segment ends by detector probe (HARD-CAPPED before the next raw
   cluster − 2s, else held-ball detections merge rallies), runs
   YOLO11n-pose + ball detector per frame, derives netX (median ball x at
   net events) and table bounds (bounce positions ± 0.05), and bakes
   verified truths from `KNOWN_TRUTH`.
3. **Label**: `python3 provisional_label.py <fixture.json ...>` (scratch
   script; shells the Dart scorer) writes pipeline outcomes as PROVISIONAL
   groundTruth. NEVER overwrite `KNOWN_TRUTH` (user/frame-verified) entries.
   Human labels also arrive via the app's "Label rallies" tab → exported
   JSON (`RallyLabel` list) → merge into fixtures + KNOWN_TRUTH.
4. **Combine**: `python3 tool/combine_footage.py` → `<video>_full.*` with
   freeze-frame gaps between rallies (gap frames MUST carry people and
   out-wait the tracker's most patient loss budget, else boundaries merge).
5. **Manifest**: `python3 tool/rebuild_manifest.py` (carries each fixture's
   groundTruth so the labeling tab shows current calls).
6. **Sync**: copy `out_corpus/*` → `assets/footage/`.

## The gate: verification sweep

Every rules/threshold/detector change MUST pass the verified suite before
shipping. For each fixture:

    dart run tool/debug_footage_scoring.dart 30 999 0.004 15 <fixture.json>

(args: maxGapFrames, maxJump(999=off), minBounceSpeed, postPointCooldown —
mirrors `_footageController` in `lib/features/match/match_screen.dart`).
Verified truths live in fixtures' `groundTruth` + extractor `KNOWN_TRUTH`.
A regression on any verified clip blocks shipping (this caught the
fine-tuned-detector regressions). `tool/dump_tracker_events.dart <fixture>`
prints the raw event stream for diagnosis.

## Known state (2026-07-14)

Shipping corpus = stock-detector extraction; the fine-tuned-detector
re-extraction (better coverage, up to 89%) regresses 4/11 verified truths —
rules need a dense-track retune (eager double-cross, tighter probe gates)
before adoption. Parked fixtures: scratchpad `out_corpus_finetuned/`.
