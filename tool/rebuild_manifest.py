"""Rebuild out_corpus/manifest.json from every fixture JSON on disk, so
multi-pass extraction runs (which each overwrite the manifest with only their
own videos) always end with a complete, sorted corpus manifest.

Handles both per-rally segments (<video>_rN) and combined multi-rally clips
(<video>_full, from combine_footage.py) — combined clips list first so the
cumulative-score demos lead the Matches screen."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "out_corpus"
full_entries, rally_entries = [], []
for p in sorted(OUT.glob("*.json")):
    if p.name == "manifest.json":
        continue
    d = json.loads(p.read_text())
    seg_id = d["name"]
    entry = {
        "id": seg_id,
        "video": f"assets/footage/{seg_id}.mp4",
        "fixture": f"assets/footage/{seg_id}.json",
        "durationMs": d["frames"][-1]["t"] if d["frames"] else 0,
        "bounces": len(d.get("groundTruthEvents") or []),
        "source": "OpenTTGames",
    }
    if seg_id.endswith("_full"):
        video = seg_id[: -len("_full")]
        rallies = len(list(OUT.glob(f"{video}_r*.json")))
        entry["title"] = (
            f"{video.replace('_', ' ').title()} — Full set ({rallies} rallies)"
        )
        full_entries.append(entry)
    else:
        video, rally = seg_id.rsplit("_r", 1)
        entry["title"] = f"{video.replace('_', ' ').title()} — Rally {rally}"
        rally_entries.append(entry)

entries = full_entries + rally_entries
(OUT / "manifest.json").write_text(json.dumps(entries, indent=1))
print(f"{len(entries)} entries:")
for e in entries:
    print(f"  {e['id']}: {e['durationMs']/1000:.1f}s, {e['bounces']} bounces")
