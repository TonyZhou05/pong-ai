"""Rebuild out_corpus/manifest.json from every fixture JSON on disk, so
multi-pass extraction runs (which each overwrite the manifest with only their
own videos) always end with a complete, sorted corpus manifest."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "out_corpus"
entries = []
for p in sorted(OUT.glob("*.json")):
    if p.name == "manifest.json":
        continue
    d = json.loads(p.read_text())
    seg_id = d["name"]
    video_name, rally = seg_id.rsplit("_r", 1)
    entries.append(
        {
            "id": seg_id,
            "title": f"{video_name.replace('_', ' ').title()} — Rally {rally}",
            "video": f"assets/footage/{seg_id}.mp4",
            "fixture": f"assets/footage/{seg_id}.json",
            "durationMs": d["frames"][-1]["t"] if d["frames"] else 0,
            "bounces": len(d.get("groundTruthEvents") or []),
            "source": "OpenTTGames",
        }
    )
(OUT / "manifest.json").write_text(json.dumps(entries, indent=1))
print(f"{len(entries)} entries:")
for e in entries:
    print(f"  {e['id']}: {e['durationMs']/1000:.1f}s, {e['bounces']} bounces")
