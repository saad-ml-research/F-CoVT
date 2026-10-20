"""Print file list, sizes, JSON keys, one sample, and how many video paths exist on disk."""
import json, os, sys
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1])
files = sorted(p for p in root.rglob("*") if p.is_file())
print(f"{len(files)} files")
for p in files[:30]:
    print(f"  {p.relative_to(root)}  {p.stat().st_size/1e6:.1f} MB")
for p in [f for f in files if f.suffix in (".json", ".jsonl")][:3]:
    data = [json.loads(l) for l in open(p)] if p.suffix == ".jsonl" else json.load(open(p))
    print(f"\n== {p.name}: {len(data)} samples")
    print("keys:", Counter(k for d in data for k in d).most_common())
    s = json.dumps(data[0], ensure_ascii=False)
    print("sample[0]:", s[:1500])
    vids = [d.get("video") or (d.get("videos") or [""])[0] for d in data]
    exists = sum(os.path.exists(v) for v in vids[:500] if v)
    print(f"video paths (first 500): {exists} exist on disk; example: {vids[0]}")
    print("reasoning_image index counts:", Counter(len(d.get("reasoning_image", [])) for d in data).most_common(5))
