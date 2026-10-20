"""Inspect Future-L1-50K: README, JSON fields + one sample, and one parquet shard's schema."""
import json, os, sys
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1])
readme = root / "README.md"
if readme.exists():
    print("== README.md\n" + readme.read_text()[:3000])

for p in sorted(root.glob("*.json")):
    data = json.load(open(p))
    print(f"\n== {p.name}: {len(data)} samples")
    print("keys:", Counter(k for d in data for k in d).most_common())
    print("sample[0]:", json.dumps(data[0], ensure_ascii=False)[:2000])
    vids = [d.get("video") or (d.get("videos") or [""])[0] for d in data]
    print("example video paths:", vids[:3])
    print("video path prefixes:", Counter(str(v).split("/")[0] for v in vids).most_common(10))
    print("reasoning_image counts:", Counter(len(d.get("reasoning_image", [])) for d in data).most_common(5))
    print("image counts:", Counter(len(d.get("image", [])) for d in data).most_common(5))

import pyarrow.parquet as pq
for p in sorted(root.glob("data/*.parquet"))[:1]:
    f = pq.ParquetFile(p)
    print(f"\n== {p.name}: {f.metadata.num_rows} rows")
    print(f.schema_arrow)
    row = f.read_row_group(0).slice(0, 1).to_pylist()[0]
    for k, v in row.items():
        if isinstance(v, (bytes, bytearray)):
            print(f"  {k}: <bytes {len(v)}>")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys {list(v)[:10]}")
        elif isinstance(v, list):
            first = v[0] if v else None
            desc = f"<bytes {len(first)}>" if isinstance(first, (bytes, bytearray)) else (
                f"dict keys {list(first)[:6]}" if isinstance(first, dict) else str(first)[:200])
            print(f"  {k}: list[{len(v)}] first={desc}")
        else:
            print(f"  {k}: {str(v)[:300]}")
