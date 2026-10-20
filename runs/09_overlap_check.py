"""Check video overlap between Future-L1-50K (training) and FutureBench (test) by YouTube ID.
Runs in seconds on the two JSON files already on disk. Writes overlap ids + affected training ids."""
import json, re
from collections import Counter
from pathlib import Path

YT = re.compile(r"[A-Za-z0-9_-]{11}")
train = json.load(open("/workspace/data/Future-L1-50K/Future-L1-50K.json"))
fb = json.load(open("/workspace/data/futurebench/futurebench.json"))

def train_id(v):                      # "-0AAhXT99Zw-0:04:09.165-0:04:57.213.mp4" -> "-0AAhXT99Zw"
    return v[:11]

def fb_ids(path):                     # every 11-char YouTube-like token in the file name
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    stem = re.sub(r"^v1-3-train_val-", "", stem)
    stem = re.sub(r"^v_", "", stem)
    cands = {stem[:11]} if len(stem) >= 11 else set()
    m = re.match(r"^\d+-(.{11})", stem)  # youcook2 "404-Xornhv6aDF4-split_4"
    if m: cands.add(m.group(1))
    return {c for c in cands if YT.fullmatch(c)}

train_by_id = {}
for d in train:
    train_by_id.setdefault(train_id(d["video"]), []).append(d["id"])
fb_src = Counter(it["data_source"] for it in fb)
print("FutureBench sources:", dict(fb_src))

hits, hit_q, hit_src = set(), 0, Counter()
for it in fb:
    ids = fb_ids(it["video_path"]) & train_by_id.keys()
    if ids:
        hits |= ids; hit_q += 1; hit_src[it["data_source"]] += 1
affected = sorted(t for i in hits for t in train_by_id[i])
print(f"training samples: {len(train)} from {len(train_by_id)} YouTube videos")
print(f"overlapping YouTube videos: {len(hits)}")
print(f"FutureBench questions on an overlapping video: {hit_q}/{len(fb)} by source {dict(hit_src)}")
print(f"training samples from those videos: {len(affected)}")
Path("/workspace/data/overlap_train_ids.txt").write_text("\n".join(affected))
print("examples:", sorted(hits)[:10])
