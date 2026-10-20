"""Random subset of a Future-L1/TwiFF JSON, keeping only samples whose video exists."""
import json, os, random, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(src))
data = [d for d in data if os.path.exists(d.get("video") or (d.get("videos") or [""])[0])]
random.seed(0); random.shuffle(data)
json.dump(data[:n], open(dst, "w"))
print(f"kept {min(n, len(data))} of {len(data)} samples with videos on disk -> {dst}")
