"""How many of the ALREADY EXTRACTED frames (01_frames.py) fall inside each annotated action segment (camera C10119)?
Tells whether option 5 (recognize past actions with the VLM) needs a new frame pass. Reads only file names.
usage: python 08_coverage.py /workspace/data/a101_anticip/data/CSVs /workspace/data/a101_vlm
"""
import bisect, collections, csv, os, sys

csvs, vlm = sys.argv[1], sys.argv[2]
segs = []
with open(os.path.join(csvs, "train.csv")) as f:
    for r in csv.DictReader(f):
        if r["video"].endswith("/C10119_rgb.mp4"):
            segs.append(("train", r["video"].split("/")[0], int(r["start_frame"]), int(r["end_frame"])))
with open(os.path.join(csvs, "validation_challenge.csv")) as f:
    for r in csv.DictReader(f):
        segs.append(("validation", r["video"].split("/")[0], int(r["start"]), int(r["end"])))

frames = {}
for seq in {s[1] for s in segs}:
    d = os.path.join(vlm, "frames", seq)
    frames[seq] = sorted(int(n[:-4]) for n in os.listdir(d) if n.endswith(".jpg")) if os.path.isdir(d) else []

cnt = collections.defaultdict(collections.Counter)
dur = []
for split, seq, st, en in segs:
    fr = frames[seq]
    k = bisect.bisect_right(fr, en) - bisect.bisect_left(fr, st)
    cnt[split][min(k, 4)] += 1
    dur.append((en - st) / 30)
dur.sort()
print(f"{len(segs)} segments | duration median {dur[len(dur) // 2]:.1f} s, 10% shorter than {dur[len(dur) // 10]:.1f} s, "
      f"90% shorter than {dur[9 * len(dur) // 10]:.1f} s")
for split, c in cnt.items():
    n = sum(c.values())
    print(f"{split:10s} frames already inside each action: " +
          " | ".join(f"{'4+' if k == 4 else k}: {100 * c[k] / n:4.1f}%" for k in range(5)))
