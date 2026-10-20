"""Get FutureBench without downloading all of V1-33K (494 GB).
1) download futurebench.json (2.7 MB), print its format, collect the video names it needs
2) stream each first_segment_video tar part from the Hub (no local copy), extract only needed videos,
   stop when all are found
3) save an HF dataset ({"test": ...}) with absolute video paths for the lmms-eval futurebench task
usage: python 08_futurebench_fetch.py [--inspect]"""
import argparse, json, os, tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download, HfFileSystem

ap = argparse.ArgumentParser(); ap.add_argument("--inspect", action="store_true")
ap.add_argument("--folder", default="first_segment_video")
a = ap.parse_args()
R = "haonan3/V1-33K"; OUT = Path("/workspace/data/futurebench"); VID = OUT / "videos"
VID.mkdir(parents=True, exist_ok=True)
fb = json.load(open(hf_hub_download(R, "futurebench.json", repo_type="dataset", local_dir=OUT)))
items = fb if isinstance(fb, list) else list(fb.values()) if isinstance(fb, dict) else fb
print(f"futurebench.json: {len(items)} items; keys: {list(items[0].keys())}")
print("item[0]:", json.dumps(items[0], ensure_ascii=False)[:1500])
paths = [it.get("video_path") or it.get("video") for it in items]
def key(p):  # match on "<source>/<file>.mp4" (json says first_part_video/, the Hub folder is first_segment_video/)
    parts = str(p).replace("\\", "/").split("/")
    return "/".join(parts[-2:])
need = {key(p) for p in paths if p}
print(f"{len(need)} unique videos needed; example raw path: {paths[0]}")
if a.inspect:
    raise SystemExit

have = {f"{d.name}/{f.name}" for d in VID.iterdir() if d.is_dir() for f in d.iterdir()}
missing = need - have
fs = HfFileSystem()
parts = sorted(fs.glob(f"datasets/{R}/{a.folder}/*.tar.gz"),
               key=lambda s: int(s.rsplit("part", 1)[1].split(".")[0]))
for part in parts:
    if not missing:
        break
    got = 0; shown = 0
    with fs.open(part, "rb", block_size=64 * 2**20) as fh, tarfile.open(fileobj=fh, mode="r|gz") as tar:
        for m in tar:
            if shown < 3 and m.isfile():
                print("  tar member example:", m.name); shown += 1
            b = key(m.name)
            if m.isfile() and b in missing:
                (VID / b).parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(VID / b, "wb") as dst:
                    dst.write(src.read())
                missing.discard(b); got += 1
    print(f"{part.rsplit('/',1)[1]}: +{got}, still missing {len(missing)}", flush=True)

from datasets import Dataset, DatasetDict
rows = []
for it in items:
    b = key(it.get("video_path") or it.get("video") or "")
    if (VID / b).exists():
        r = dict(it); r["video_path"] = str(VID / b); rows.append(r)
print(f"{len(rows)}/{len(items)} questions have their video")
DatasetDict({"test": Dataset.from_list(rows)}).save_to_disk(str(OUT / "hf"))
print(f"FB_DATA={OUT/'hf'}  FB_VIDEOS={VID}")
