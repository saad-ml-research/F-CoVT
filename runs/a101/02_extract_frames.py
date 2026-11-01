"""Stream Assembly101 videos: download one recording's C10119 video, cut out the frames the questions need, delete it.
Frames go to <frames_root>/<question id>/f1.jpg..f7.jpg (f1-f5 input, f6 near future, f7 step result), longest side 640 px.
Resumable: a recording whose questions all have their last frame is skipped. Disk use: one ~1 GB video at a time.
usage: python 02_extract_frames.py /workspace/data/a101/train_frames.json /workspace/data/a101/validation_frames.json
"""
import json, os, shutil, sys, time
import decord
from huggingface_hub import hf_hub_download
from PIL import Image

REPO, VIEW = "cvml-nus/assembly101", "C10119_rgb.mp4"
ROOT = os.environ.get("A101_FRAMES", "/workspace/data/a101/frames")
TMP = os.environ.get("A101_TMP", "/workspace/data/a101/_tmp_video")

todo = {}
for p in sys.argv[1:]:
    for seq, qs in json.load(open(p)).items():
        todo.setdefault(seq, {}).update(qs)
seqs = sorted(todo)
print(f"{len(seqs)} recordings, {sum(len(v) for v in todo.values())} questions", flush=True)
t0 = time.time()
for i, seq in enumerate(seqs, 1):
    qs = todo[seq]
    if all(os.path.exists(os.path.join(ROOT, q, f"f{len(fr)}.jpg")) for q, fr in qs.items()):
        continue
    os.makedirs(TMP, exist_ok=True)
    path = hf_hub_download(REPO, f"recordings/{seq}/{VIEW}", repo_type="dataset", local_dir=TMP)
    vr = decord.VideoReader(path, num_threads=4)
    fps, n = vr.get_avg_fps(), len(vr)
    want = sorted({min(n - 1, int(round(f * fps / 30))) for fr in qs.values() for f in fr})
    imgs = {}
    for c in range(0, len(want), 48):  # decode in chunks: 1080p frames are ~6 MB each in memory
        for idx, arr in zip(want[c:c + 48], vr.get_batch(want[c:c + 48]).asnumpy()):
            im = Image.fromarray(arr); im.thumbnail((640, 640)); imgs[idx] = im
    for q, fr in qs.items():
        d = os.path.join(ROOT, q); os.makedirs(d, exist_ok=True)
        for k, f in enumerate(fr, 1):
            imgs[min(n - 1, int(round(f * fps / 30)))].save(os.path.join(d, f"f{k}.jpg"), quality=90)
    del vr
    shutil.rmtree(TMP, ignore_errors=True)
    el = time.time() - t0
    print(f"[{i}/{len(seqs)}] {seq}: {len(qs)} questions, {len(want)} frames @ {fps:.0f} fps | "
          f"{el/60:.1f} min elapsed, ~{el/i*(len(seqs)-i)/60:.0f} min left", flush=True)
print("done")
