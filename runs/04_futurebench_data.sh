#!/bin/bash
# Find FutureBench: list every NON-video file in V1-33K (+ its README), and folder totals.
# Nothing big is downloaded.
set -e
source /workspace/envs/fl1/bin/activate
export HF_HOME=/workspace/hf
mkdir -p /workspace/F-CoVT/logs
python - <<'PY' 2>&1 | tee /workspace/F-CoVT/logs/04_futurebench_list.log
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
rid = "haonan3/V1-33K"
info = api.dataset_info(rid, files_metadata=True)
folders = defaultdict(lambda: [0, 0.0])
for s in info.siblings:
    top = s.rfilename.split("/")[0] if "/" in s.rfilename else "(root)"
    folders[top][0] += 1; folders[top][1] += (s.size or 0) / 1e9
print(f"== {rid} folders:")
for k, (n, gb) in sorted(folders.items()):
    print(f"  {k:40s} {n:4d} files {gb:8.1f} GB")
print("\n== non-archive files:")
for s in info.siblings:
    if not s.rfilename.endswith((".tar.gz", ".tar", ".zip")):
        print(f"  {(s.size or 0)/1e6:9.1f} MB  {s.rfilename}")
print("\n== README.md")
print(open(hf_hub_download(rid, "README.md", repo_type="dataset")).read()[:4000])
PY
