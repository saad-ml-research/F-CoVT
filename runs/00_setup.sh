#!/bin/bash
# One-time setup on the pod. Two separate envs: CoVT and Future-L1 pin different torch/transformers.
set -e
W=/workspace
cd $W
[ -d CoVT ] || git clone https://github.com/Wakals/CoVT.git
[ -d Future-L1 ] || git clone https://github.com/OpenGVLab/Future-L1.git
pip install -q uv huggingface_hub[cli]

# --- env 1: covt (torch 2.5.1, transformers 4.50.1) ---
if [ ! -d $W/envs/covt ]; then
  uv venv $W/envs/covt --python 3.10
  source $W/envs/covt/bin/activate
  (cd CoVT/VLMEvalKit && uv pip install -e . && uv pip install datasets scikit-learn)
  (cd CoVT/train && uv pip install -r requirements.txt \
     && uv pip install git+https://github.com/facebookresearch/segment-anything.git \
     && uv pip install -r src/anchors/DepthAnything/requirements.txt \
     && uv pip install xformers==0.0.29.post1)
  uv pip install flash-attn==2.7.4.post1 --no-build-isolation
  deactivate
fi

# --- env 2: fl1 (torch 2.6.0, transformers 4.57.1) ---
if [ ! -d $W/envs/fl1 ]; then
  uv venv $W/envs/fl1 --python 3.10
  source $W/envs/fl1/bin/activate
  (cd Future-L1 && grep -v flash_attn requirements_sft.txt > /tmp/req_fl1.txt && uv pip install -r /tmp/req_fl1.txt)
  uv pip install flash-attn==2.7.4.post1 --no-build-isolation
  (cd Future-L1/lmms-eval && uv pip install -e .)
  deactivate
fi

# --- Future-L1 patches: LoRA wrap (upstream never applies it) + answer-only mode / response-mode fix ---
cd $W/Future-L1
for P in fl1_lora fl1_response_mode; do
  if git apply --check $W/F-CoVT/runs/patches/$P.patch 2>/dev/null; then git apply $W/F-CoVT/runs/patches/$P.patch && echo "applied $P"
  else echo "skip $P (already applied?)"; fi
done
echo "setup done"
