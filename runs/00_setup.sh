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
# Uses our minimal requirements; theirs is a full env dump with conflicting pins.
# Marker file so a half-built env from a failed run is rebuilt.
if [ ! -f $W/envs/fl1/.ok ]; then
  rm -rf $W/envs/fl1
  uv venv $W/envs/fl1 --python 3.10
  source $W/envs/fl1/bin/activate
  uv pip install -r $W/F-CoVT/runs/fl1_requirements.txt
  uv pip install flash-attn==2.7.4.post1 --no-build-isolation
  (cd $W/Future-L1/lmms-eval && uv pip install -e .)
  touch $W/envs/fl1/.ok
  deactivate
fi

# --- RunPod sets HF_HUB_ENABLE_HF_TRANSFER=1, which needs hf_transfer in every env ---
for E in covt fl1; do uv pip install -q --python $W/envs/$E/bin/python hf_transfer; done

# --- Future-L1 patches: LoRA wrap (upstream never applies it) + answer-only mode / response-mode fix ---
cd $W/Future-L1
for P in fl1_lora fl1_response_mode fl1_frames_dir fl1_trainer_import fl1_context_frames fl1_fcovt fl1_fcovt_staged fl1_fcovt_staged_fix fl1_fcovt_staged_dbg fl1_fcovt_staged_pos fl1_fcovt_emb_init; do
  if git apply --check $W/F-CoVT/runs/patches/$P.patch 2>/dev/null; then git apply $W/F-CoVT/runs/patches/$P.patch && echo "applied $P"
  else echo "skip $P (already applied?)"; fi
done
echo "setup done"
