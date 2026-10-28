# Run order (pod) — all rows on Qwen2.5-VL-7B

```bash
cd /workspace/F-CoVT && git pull
bash runs/00_setup.sh                       # once: 2 envs + Future-L1 patches
bash runs/01_covt_eval.sh                   # JOB 1: CoVT vs base on CV-Bench
bash runs/02_fl1_data_check.sh              # training data (Future-L1-50K): what it needs
bash runs/04_futurebench_data.sh            # eval data: list FutureBench / V1-33K files + sizes
# after data is on disk:
FB_DATA=... FB_VIDEOS=... bash runs/05_rowA_zeroshot.sh                 # row A
python runs/make_subset.py <50K json> /workspace/data/fl1_10k.json 10000
MODE=answer SMOKE=50 bash runs/06_rowBC_train.sh                         # speed test
MODE=answer bash runs/06_rowBC_train.sh                                  # row B
MODE=cot    bash runs/06_rowBC_train.sh                                  # row C
MODEL=<run>/merged TAG=rowB FB_DATA=... FB_VIDEOS=... bash runs/05_rowA_zeroshot.sh   # eval B (same for C)
# rows D/E/F (F-CoVT): bash runs/13_fcovt_setup.sh once (applies fl1_fcovt + fl1_fcovt_staged patches)
MODE=fcovt CONTEXT=4 bash runs/06_rowBC_train.sh                         # row E (FCOVT_GAMMA=0.1 / 0 for ablations)
MODE=fcovt_generic CONTEXT=4 bash runs/06_rowBC_train.sh                 # row D (generic future latents)
MODE=fcovt_staged CONTEXT=4 bash runs/06_rowBC_train.sh                  # row F: CoVT's 4-stage curriculum
MODE=fcovt_staged CONTEXT=4 SMOKE=24 GA=2 WORKERS=1 bash runs/06_rowBC_train.sh   # row F smoke: all 4 stages in ~24 steps
MODE=fcovt RUN=<run> STEPS=1 bash runs/14_eval_thoughts.sh               # eval D/E (forced thought prefix)
MODE=fcovt_staged RUN=<run> STEPS=0 bash runs/14_eval_thoughts.sh        # eval F (model writes its own thoughts)
```

Row F follows CoVT train/scripts/finetune.sh, scaled to our step budget (FCOVT_STAGES=0.375,0.5625,0.75,0.875):
1. tokens in the prompt right after the current frame, answer CE + experts (targets: current frame, FCOVT_S1_TARGET)
2. "What is the segmentation, depth map, perception feature of the future frame?" -> tokens only
3. `<think> Because the segmentation of the future frame is [..], ... </think> <answer> .. </answer>`
4. as 3, but 1/6 of samples without thoughts and a random subset of experts; expert losses off from 87.5%.
Thought tokens are in the CE (their output rows train as heads.out_rows); input rows stay frozen at the mean;
expert heads at lr 1e-5 (FCOVT_HEAD_LR).

After each step: `git add logs && git commit -m logs && git push`. Checkpoints/data stay outside the repo.
`03_fl1_qwen25_train.sh` (Future-L1's own latent training) is parked.
