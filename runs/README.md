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
```

After each step: `git add logs && git commit -m logs && git push`. Checkpoints/data stay outside the repo.
`03_fl1_qwen25_train.sh` (Future-L1's own latent training) is parked.
