# Run order (pod)

```bash
cd /workspace/F-CoVT && git pull
bash runs/00_setup.sh                 # once, ~20-30 min
bash runs/01_covt_eval.sh             # CoVT vs base on CV-Bench, ~1 h
bash runs/02_fl1_data_check.sh        # download Future-L1-50K, print what it needs
# if TwiFF videos are on disk:
python runs/make_subset.py /workspace/data/Future-L1-50K/<file>.json /workspace/data/fl1_10k.json 10000
SMOKE=50 bash runs/03_fl1_qwen25_train.sh   # speed test, read s/it from the log
bash runs/03_fl1_qwen25_train.sh            # row D, ~3 h on 10K
```

After each step: `git add logs && git commit -m "logs" && git push` so the logs can be read from GitHub.
Never commit checkpoints or data (they are in /workspace/outputs and /workspace/data, outside the repo).
