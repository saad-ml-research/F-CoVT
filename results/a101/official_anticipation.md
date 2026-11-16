# Assembly101 official anticipation set, camera C10119 (TSM features)

Official anticipation labels (17 verbs, 90 objects, 1,063 actions; from the TempAgg release), predicting each
segment 1 s before it starts. Train: 41,229 segments (train.csv rows of camera C10119). Validation: 13,494 segments
(validation_challenge). Features: the released Assembly101 TSM features (2,048-D per frame, 30 fps, fine-tuned on
Assembly101 recognition), max-pooled into windows (`runs/tsm/01_bank.py`). Probe: attentive probe, 3 queries, focal
loss (`runs/tsm/02_teach.py`). Metric: class-mean top-5 recall (same computation as the official code). 3 seeds.

| model | trained on | verb | object | action | action top-1 |
|---|---|---|---|---|---|
| TempAgg, official released checkpoint, tested on C10119 | all 12 cameras | 58.7 | 32.0 | 11.1 | - |
| recent (last 2 s) | C10119 | 53.2 +- 0.3 | 30.8 +- 0.2 | 8.1 +- 0.2 | 9.5 |
| hist (last 60 s + last 2 s) | C10119 | 59.5 +- 0.1 | 31.5 +- 0.8 | 10.8 +- 0.0 | 11.8 |
| **kd (hist student + future teacher)** | C10119 | **59.8 +- 0.6** | **34.7 +- 0.3** | **11.7 +- 0.4** | **12.0** |
| kd_rec (recent student + future teacher) | C10119 | 53.1 +- 0.4 | 32.7 +- 0.8 | 8.3 +- 0.1 | 9.8 |
| oracle (real next 2 s, cheating; the teacher) | C10119 | 81.3 +- 0.9 | 45.8 +- 1.4 | 26.9 +- 0.6 | 36.5 |

Gains (mean +- sd over seeds): history over recent: verb +6.3 +- 0.3, object +0.7 +- 0.9, action +2.7 +- 0.1.
Future teaching over the history student: verb +0.3 +- 0.6, **object +3.2 +- 1.0**, **action +0.9 +- 0.4**.
Future teaching over the recent student: object +1.9 +- 1.0.

TempAgg row: `main.py --mode validate --modality fixed --views view4` (C10119 = their view4), 13,481 of the 13,494
segments scored (13 with a cut before 2 s are skipped by their code). Patches needed for current libraries: `vs` for
a single fixed view, `weights_only=False`, `noun -> object` key rename, one shared lmdb environment.
Next: TempAgg trained on C10119 only (same data as ours).
