# Assembly101 official anticipation set, camera C10119 (TSM features)

Official anticipation labels (17 verbs, 90 objects, 1,063 actions; from the TempAgg release), predicting each
segment 1 s before it starts. Train: 41,229 segments (train.csv rows of camera C10119). Validation: 13,494 segments
(validation_challenge). Features: the released Assembly101 TSM features (2,048-D per frame, 30 fps, fine-tuned on
Assembly101 recognition), max-pooled into windows (`runs/tsm/01_bank.py`). Probe: attentive probe, 3 queries, focal
loss (`runs/tsm/02_teach.py`). Metric: class-mean top-5 recall (same computation as the official code). 3 seeds.

| model | trained on | verb | object | action | action top-1 |
|---|---|---|---|---|---|
| TempAgg, official released checkpoint, tested on C10119 | all 12 cameras | 58.7 | 32.0 | 11.1 | - |
| TempAgg, trained by us with the official code (`--views view4`, 15 epochs) | C10119 | 56.8 | 32.4 | 10.1 | - |
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
TempAgg's own code picks its best epoch on the validation set; ours picks on 10% of train held out, so the comparison
favours TempAgg slightly. Same data (C10119 only): kd is +3.0 verb, +2.3 object, +1.6 action over TempAgg.

## Ablation: KD weight x temperature, and feature distillation (2 seeds, mean)

Student = history + recent (hist); teacher = oracle. Same split and metric as above.

| KD weight | temperature | verb | object | action |
|---|---|---|---|---|
| no teacher (hist) | - | 59.5 | 32.3 | 10.9 |
| 0.5 | 1 / 2 / 4 | 59.2 / 59.9 / 59.2 | 33.2 / 33.9 / 33.5 | 11.0 / 11.5 / 11.4 |
| 1 | 1 / 2 / 4 | 60.3 / 60.1 / 58.2 | 33.5 / 34.0 / 33.7 | 10.7 / 11.3 / 11.5 |
| 2 | 1 / 2 / 4 | 60.1 / 59.3 / 58.1 | 33.8 / 34.5 / 34.0 | 11.2 / 11.5 / 10.9 |
| 4 | 1 / 2 / 4 | 60.5 / 59.2 / 58.4 | 33.8 / 34.5 / 34.5 | 11.3 / 11.2 / 10.9 |
| 1 + feature distillation x1 / x5 | 2 | 59.9 / 59.7 | 34.3 / 34.3 | 11.3 / 10.8 |

- Every setting improves object over the no-teacher student (+0.9 to +2.2).
- Temperature 4 costs about 1 verb point; temperatures 1-2 are the useful range.
- Best trade-offs: weight 2 / temperature 2 (object 34.5, action 11.5) and weight 4 / temperature 1 (verb 60.5).
- Feature distillation (cosine between the 3 query vectors) adds nothing over plain KD.
- Neighbouring settings differ by less than ~1 point, about the seed noise with 2 seeds.

## VLM ladder: Qwen3-VL-8B (frozen) on the same set

Same segments, same probe, metric and 3 seeds as above. Zero-shot = the model lists 5 "verb | object" answers
(`runs/qwen3/02_zeroshot.py`). Probe rows use frozen Qwen3-VL features: per frame, the mean of its image tokens at a
given LM layer (`runs/qwen3/03_features.py`, probe `runs/qwen3/04_probe.py`). History = 8 frames over the last 60 s,
recent = 4 frames over the last 2 s. KD: lambda 2, T 2, teacher = TSM history + recent + real future.

| rung | row | verb | object | action |
|---|---|---|---|---|
| 0 | zero-shot (recent frames, 1 run) | 32.3 | 6.5 | 0.8 |
| 1 | VLM last prompt token | 56.0 ± 0.5 | 20.6 ± 0.4 | 5.3 ± 0.1 |
| 1 | VLM 4 recent frames (layer 27) | 57.9 ± 1.9 | 24.6 ± 0.4 | 7.3 ± 0.1 |
| 2a | VLM history + recent, layer 18 | 56.9 ± 0.4 | 24.5 ± 0.4 | 7.2 ± 0.3 |
| 2a | VLM history + recent, layer 27 | 57.5 ± 1.4 | 24.8 ± 0.7 | 7.2 ± 0.3 |
| 2a | VLM history + recent, layer 36 | 55.9 ± 2.4 | 22.4 ± 0.3 | 6.4 ± 0.2 |
| ref | TSM history + recent | 59.9 ± 0.6 | 32.2 ± 0.6 | 10.7 ± 0.1 |
| 3 | VLM + TSM (fuse) | 60.2 ± 1.6 | 32.1 ± 1.1 | 11.0 ± 0.3 |
| 4 | VLM + TSM + future teaching | 58.9 ± 1.4 | 34.9 ± 0.5 | **11.5 ± 0.3** |
| ref | TSM + future teaching | 58.7 ± 0.3 | 35.7 ± 1.0 | 11.2 ± 0.4 |
| – | oracle teacher (sees the future) | 80.3 ± 0.2 | 46.4 ± 0.7 | 27.1 ± 0.2 |

Findings: probing frozen features lifts the VLM far above zero-shot; it is competitive on verbs but weak on objects
(mean-pooled frames, no Assembly101 training, while TSM was fine-tuned on Assembly101). VLM history adds nothing over
the recent frames; layer 36 is worst. Adding the frozen VLM to TSM is within noise. Future teaching again helps
objects (+2.8 over fuse).
