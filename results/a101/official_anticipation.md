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

### Option A: 3 x 3 grid per frame instead of one mean (layer 27; `runs/qwen3/05_features_grid.py`)

Same probe, 3 seeds; all rows from the same run (`--tag grid3`).

| row | verb | object | action |
|---|---|---|---|
| VLM 4 recent frames | 56.6 ± 1.3 | 23.6 ± 0.5 | 6.4 ± 0.4 |
| VLM history + recent | 58.0 ± 1.1 | 24.4 ± 0.9 | 7.0 ± 0.7 |
| TSM history + recent | 60.0 ± 0.6 | 32.8 ± 0.6 | 10.8 ± 0.3 |
| VLM grid + TSM (fuse) | 61.3 ± 0.7 | 31.5 ± 1.1 | 10.7 ± 0.5 |
| TSM + future teaching | 60.3 ± 0.7 | 34.1 ± 0.4 | 11.1 ± 0.5 |
| **VLM grid + TSM + future teaching** | 59.7 ± 1.2 | **36.0 ± 0.6** | **11.9 ± 0.1** |
| oracle teacher | 81.0 ± 0.9 | 45.9 ± 1.0 | 26.7 ± 0.2 |

The grid does not help the VLM on its own, but with TSM and future teaching it gives our best action score so far
(paired per seed vs TSM + teaching: object +1.9, action +0.8). The VLM adds information only once the teacher
tells the student what to look for.

### Option B: past actions as label tokens (`runs/qwen3/06_recog.py`, probe `--recog --tag hist`)

Last 8 past actions (ending before the cut) as tokens = learned embeddings of their action / verb / object labels.
Predicted labels: recognizer on TSM features of each past segment (validation top-1 action 35.7, verb 62.3,
object 45.3; train predictions out-of-fold by recording).

| model | past labels (train -> test) | verb | object | action |
|---|---|---|---|---|
| history labels only, no video | true -> true | 60.9 ± 0.6 | 55.4 ± 0.5 | 24.9 ± 0.6 |
| history labels only, no video | pred -> pred | 51.8 ± 0.5 | 21.6 ± 0.6 | 6.5 ± 0.2 |
| VLM grid + TSM + future teaching | none | 59.1 ± 0.9 | 35.4 ± 0.4 | 11.3 ± 0.7 |
| + history (upper bound) | true -> true | 62.2 ± 0.2 | 41.4 ± 0.2 | 15.8 ± 0.9 |
| + history (AAG's way) | true -> pred | 57.1 ± 0.6 | 34.7 ± 0.1 | 10.7 ± 0.3 |
| + history (fair) | pred -> pred | 60.2 ± 1.3 | 34.6 ± 1.0 | 11.4 ± 0.3 |
| TSM + future teaching + history | pred -> pred | 59.4 ± 0.1 | 33.0 ± 1.0 | 11.2 ± 0.4 |

True past labels alone match AAG's true-history result (24.86); fair predicted history adds nothing.

### B+: privileged teacher (true past labels + future) and soft history (`--tag bplus`)

oracleH = oracle + true past-action labels (teacher only). Soft history (+hS) = the recognizer's probabilities for
each past action, turned into an expected label embedding.

| model | teacher | verb | object | action | action top-1 |
|---|---|---|---|---|---|
| oracle (teacher) | – | 80.7 ± 0.1 | 45.8 ± 0.6 | 27.1 ± 0.3 | 35.8 ± 0.5 |
| oracleH (teacher) | – | 82.0 ± 0.6 | 53.1 ± 0.6 | 34.0 ± 1.0 | 40.6 ± 0.2 |
| VLM grid + TSM | oracle | 59.1 ± 0.9 | 35.3 ± 0.3 | 11.3 ± 0.6 | 11.7 ± 0.4 |
| VLM grid + TSM | oracleH | 59.4 ± 0.8 | 34.9 ± 0.8 | 11.5 ± 0.2 | 11.6 ± 0.2 |
| VLM grid + TSM + soft history | oracle | 61.8 ± 1.6 | 34.1 ± 1.0 | 11.6 ± 0.1 | 12.8 ± 0.9 |
| VLM grid + TSM + soft history | oracleH | 59.1 ± 1.1 | 35.0 ± 0.6 | 11.5 ± 0.1 | 11.7 ± 0.0 |
| TSM only | oracleH | 59.0 ± 0.8 | 33.7 ± 1.0 | 10.6 ± 0.3 | 12.2 ± 0.4 |

A much stronger teacher does not make a better student: the frozen-feature students plateau at action 11.3–11.9.

### True-history setting and long history (`--tag hist2`, recognizer with 256-action histories)

Histories: median ~100 past actions per segment; 256 covers ~95% of segments in full.

| model | setting | verb | object | action | action top-1 | action top-5 acc |
|---|---|---|---|---|---|---|
| true labels only, last 8 | true history | 60.9 ± 0.7 | 55.1 ± 0.6 | 24.4 ± 0.4 | 14.6 ± 0.5 | 41.7 ± 0.5 |
| true labels only, whole history | true history | 60.5 ± 0.4 | 53.0 ± 1.3 | 23.2 ± 0.7 | 14.1 ± 0.3 | 40.9 ± 0.5 |
| video + last 8 true labels, no teacher | true history | 62.3 ± 1.1 | 40.6 ± 1.5 | 17.9 ± 0.3 | 14.9 ± 0.8 | 38.6 ± 2.1 |
| video + whole true history, no teacher | true history | 62.3 ± 0.6 | 39.1 ± 1.2 | 16.0 ± 0.7 | 13.8 ± 0.4 | 36.7 ± 1.0 |
| video + last 8 true labels, history-aware teacher | true history | 64.5 ± 1.6 | 42.4 ± 0.1 | 18.3 ± 0.8 | 16.2 ± 0.6 | 40.1 ± 2.8 |
| video + soft history, last 8 | fair | 59.3 ± 0.8 | 34.8 ± 0.7 | 11.3 ± 0.6 | 11.7 ± 0.4 | 30.0 ± 0.9 |
| video + soft history, whole video | fair | 59.3 ± 1.0 | 35.6 ± 0.8 | 11.5 ± 0.5 | 12.0 ± 1.0 | 30.7 ± 2.4 |
| soft history only, whole video | fair | 55.7 ± 0.3 | 26.0 ± 0.3 | 7.9 ± 0.3 | 9.0 ± 0.4 | 27.2 ± 0.4 |
| oracle teacher | teacher | 80.5 ± 0.2 | 45.9 ± 1.0 | 27.1 ± 0.2 | 36.2 ± 0.1 | 58.5 ± 0.3 |
| oracleH teacher (+ last 8 true labels) | teacher | 81.3 ± 0.4 | 52.8 ± 0.9 | 34.5 ± 0.6 | 41.1 ± 0.3 | 64.1 ± 0.2 |

With true labels, adding video hurts (with or without a teacher), likely because the released TSM features were
fine-tuned on the training videos. Whole history does not beat the last 8; long soft history adds only +0.2.

### Noise curve, aligned video history, late fusion (`--tag combo`)

hN = true past labels with that share replaced by a random training segment's labels (train and test). hV = TSM
features of each past action segment (annotated boundaries). Late fusion = sum of two models' log-probabilities.

| model | verb | object | action | action top-1 | action top-5 acc |
|---|---|---|---|---|---|
| true labels only, 0% wrong | 60.9 ± 0.7 | 55.1 ± 0.6 | 24.4 ± 0.4 | 14.6 ± 0.5 | 41.7 ± 0.5 |
| 10% wrong | 57.8 ± 0.3 | 51.4 ± 0.7 | 21.3 ± 0.6 | 11.9 ± 0.2 | 36.0 ± 0.4 |
| 20% wrong | 56.4 ± 0.8 | 46.6 ± 0.6 | 18.4 ± 0.2 | 11.2 ± 1.0 | 33.4 ± 2.9 |
| 40% wrong | 50.7 ± 0.6 | 37.4 ± 0.4 | 12.7 ± 0.0 | 8.8 ± 0.3 | 27.9 ± 1.5 |
| 60% wrong | 45.1 ± 0.2 | 28.8 ± 1.0 | 8.3 ± 0.1 | 6.6 ± 0.5 | 23.1 ± 0.6 |
| aligned video history only | 53.7 ± 0.6 | 28.5 ± 1.0 | 7.9 ± 0.2 | 9.5 ± 0.1 | 28.7 ± 0.2 |
| soft history only (whole video) | 55.7 ± 0.3 | 26.0 ± 0.3 | 7.9 ± 0.3 | 9.0 ± 0.4 | 27.2 ± 0.4 |
| VLM grid + TSM + teaching | 59.3 ± 1.3 | 35.5 ± 1.1 | 11.5 ± 0.3 | 11.5 ± 0.4 | 30.4 ± 0.6 |
| + aligned video history (8) | 60.6 ± 1.6 | 35.7 ± 1.7 | 11.5 ± 0.2 | 12.1 ± 0.8 | 32.0 ± 3.1 |
| + aligned video history (32) | 60.0 ± 1.3 | 35.7 ± 0.6 | 11.4 ± 0.6 | 12.2 ± 0.4 | 31.9 ± 1.6 |
| fusion: video model + soft history | 60.9 ± 0.6 | 32.8 ± 0.8 | 11.8 ± 0.1 | 13.0 ± 0.4 | 34.7 ± 0.4 |
| fusion: video model + aligned video history | 60.2 ± 0.7 | 33.8 ± 0.6 | 11.5 ± 0.3 | 12.7 ± 0.2 | 34.3 ± 0.5 |
| fusion: true labels only + video model | 63.8 ± 0.4 | 49.3 ± 1.0 | 21.6 ± 0.9 | 16.8 ± 0.4 | 43.9 ± 0.5 |

History alone beats the video model only below ~45% label noise (recognition above ~55% correct).

### History as text through the frozen VLM (`runs/qwen3/11_text_feats.py`, probe `--tag text`)

Each past step = mean of its words' layer-27 states in "Steps so far, oldest first: ...; ..." (no images, last 8).

| model | history given as | verb | object | action | action top-1 | action top-5 acc |
|---|---|---|---|---|---|---|
| history only | true labels, learned vectors | 61.2 ± 0.4 | 55.7 ± 0.9 | 24.8 ± 0.4 | 14.3 ± 0.1 | 41.6 ± 0.6 |
| history only | true labels as text | 62.1 ± 0.2 | 54.1 ± 0.5 | 22.3 ± 0.1 | 14.5 ± 0.2 | 41.3 ± 0.3 |
| history only | predicted labels, learned vectors | 51.5 ± 0.4 | 21.8 ± 0.2 | 6.4 ± 0.3 | 8.5 ± 0.0 | 25.3 ± 0.6 |
| history only | predicted labels as text | 53.0 ± 0.2 | 22.9 ± 0.4 | 6.9 ± 0.2 | 8.8 ± 0.2 | 25.7 ± 0.6 |
| VLM grid + TSM + teaching | none | 58.1 ± 0.9 | 34.6 ± 0.4 | 11.2 ± 0.6 | 11.6 ± 0.4 | 30.6 ± 1.2 |
| + predicted text history | predicted as text | 61.9 ± 1.1 | 34.5 ± 0.4 | 11.5 ± 0.4 | 12.6 ± 0.9 | 33.1 ± 2.2 |
| late fusion with predicted-text-only model | predicted as text | 60.0 ± 0.3 | 30.4 ± 0.2 | 10.5 ± 0.3 | 13.0 ± 0.4 | 34.5 ± 1.0 |

Text does not beat learned label vectors; history mainly helps the verb and accuracy, not class-mean action recall.
