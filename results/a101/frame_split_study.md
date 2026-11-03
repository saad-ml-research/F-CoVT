# Assembly101 frame-split study (frozen DINOv2)

Which past frames to look at, and which future frame is worth imagining, for next-action anticipation on
Assembly101 (fixed camera C10119; the cut is 1 s before the next action starts).

**Setup.** Frozen DINOv2 ViT-L/14 features ([CLS ; mean patch], 2,048 dims) of the frames named below
(`runs/a101/09_dino_bank.py`). A linear probe (LayerNorm + linear; epoch chosen on 10% of train) predicts the
next action / verb / object (`runs/a101/10_split_study.py`). Train: 10,000 questions; validation: 2,000 questions
from different recordings (264 recordings in total). Metric: class-mean top-5 recall (official Assembly101 metric).
"Oracle" = the probe also sees the REAL future frame (an upper bound, not usable at test time).

| past option | frames (seconds relative to the cut) |
|---|---|
| P1 f5 only | 0 |
| P2 5 x 1 s | -4, -3, -2, -1, 0 |
| P3 8 x 0.25 s | -1.75 ... 0, every 0.25 s |
| P4 8 x 1 s | -7 ... 0, every 1 s |
| P5 long + dense | -8, -6, -4, -2, -1.5, -1, -0.5, 0 |

| target option | frame |
|---|---|
| T1 | +0.5 s (before the next action starts) |
| T2 | +1.0 s (the next action starts) |
| T3 | +1.5 s (inside the next action; about our old f6) |
| T4 | end of the next action |
| T5 | end of the coarse step (our old f7, ~11 s ahead) |

## Ablation 1: 5 pasts x 5 targets (seed 0)

Verb class-mean top-5 recall gain over the past-only probe, **oracle (real future) / predicted (ridge guess)**.

| past (verb, past only) | T1 +0.5 s | T2 +1.0 s | T3 +1.5 s | T4 next action end | T5 step end |
|---|---|---|---|---|---|
| P1 f5 only (29.2) | +1.7 / +2.2 | +7.5 / +1.8 | +8.6 / +1.5 | +9.5 / +1.8 | +2.8 / +1.6 |
| P2 5 x 1 s (28.6) | +2.3 / +1.7 | +5.8 / +1.2 | +8.9 / +1.7 | **+11.0** / +1.4 | +2.8 / +1.6 |
| P3 8 x 0.25 s (30.2) | +2.8 / +1.4 | +5.0 / +2.2 | +7.7 / +0.4 | +8.7 / +0.6 | +1.1 / -1.7 |
| P4 8 x 1 s (28.1) | +1.8 / +0.3 | +5.3 / +0.3 | +5.3 / -0.0 | +5.3 / +0.1 | +3.6 / -1.1 |
| P5 long + dense (31.0) | -0.8 / -1.7 | +1.8 / -2.8 | +4.2 / -1.4 | +4.9 / -1.7 | +2.5 / -2.6 |

"Explained": share of the change from the current frame to the target that the ridge predictor recovers
(1 - |pred - real|^2 / |now - real|^2; > 0 means better than copying the current frame).

| past | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|
| P1 | -0.04 | 0.04 | 0.08 | 0.10 | 0.15 |
| P2 | -0.73 | -0.58 | -0.53 | -0.50 | -0.40 |
| P3 | -0.94 | -0.83 | -0.78 | -0.76 | -0.59 |
| P4 | -1.20 | -1.00 | -0.91 | -0.86 | -0.68 |
| P5 | -1.14 | -0.95 | -0.89 | -0.84 | -0.66 |

Object / action oracle gains are in `split_study_seed0.json` on the pod (`/workspace/outputs/dino_bank`).

**Reading.**
- The useful future is inside / at the end of the next action (T3, T4: +8 to +11 verb). +0.5 s adds little
  (looks like now); the step end (f7) adds little for the *next* action.
- Past-only scores are 28-31 for every past option: more or denser frames do not help a linear probe.
- Caveats: one seed (differences of ~1-2 points are noise). The "predicted" column is not informative by
  construction (a linear guess of the future is a linear function of the past, so it adds no information to a
  linear probe), and ridge on many frames overfits (negative "explained"); predictability is tested properly by
  model A (`runs/a101/11_jepa.py`), which predicts the change from now with a nonlinear predictor.

## Ablation 2: combined pasts and targets (oracle only, 3 seeds, mean +- sd)

| past | target(s) | verb: past only | verb: + real future | verb gain | object | action |
|---|---|---|---|---|---|---|
| **P3** (8 frames) | **T4 + T3** | 28.8 +- 1.4 | **39.4 +- 1.9** | **+10.6 +- 2.3** | 13.9 -> **17.3** | 3.1 -> **4.4** |
| P3 (8 frames) | T4 | 28.8 +- 1.4 | 37.4 +- 1.4 | +8.6 +- 1.1 | 13.9 -> 16.9 | 3.1 -> 4.0 |
| P3 + P5 (12 frames) | T4 + T3 | 28.3 +- 1.3 | 37.7 +- 1.1 | +9.3 +- 1.9 | 13.7 -> 15.6 | 3.0 -> 4.1 |
| P5 (8 frames) | T4 + T3 | 29.7 +- 1.1 | 37.4 +- 0.4 | +7.7 +- 0.8 | 13.7 -> 15.8 | 3.3 -> 4.2 |
| P5 (8 frames) | T4 | 29.7 +- 1.1 | 35.9 +- 0.1 | +6.1 +- 1.1 | 13.7 -> 15.8 | 3.3 -> 4.1 |
| P3 + P5 (12 frames) | T4 | 28.3 +- 1.3 | 33.7 +- 1.8 | +5.4 +- 1.0 | 13.7 -> 14.1 | 3.0 -> 3.6 |

**Reading.** Best setup: dense last 2 s (P3) as input, with the middle (T3) and the end (T4) of the next action
as future targets. Two targets beat one (T3 and T4 carry different information). Adding long-range context (P5)
does not help. This is the setup used for model A: past-only ~28.8 verb, oracle ceiling ~39.4.

Reproduce: `python runs/a101/10_split_study.py /workspace/outputs/dino_bank` (ablation 1) and
`python runs/a101/10_split_study.py /workspace/outputs/dino_bank --combos "P3+P5:T4+T3,P3+P5:T4,P3:T4+T3,P5:T4+T3,P3:T4,P5:T4"` (ablation 2).
