# F-CoVT

Future visual thoughts for action anticipation.

## Results

- [Assembly101 frame-split study (frozen DINOv2)](results/a101/frame_split_study.md): which past frames to use
  and which future frame is worth imagining. Best: dense last 2 s as input, middle + end of the next action as
  future targets (verb class-mean top-5 recall 28.8 past-only, 39.4 with the real future).
- Model A v1 (JEPA on frozen DINOv2): oracle future worth +14 verb, but only ~5% of the future is predictable;
  no gain over the baseline (Gate 1 failed). Same page, last section.
- [Official anticipation set, C10119, TSM features](results/a101/official_anticipation.md): history + future teaching
  (kd) reaches 59.8 verb / 34.7 object / 11.7 action, vs 58.7 / 32.0 / 11.1 for the official TempAgg checkpoint on
  the same camera; future teaching adds +3.2 object and +0.9 action over the same student without it.
