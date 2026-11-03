# F-CoVT

Future visual thoughts for action anticipation.

## Results

- [Assembly101 frame-split study (frozen DINOv2)](results/a101/frame_split_study.md): which past frames to use
  and which future frame is worth imagining. Best: dense last 2 s as input, middle + end of the next action as
  future targets (verb class-mean top-5 recall 28.8 past-only, 39.4 with the real future).
