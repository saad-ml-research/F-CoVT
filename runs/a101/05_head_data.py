"""Classification-head version of the Assembly101 questions: same ids, frames and labels, but the question has no
options ("What will the person do next?"). The answer is predicted by the action head from the hidden state at the
end of the reply's "<answer>"; action_id / verb_id / noun_id are the labels.
usage: python 05_head_data.py /workspace/data/a101      (writes <split>_head.json for train / validation / test)
"""
import json, os, sys

d = sys.argv[1]
Q = "frame_1\n<image>\nWhat will the person do next?"
for split in ("train", "validation", "test"):
    p = os.path.join(d, f"{split}.json")
    if not os.path.exists(p):
        continue
    qs = json.load(open(p))
    for q in qs:
        q["conversations"][0]["value"] = Q
        q.pop("options", None)
    json.dump(qs, open(os.path.join(d, f"{split}_head.json"), "w"))
    print(f"{split}: {len(qs)} questions, {len({q['action_id'] for q in qs})} distinct actions, "
          f"{len({q['verb_id'] for q in qs})} verbs, {len({q['noun_id'] for q in qs})} objects")
