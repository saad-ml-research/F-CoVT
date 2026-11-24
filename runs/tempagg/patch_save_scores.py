"""Add --save_scores PATH to TempAgg's main.py (validate mode): saves the raw verb / object / action scores, labels and
ids of the validation segments to an .npz, so they can be scored with our metrics and logit adjustment (la_eval.py).
usage: python patch_save_scores.py /workspace/assembly101-action-anticipation/tempAgg-action-anticipation/main.py
"""
import sys
p = sys.argv[1]
s = open(p).read()
if "--save_scores" in s:
    print("already patched"); sys.exit()
a = "parser.add_argument('--debug_on'"
assert a in s
s = s.replace(a, "parser.add_argument('--save_scores', type=str, default=None, help='save raw validation scores (.npz)')\n" + a, 1)
b = "        verb_accuracies = topk_accuracy(verb_scores, verb_labels, (1,))[0]"
assert b in s
s = s.replace(b, """        if args.save_scores:
            np.savez(args.save_scores, verb=verb_scores, object=object_scores, action=action_scores,
                     verb_labels=verb_labels, object_labels=object_labels, action_labels=action_labels, ids=ids)
            print(f'scores saved to {args.save_scores} ({len(ids)} segments)')
""" + b, 1)
open(p, "w").write(s)
print("patched", p)
