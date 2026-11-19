"""Qwen3-VL-8B smoke test: load, answer one question about synthetic frames, report memory, speed and the
hidden-state size we will probe. usage: python 00_check.py [Qwen/Qwen3-VL-8B-Instruct]"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-8B-Instruct"
t0 = time.time()
proc = AutoProcessor.from_pretrained(name)
model = Qwen3VLForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
print(f"loaded {name} in {time.time() - t0:.0f}s | {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B params", flush=True)

frames = []
for k in range(4):                                            # a hand (circle) moving toward a wheel (square)
    im = Image.new("RGB", (448, 252), (235, 235, 235)); d = ImageDraw.Draw(im)
    d.rectangle([300, 110, 360, 170], fill=(40, 40, 40)); d.ellipse([60 + 50 * k, 120, 100 + 50 * k, 160], fill=(220, 170, 140))
    frames.append(im)
content = [{"type": "text", "text": "These are the last 4 frames of a video, oldest first."}]
content += [{"type": "image", "image": f} for f in frames]
content.append({"type": "text", "text": "List the 5 most likely next actions of the person, one per line, as 'verb object'."})
msgs = [{"role": "user", "content": content}]
inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
torch.cuda.reset_peak_memory_stats(); t = time.time()
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=60, do_sample=False)
    h = model(**inp, output_hidden_states=True).hidden_states
print(f"prompt tokens {inp['input_ids'].shape[1]} | generate+forward {time.time() - t:.1f}s | peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"hidden states: {len(h)} layers x dim {h[-1].shape[-1]}")
print("answer:", proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0])
