"""Merge a LoRA run (B or C) into a plain Qwen2.5-VL checkpoint for evaluation.
usage: python merge_lora.py <run_dir> <out_dir>"""
import sys, glob, os
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

run, out = sys.argv[1], sys.argv[2]
adapter = run if os.path.exists(f"{run}/adapter_config.json") else sorted(
    glob.glob(f"{run}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
src = next((d for d in (run, adapter) if os.path.exists(f"{d}/tokenizer_config.json")), None)
if src:
    proc = AutoProcessor.from_pretrained(src)
else:  # trainer didn't save the processor: rebuild it with the 3 latent tokens train.py adds
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    proc.tokenizer.add_tokens(["<|latent|>", "<|latent_start|>", "<|latent_end|>"], special_tokens=False)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu")
base.resize_token_embeddings(len(proc.tokenizer))
model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
proc.save_pretrained(out)
print("merged", adapter, "->", out)
