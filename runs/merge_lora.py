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
# F-CoVT runs: training set the thought-token rows to the mean real-token embedding inside the BASE model
# (frozen, so not in the adapter). Reproduce that here, or eval would see different thought embeddings.
tok = proc.tokenizer
if "<|fcovt_start|>" in tok.get_vocab():
    names = ["<|fcovt_start|>", "<|fcovt_end|>", "<|fcovt_seg|>", "<|fcovt_depth|>", "<|fcovt_dino|>", "<|fcovt_lat|>"]
    ids = [tok.convert_tokens_to_ids(t) for t in names]
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    with torch.no_grad():
        emb = base.get_input_embeddings().weight
        emb[ids] = emb[:latent_id].float().mean(0, keepdim=True).to(emb.dtype)
    print(f"set {len(ids)} F-CoVT thought-token embeddings to the mean of rows [0, {latent_id})")
model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
# row F: the thought tokens' OUTPUT rows were trained as heads.out_rows (so the model can write them itself)
hp = os.path.join(run, "fcovt_heads.pt")
if os.path.exists(hp) and "out_rows" in (sd := torch.load(hp, map_location="cpu")):
    with torch.no_grad():
        model.get_output_embeddings().weight[ids] = sd["out_rows"].to(model.get_output_embeddings().weight.dtype)
    print(f"set {len(ids)} F-CoVT thought-token lm_head rows from fcovt_heads.pt (row F)")
model.save_pretrained(out, safe_serialization=True)
proc.save_pretrained(out)
print("merged", adapter, "->", out)
