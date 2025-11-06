#!/usr/bin/env python3
import os
# ---- Silence GPU probes & noisy logs (CPU-only machine) ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"
os.environ["FAISS_DISABLE_GPU"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# -------- Buckets --------
LINEAR_OPS = {"aten::linear", "aten::addmm", "aten::mm", "aten::bmm", "aten::matmul"}
NONLINEAR_OPS = {
    "aten::gelu", "aten::gelu_backward", "aten::tanh", "aten::erf",
    "aten::softmax", "aten::_softmax", "aten::softmax_backward_data",
    "aten::native_layer_norm", "aten::native_layer_norm_backward",
    "aten::layer_norm", "aten::layer_norm_backward",
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_flash_attention",
}

def bucket_name(op_name: str) -> str:
    if op_name in LINEAR_OPS:
        return "linear"
    if op_name in NONLINEAR_OPS:
        return "nonlinear"
    return "other"

def aggregate_ms(prof) -> tuple[float, float, float]:
    events = prof.key_averages(group_by_input_shape=False)
    times_us = defaultdict(float)
    total_us = 0.0
    for e in events:
        t_us = float(e.self_cpu_time_total)
        total_us += t_us
        times_us[bucket_name(e.key)] += t_us
    linear_ms = times_us["linear"] / 1000.0
    nonlinear_ms = times_us["nonlinear"] / 1000.0
    total_ms = total_us / 1000.0
    return linear_ms, nonlinear_ms, total_ms

# ---- Workloads ----
def generate_with_cache(model, tokenizer, text: str, max_new_tokens: int):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    with torch.inference_mode():
        # Build initial cache from full prompt
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        cur = input_ids[:, -1:]
        for _ in range(max_new_tokens):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

def generate_no_cache(model, tokenizer, text: str, max_new_tokens: int):
    """Naive loop: re-feed the entire growing sequence each step; no KV cache."""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids, use_cache=False)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--text", type=str, default="Dear GPT, I would like you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something.",
                        help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=32)
    args = parser.parse_args()

    # Threads
    torch.set_num_threads(max(1, args.num_threads))
    try:
        torch.set_num_interop_threads(max(1, min(args.num_threads, 8)))
    except Exception:
        pass
    set_seed(42)
    device = torch.device("cpu")

    # Model / tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    model.config.use_cache = True

    # Warmup
    with torch.inference_mode():
        _ = model(**tokenizer("warmup", return_tensors="pt").to(device))

    # ---- Profile NO-CACHE ----
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False, with_stack=False, profile_memory=False, with_flops=False,
    ) as prof_nc:
        generate_no_cache(model, tokenizer, args.text, args.max_new_tokens)
    nc_lin_ms, nc_nonlin_ms, nc_total_ms = aggregate_ms(prof_nc)

    # ---- Profile WITH KV-CACHE ----
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False, with_stack=False, profile_memory=False, with_flops=False,
    ) as prof_kv:
        generate_with_cache(model, tokenizer, args.text, args.max_new_tokens)
    kv_lin_ms, kv_nonlin_ms, kv_total_ms = aggregate_ms(prof_kv)

    # ---- Print exactly two lines, labeled, ms ----
    print(f"no_cache   -> total_linear: {nc_lin_ms:.3f}  total_nonlinear: {nc_nonlin_ms:.3f}  total: {nc_total_ms:.3f}")
    print(f"kv_cache   -> total_linear: {kv_lin_ms:.3f}  total_nonlinear: {kv_nonlin_ms:.3f}  total: {kv_total_ms:.3f}")

if __name__ == "__main__":
    main()
