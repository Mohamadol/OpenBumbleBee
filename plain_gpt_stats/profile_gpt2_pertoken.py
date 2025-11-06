#!/usr/bin/env python3
import os
# Silence CUDA / GPU logs on CPU machines
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"
os.environ["FAISS_DISABLE_GPU"] = "1"

import argparse
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# ---- Operator buckets ----
LINEAR_OPS = {"aten::linear", "aten::addmm", "aten::mm", "aten::bmm", "aten::matmul"}
NONLINEAR_OPS = {
    "aten::gelu", "aten::tanh", "aten::erf",
    "aten::softmax", "aten::_softmax",
    "aten::native_layer_norm", "aten::layer_norm",
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_flash_attention",
}

def bucket_name(op_name: str):
    if op_name in LINEAR_OPS:
        return "linear"
    if op_name in NONLINEAR_OPS:
        return "nonlinear"
    return "other"

def aggregate_ms(prof):
    events = prof.key_averages()
    sums = defaultdict(float)
    total_us = 0.0
    for e in events:
        t_us = float(e.self_cpu_time_total)
        total_us += t_us
        sums[bucket_name(e.key)] += t_us
    return sums["linear"]/1000.0, sums["nonlinear"]/1000.0, total_us/1000.0

def measure_prefill(model, tokenizer, prompt):
    """Profile full prompt forward (prefill)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with torch.inference_mode():
            _ = model(**inputs, use_cache=True)
    return aggregate_ms(prof), inputs["input_ids"].shape[-1]

def profile_loop(model, tokenizer, prompt, steps, use_cache):
    """Run generation for 'steps' tokens and profile each token separately."""
    stats = []
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs, use_cache=use_cache)
        past = out.past_key_values if use_cache else None
        input_ids = inputs["input_ids"]
        for step in range(steps):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                if use_cache:
                    out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
                    past = out.past_key_values
                else:
                    out = model(input_ids=input_ids, use_cache=False)
            lin, nonlin, tot = aggregate_ms(prof)
            stats.append((lin, nonlin, tot))
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return stats

def print_model_info(model, tokenizer, input_tokens):
    cfg = model.config
    d_model = getattr(cfg, "n_embd", None)
    n_layer = getattr(cfg, "n_layer", None)
    n_head = getattr(cfg, "n_head", None)
    vocab = getattr(cfg, "vocab_size", None)
    d_head = d_model / n_head if (d_model and n_head) else None

    print("=== Model Info ===")
    print(f"Model: {cfg._name_or_path}")
    print(f"Layers: {n_layer}")
    print(f"Model dim (d_model): {d_model}")
    print(f"Attention heads: {n_head}")
    print(f"Head dim (d_head): {d_head:.1f}")
    print(f"Vocab size: {vocab}")
    print(f"Input tokens: {input_tokens}")
    print("==================")

def main():
    parser = argparse.ArgumentParser(description="Profile GPT-2 linear vs nonlinear CPU time with prefill info")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--text", type=str, default="Dear GPT, I would like you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something.",
                        help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--mode", type=str, choices=["kv", "nocache"], default="kv",
                        help="Profiling mode: 'kv' for KV-cache, 'nocache' for no-cache generation.")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    set_seed(0)
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    model.config.use_cache = True

    print(f"Profiling {args.max_new_tokens} tokens on CPU ({args.num_threads} threads) with mode='{args.mode}'…")

    # --- Prefill ---
    (pre_lin, pre_nonlin, pre_tot), input_tokens = measure_prefill(model, tokenizer, args.text)

    # --- Model info ---
    print_model_info(model, tokenizer, input_tokens)

    # --- Token-by-token ---
    use_cache = (args.mode == "kv")
    stats = profile_loop(model, tokenizer, args.text, args.max_new_tokens, use_cache=use_cache)

    print("\nToken\tLinear(ms)\tNonLinear(ms)\tTotal(ms)")
    for i, (lin, nonlin, tot) in enumerate(stats, start=1):
        print(f"{i:02d}\t{lin:8.3f}\t{nonlin:8.3f}\t{tot:8.3f}")

    last = stats[-1]
    print("\n=== Summary ===")
    print(f"Prefill: linear={pre_lin:.3f} ms, nonlinear={pre_nonlin:.3f} ms, total={pre_tot:.3f} ms")
    print(f"Last token ({args.mode}) -> linear={last[0]:.3f} ms, nonlinear={last[1]:.3f} ms, total={last[2]:.3f} ms")

if __name__ == "__main__":
    main()
