#!/usr/bin/env python3
import argparse
import time
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def timeit(fn, *args, **kwargs) -> Tuple[float, any]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return dt, out


def generate_naive_no_cache(
    model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, top_p: float, top_k: int
) -> str:
    """
    Naive autoregressive loop WITHOUT KV cache:
    - Each step re-feeds the entire sequence and recomputes attention from scratch.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attn = inputs["attention_mask"].to(model.device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            next_logits = out.logits[:, -1, :]
            if do_sample:
                next_token = torch.softmax(next_logits, dim=-1).topk(top_k).indices
                next_logits_filtered = next_logits.clone()
                # nucleus + top-k filtering (simple; good enough for benchmarking)
                probs = torch.softmax(next_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                filtered = probs.clone()
                filtered.scatter_(1, sorted_idx[mask], 0.0)
                next_id = torch.multinomial(filtered, num_samples=1)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            attn = torch.ones_like(input_ids, dtype=torch.long, device=model.device)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def generate_with_cache_manual(
    model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, top_p: float, top_k: int
) -> str:
    """
    Autoregressive loop WITH KV cache (manual):
    - First pass on full prompt with use_cache=True to get past_key_values
    - Subsequent steps feed only the last token + past_key_values
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.inference_mode():
        # Warm start to build initial cache from the whole prompt
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        cur_id = input_ids[:, -1:]

        for _ in range(max_new_tokens):
            out = model(input_ids=cur_id, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

            if do_sample:
                # simple top-k + top-p sampling
                probs = torch.softmax(logits, dim=-1)
                # top-k
                if top_k > 0:
                    topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                    probs_filtered = torch.zeros_like(probs).scatter_(1, topk_idx, topk_probs)
                    probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)
                else:
                    probs_filtered = probs

                # top-p (nucleus)
                sorted_probs, sorted_idx = torch.sort(probs_filtered, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                filtered = sorted_probs.masked_fill(mask, 0.0)
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)
                choice_sorted = torch.multinomial(filtered, num_samples=1)
                next_id = sorted_idx.gather(1, choice_sorted)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            cur_id = next_id  # only feed the last token next step

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def generate_with_cache_generate_api(
    model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, top_p: float, top_k: int
) -> str:
    """
    Using HF .generate(), which uses KV caching internally for supported models.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    with torch.inference_mode():
        out = model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser("Benchmark GPT-2 small on CPU with/without KV cache")
    parser.add_argument("--text", type=str, default="Dear GPT, I would like you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something.",
                        help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="New tokens to generate for each method.")
    parser.add_argument("--num-threads", type=int, default=32,
                        help="Number of CPU threads for PyTorch.")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HF model id (default: gpt2 for GPT-2 small).")
    parser.add_argument("--do-sample", action="store_true",
                        help="Use sampling instead of greedy.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p for nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k for sampling.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    # Threads & device
    torch.set_num_threads(max(1, args.num_threads))
    try:
        torch.set_num_interop_threads(max(1, min(args.num_threads, 8)))
    except Exception:
        pass
    device = torch.device("cpu")

    # Reproducibility for sampling
    set_seed(args.seed)

    print(f"Loading model '{args.model}' on CPU…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.config.use_cache = True  # allow caching when requested

    prompt = args.text
    print(f"\nPrompt: {prompt!r}")
    print(f"max_new_tokens={args.max_new_tokens}, do_sample={args.do_sample}, "
          f"top_p={args.top_p}, top_k={args.top_k}, threads={args.num_threads}")

    # Warmup (helps JIT & caches)
    _ = tokenizer("Warmup run.", return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = model(**_)

    # ---- 1) Naive (no cache) ----
    dt1, text1 = timeit(
        generate_naive_no_cache, model, tokenizer, prompt, args.max_new_tokens, args.do_sample, args.top_p, args.top_k
    )
    tok_s1 = args.max_new_tokens / dt1
    print("\n[1] NAIVE (no KV cache)")
    print(f"Time: {dt1:.3f}s  |  Tokens/sec: {tok_s1:.2f}")
    print(f"{text1}")

    # ---- 2) Cached (manual) ----
    dt2, text2 = timeit(
        generate_with_cache_manual, model, tokenizer, prompt, args.max_new_tokens, args.do_sample, args.top_p, args.top_k
    )
    tok_s2 = args.max_new_tokens / dt2
    print("\n[2] CACHED (manual past_key_values)")
    print(f"Time: {dt2:.3f}s  |  Tokens/sec: {tok_s2:.2f}  |  Speedup vs naive: {dt1/dt2:.2f}×")
    print(f"{text2}")



    # ---- 3) Cached via generate() ----
    dt3, text3 = timeit(
        generate_with_cache_generate_api, model, tokenizer, prompt, args.max_new_tokens, args.do_sample, args.top_p, args.top_k
    )
    tok_s3 = args.max_new_tokens / dt3
    print("\n[3] CACHED via model.generate()")
    print(f"Time: {dt3:.3f}s  |  Tokens/sec: {tok_s3:.2f}  |  Speedup vs naive: {dt1/dt3:.2f}×")
    print(f"{text3}")


    print("\nDone.")


if __name__ == "__main__":
    main()
