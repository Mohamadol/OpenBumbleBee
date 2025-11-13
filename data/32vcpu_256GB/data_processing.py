#!/usr/bin/env python3
import re
from datetime import datetime
from pathlib import Path

from plot import *

# ---------- Timestamp regex is constant ----------
TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")

def parse_timestamp(s, ts_re=TIMESTAMP_RE):
    m = ts_re.search(s)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None

def load_raw_lines(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()

def build_indices_and_times(lines):
    times, texts, last_ts = [], [], None
    for ln in lines:
        ts = parse_timestamp(ln)
        if ts is not None:
            last_ts = ts
        times.append(ts)
        texts.append(ln.strip())
    return times, texts, last_ts

def time_at_or_prev(i, times):
    n = len(times)
    if i < 0: i = 0
    if i >= n: i = n - 1
    if times[i] is not None:
        return times[i]
    j = i - 1
    while j >= 0:
        if times[j] is not None:
            return times[j]
        j -= 1
    j = i + 1
    while j < n:
        if times[j] is not None:
            return times[j]
        j += 1
    return None

def time_at_or_next(i, times):
    n = len(times)
    if i < 0: i = 0
    if i >= n: i = n - 1
    if times[i] is not None:
        return times[i]
    j = i + 1
    while j < n:
        if times[j] is not None:
            return times[j]
        j += 1
    j = i - 1
    while j >= 0:
        if times[j] is not None:
            return times[j]
        j -= 1
    return None

def contiguous_ranges(indices):
    if not indices:
        return
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            yield (start, prev)
            start = idx
            prev = idx
    yield (start, prev)

def compute_embedding_total(times, texts, embedding_pat):
    emb_idxs = [i for i, t in enumerate(texts) if embedding_pat.search(t)]
    total = 0.0
    for a, b in contiguous_ranges(emb_idxs):
        t0 = time_at_or_prev(a, times)
        t1 = time_at_or_prev(b, times)
        if t0 is None or t1 is None:
            continue
        total += (t1 - t0).total_seconds()
    return total

def dedup_projecting_indices(proj_raw_indices, texts, projecting_pat):
    deduped = []
    last_idx, last_layer = None, None
    for idx in proj_raw_indices:
        m = projecting_pat.search(texts[idx])
        layer = m.group(1) if m else None
        if last_idx is None or not (idx == last_idx + 1 and layer == last_layer):
            deduped.append(idx)
        last_idx, last_layer = idx, layer
    return deduped

def first_after(indices, base):
    for x in indices:
        if x > base:
            return x
    return None

def compute_attention_ffn(times, texts, last_ts,
                          projecting_pat, ffn_enter_pat, ffn_exit_pat, out_embed_pat):
    n = len(texts)
    proj_raw     = [i for i, txt in enumerate(texts) if projecting_pat.search(txt)]
    proj_idx     = dedup_projecting_indices(proj_raw, texts, projecting_pat)
    ffn_enter_ix = [i for i, txt in enumerate(texts) if ffn_enter_pat.search(txt)]
    ffn_exit_ix  = [i for i, txt in enumerate(texts) if ffn_exit_pat.search(txt)]
    out_ix       = [i for i, txt in enumerate(texts) if out_embed_pat.search(txt)]

    total_attn = 0.0
    total_ffn  = 0.0

    for k, p in enumerate(proj_idx):
        # ---- Attention: from one-before projecting -> one-before first 1@...x3072 after p
        attn_start_i = max(p - 1, 0)
        t_attn_start = time_at_or_prev(attn_start_i, times)
        if t_attn_start is None:
            continue

        f = first_after(ffn_enter_ix, p)
        if f is None:
            continue
        attn_end_i = max(f - 1, 0)
        t_attn_end = time_at_or_prev(attn_end_i, times)
        if t_attn_end is None:
            continue

        attn_sec = (t_attn_end - t_attn_start).total_seconds()
        if attn_sec > 0:
            total_attn += attn_sec

        # ---- FFN: from one-before 1@...x3072 -> first 1@...x3072x... (down-proj)
        ffn_start_i = max(f - 1, 0)
        t_ffn_start = time_at_or_prev(ffn_start_i, times)
        if t_ffn_start is None:
            continue

        fx = first_after(ffn_exit_ix, f)
        if fx is not None:
            t_ffn_end = time_at_or_next(fx, times)
        else:
            next_proj_i = proj_idx[k + 1] if (k + 1) < len(proj_idx) else n
            out_after_f = next((oi for oi in out_ix if f < oi < next_proj_i), None)
            if out_after_f is not None:
                t_ffn_end = time_at_or_prev(max(out_after_f - 1, 0), times)
            else:
                t_ffn_end = time_at_or_next(next_proj_i, times) if next_proj_i < n else last_ts

        if t_ffn_end is None:
            continue

        ffn_sec = (t_ffn_end - t_ffn_start).total_seconds()
        if ffn_sec > 0:
            total_ffn += ffn_sec

    return total_attn, total_ffn

def make_patterns(seq_len, d_model, d_ff, vocab):
    # Build all regexes for one configuration
    embedding_pat  = re.compile(fr"1@1x{vocab}x{d_model}")
    ffn_enter_pat  = re.compile(fr"1@{seq_len}x{d_model}x{d_ff}")
    ffn_exit_pat   = re.compile(fr"1@{seq_len}x{d_ff}x{d_model}")
    projecting_pat = re.compile(r"projecting for layer\s+(\d+)", re.IGNORECASE)
    out_embed_pat  = re.compile(fr"1@{seq_len}x{d_model}x{vocab}")
    return embedding_pat, ffn_enter_pat, ffn_exit_pat, projecting_pat, out_embed_pat

def get_data():

    IN_TOKENS = [128, 192, 255, 320, 382, 449, 508]
    FILE_NAMES = {128: 128, 192:192, 255:256, 320:320, 382:384, 449:448, 508:512}
    D_MODEL   = 768
    D_FF      = 3072
    VOCAB     = 50257

    ATTN_S    = []
    FFN_S     = []

    for in_token_len in IN_TOKENS:
        LOG_PATH = Path(f"server_{FILE_NAMES[in_token_len]}in_1out.txt")
        try:
            lines = load_raw_lines(LOG_PATH)
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {in_token_len}: {e}")
            continue  # keep going with other files

        times, texts, last_ts = build_indices_and_times(lines)
        if last_ts is None:
            print(f"⚠️ Skipping {in_token_len}: no timestamped lines found.")
            continue

        (embedding_pat,
         ffn_enter_pat,
         ffn_exit_pat,
         projecting_pat,
         out_embed_pat) = make_patterns(in_token_len, D_MODEL, D_FF, VOCAB)

        embedding_total = compute_embedding_total(times, texts, embedding_pat)
        attention_total, ffn_total = compute_attention_ffn(
            times, texts, last_ts,
            projecting_pat, ffn_enter_pat, ffn_exit_pat, out_embed_pat
        )

        ATTN_S.append(attention_total)
        FFN_S.append(ffn_total)

        # print(f"\nInput length: {in_token_len}")
        # print("== Totals (seconds) ==")
        # print(f"Total input tokens embedding: {embedding_total:.3f} s")
        # print(f"Total attention (all layers): {attention_total:.3f} s")
        # print(f"Total FFN (all layers):       {ffn_total:.3f} s")

    return IN_TOKENS, ATTN_S, FFN_S


def main():

    IN_TOKENS, ATTN_S, FFN_S = get_data()

    # print("\n\nAttention")
    # print(ATTN_S)

    # print("\n\nAttention")
    # print(FFN_S)

    args = parse_args()

    if args.mode in ("plot", "accumulated"):
        max_tok = int(np.clip(args.max_tokens, 1, 1024))
        plot_latency_curves(IN_TOKENS, ATTN_S, FFN_S,
                            max_tokens=max_tok,
                            accumulated=(args.mode == "accumulated"),
                            save=args.save,
                            prefix=args.prefix)
        
    elif args.mode == "per_token":
        n = int(np.clip(args.token_id, 1, 1024))
        token_l = per_token_latency(n)
        print(f"Latency of token {n}: {token_l} seconds")

    elif args.mode == "totals":
        n = int(np.clip(args.tokens, 1, 1024))
        tot_s = total_latency(n, IN_TOKENS, ATTN_S, FFN_S)
        print(f"Accumulated total latency (1→{n}): {tot_s:.2f} s  ({tot_s/3600:.2f} hr)")

    elif args.mode == "incremental":
        inc_s = incremental_latency(int(args.input_len), int(args.tokens), IN_TOKENS, ATTN_S, FFN_S)
        print(f"Incremental latency ({args.input_len}→{args.input_len + args.tokens}): {inc_s:.2f} s  ({inc_s/3600:.2f} hr)")

    elif args.mode == "budget":
        max_out = max_output_tokens(int(args.input_len), float(args.deadline_min), IN_TOKENS, ATTN_S, FFN_S)
        print(f"Max output tokens within {args.deadline_min} min (start={args.input_len}): {max_out}")

if __name__ == "__main__":
    main()

