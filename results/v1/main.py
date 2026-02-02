import re
from collections import defaultdict
from datetime import datetime, time
from typing import Optional, Dict, Any
from collections import defaultdict

from plots import plot_grouped_stacked

def summarize_components(d):
    """
    Given a dict with breakdowns, compute total embedding, attention, and FNN.
    """
    # embedding comes directly
    embedding = d.get("embedding", 0.0)

    # attention components
    attention_keys = ["Project QKV", "QK'", "Softmax", "softmax(QK'V)", "Output combination"]
    attention = sum(d.get(k, 0.0) for k in attention_keys)

    # FNN components
    fnn_keys = ["FNN-MatMul1", "GeLU", "FNN-MatMul2"]
    fnn = sum(d.get(k, 0.0) for k in fnn_keys)

    return {"embedding": embedding, "attention": attention, "fnn": fnn}

def parse_token_embedding_latency(filename: str) -> Dict[str, Any]:
    """
    Compute token-embedding latency from logs.

    We detect lines like:
      [..timestamp..] [info] [cheetah_dot.cc:475] 1@1x50257x768 => 1x8192x1 ... (pack_lwes)

    Latency = (timestamp of last such line) - (timestamp of first such line), in milliseconds.

    Returns:
        {
          "count": int,                 # number of token-embedding lines found
          "start_ts": str | None,       # first timestamp (ISO)
          "end_ts": str | None,         # last timestamp (ISO)
          "latency_ms": float | None    # elapsed time in ms (None if <2 lines)
        }
    """

    # Pattern for token-embedding lines (pack_lwes path)
    embed_re = re.compile(
        r"\[.*?\]\s+\[info\]\s+\[cheetah_dot\.cc:475\]\s+\d+@1x50257x768\s*=>\s*1x8192x1.*\(pack_lwes\)",
        re.IGNORECASE,
    )

    # Timestamps: prefer full [YYYY-mm-dd HH:MM:SS.mmm], else fall back to HH:MM:SS.mmm (handles minor glitches)
    full_ts_re = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")
    time_only_re = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d+)\]")

    def parse_ts(line: str) -> Optional[float]:
        """Return POSIX seconds (float). Falls back to seconds since midnight if date is missing."""
        m = full_ts_re.search(line)
        if m:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
        m = time_only_re.search(line)
        if m:
            # seconds since midnight; OK as long as all embedding lines are same day (true for your blocks)
            t = datetime.strptime(m.group(1), "%H:%M:%S.%f").time()
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
        return None

    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    first_iso: Optional[str] = None
    last_iso: Optional[str] = None
    count = 0

    with open(filename, "r") as f:
        for line in f:
            if not embed_re.search(line):
                continue
            ts_val = parse_ts(line)
            if ts_val is None:
                continue
            count += 1
            if first_ts is None:
                first_ts = ts_val
                # Best-effort ISO string (use full if present; else time-of-day)
                m = full_ts_re.search(line)
                first_iso = m.group(1) if m else time_only_re.search(line).group(1)
            # Always update last to this match
            last_ts = ts_val
            m = full_ts_re.search(line)
            last_iso = m.group(1) if m else (time_only_re.search(line).group(1) if time_only_re.search(line) else None)

    if count < 1:
        return {"count": 0, "start_ts": None, "end_ts": None, "latency_ms": None}
    if count == 1 or first_ts is None or last_ts is None:
        return {"count": count, "start_ts": first_iso, "end_ts": last_iso, "latency_ms": 0.0}

    # If we mixed absolute and time-only seconds, the subtraction still works as long as they’re same day.
    latency_ms = (last_ts - first_ts) * 1000.0
    return {"count": count, "start_ts": first_iso, "end_ts": last_iso, "latency_ms": latency_ms}



def sum_layer_timings(layer_data: dict[int, dict[str, float]]) -> dict[str, float]:
    """
    Aggregate layer-wise timings into total per-kernel timings.

    Args:
        layer_data: dict[layer_id -> dict[kernel -> latency_ms]]

    Returns:
        dict[kernel -> total_latency_ms] across all layers
    """
    totals = defaultdict(float)

    for _, timings in layer_data.items():
        for kernel, value in timings.items():
            totals[kernel] += value

    return dict(totals)


def parse_layer_timing(filename: str):
    """
    Parse GPT-2 per-layer kernel timings from logs using timestamps only.
    Returns: dict[layer_id] = { kernel_name: latency_ms }
    - A kernel's latency = start_of_next_kernel - start_of_this_kernel.
    - The last kernel of each layer is closed using the timestamp of the
      first kernel of the next layer (final layer falls back to 0 ms).
    """

    layer_hdr_re = re.compile(r"projecting for layer (\d+)")
    ts_re        = re.compile(r"^\[([0-9:\-\. ]+)\]")
    shape_re     = re.compile(r"@(\d+)x(\d+)x(\d+)\s*=>\s*(\d+)x(\d+)x(\d+)")

    def parse_ts(line: str) -> Optional[float]:
        m = ts_re.match(line)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
        except ValueError:
            return None

    def identify_kernel(line: str) -> Optional[str]:
        # Prefer classification from the LHS matmul shape ONLY.
        m = shape_re.search(line)
        if m:
            A, B, C, D, E, F = map(int, m.groups())  # LHS: A x B x C, RHS: D x E x F

            # Known dense matmuls
            if B == 768 and C == 2304:
                return "Project QKV"
            if B == 768 and C == 768:
                return "Output combination"
            if B == 768 and C == 3072:
                return "FNN-MatMul1"
            if B == 3072 and C == 768:
                return "FNN-MatMul2"

            # Attention matmuls (sequence-length agnostic)
            # QK′ : (tokens x head_dim x tokens)  -> A == C, and not one of the dense combos
            if A == C and B not in (768, 3072):
                return "QK'"

            # softmax(QK′V): (head_dim x tokens x tokens) -> B == C, with small-ish head dim A
            # Avoid clashes with dense combos by excluding large dims.
            if B == C and A not in (768, 3072):
                return "softmax(QK'V)"

        # Activations (no matmul shape)
        if "f_nexp" in line:
            return "Softmax"
        if "gelu" in line:
            return "GeLU"
        return None

    result = defaultdict(dict)
    current_layer: Optional[int] = None
    pending_layer: Optional[int] = None

    active_kernel: Optional[str] = None
    kernel_start_ts: Optional[float] = None

    def close_kernel(close_ts: float, kernel: Optional[str], start_ts: Optional[float], layer: Optional[int]):
        if kernel and start_ts is not None and layer is not None:
            latency = (close_ts - start_ts) * 1000.0
            if kernel not in result[layer]:
                result[layer][kernel] = latency

    with open(filename, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            # Catch layer headers (can appear twice)
            m_hdr = layer_hdr_re.search(line)
            if m_hdr:
                pending_layer = int(m_hdr.group(1))
                continue

            # Identify kernel + timestamp
            kernel = identify_kernel(line)
            if kernel is None:
                continue
            ts = parse_ts(line)
            if ts is None:
                continue

            # If a new layer was announced, close the previous layer's last kernel at this ts
            if pending_layer is not None:
                close_kernel(ts, active_kernel, kernel_start_ts, current_layer)
                current_layer = pending_layer
                pending_layer = None
                active_kernel = None
                kernel_start_ts = None

            # First kernel in file/layer
            if active_kernel is None:
                active_kernel = kernel
                kernel_start_ts = ts
                continue

            # Duplicate line for the same kernel → ignore
            if kernel == active_kernel:
                continue

            # Switch: close previous, start new
            close_kernel(ts, active_kernel, kernel_start_ts, current_layer)
            active_kernel = kernel
            kernel_start_ts = ts

    # EOF: close last kernel of final layer with 0 ms (no next timestamp available)
    if active_kernel and kernel_start_ts is not None and current_layer is not None:
        if active_kernel not in result[current_layer]:
            result[current_layer][active_kernel] = 0.0

    return dict(result)



# Example usage
if __name__ == "__main__":
    _128_logs = "logs_128tokens.txt"
    _256_logs = "logs_256tokens.txt"
    _512_logs = "logs_512tokens.txt"
    
    _128_layer_Details = parse_layer_timing(_128_logs)
    _256_layer_Details = parse_layer_timing(_256_logs)
    _512_layer_Details = parse_layer_timing(_512_logs)
    _512_layer_Details[12]["FNN-MatMul2"] = 130*1000 # manually fix the missing party
    
    # print details of each layer
    # print("\n\n\n128 Layers")
    # for layer, timings in _128_layer_Details.items():
    #     print(f"Layer {layer}:")
    #     for k, v in timings.items():
    #         print(f"  {k}: {v}")
    # print("\n\n\n256 Layers")
    # for layer, timings in _256_layer_Details.items():
    #     print(f"Layer {layer}:")
    #     for k, v in timings.items():
    #         print(f"  {k}: {v}")
    # print("\n\n\n512 Layers")
    # for layer, timings in _512_layer_Details.items():
    #     print(f"Layer {layer}:")
    #     for k, v in timings.items():
    #         print(f"  {k}: {v}")

    
    for _n_in, data in zip([128, 256, 512], [_128_layer_Details, _256_layer_Details,  _512_layer_Details]):
        print(f"\n\n{_n_in} input tokens")
        for layer, timings in data.items():
            if(layer != 1):
                continue

            print(f"Layer {layer}:")
            for k, v in timings.items():
                print(f"  {k}: {v}")
                
    print("\n\n\nTotals\n===================")
    _128_layer_totals = sum_layer_timings(_128_layer_Details)
    _256_layer_totals = sum_layer_timings(_256_layer_Details)
    _512_layer_totals = sum_layer_timings(_512_layer_Details)
    _128_tokens = parse_token_embedding_latency(_128_logs)
    _256_tokens = parse_token_embedding_latency(_256_logs)
    _512_tokens = parse_token_embedding_latency(_512_logs)
    
    for _n_in, timings, embedding_latency in zip([128, 256, 512], [_128_layer_totals, _256_layer_totals,  _512_layer_totals], [_128_tokens, _256_tokens, _512_tokens]):
        print(f"\n\n{_n_in} input tokens totals")
        _total = 0.0
        for k, v in timings.items():
            print(f"  {k}: {v*0.001:.2f}")
            _total += v

        timings["total layers"] = _total
        timings["embedding"] = embedding_latency['latency_ms']
        timings["total"] = _total + embedding_latency['latency_ms']

        print(f"total layers: {timings['total layers'] * 0.001:.2f}")
        print(f"embedding: {timings['embedding'] * 0.001:.2f}")
        print(f"total: {timings['total'] * 0.001:.2f}")
        
    for _n_in, timings in  zip([128, 256, 512], [_128_layer_totals, _256_layer_totals,  _512_layer_totals]):
        print(f"\n\n{_n_in} summerized")
        print(summarize_components(timings))



    # motivation plot
    _128_original = summarize_components(_128_layer_totals)   
    _256_original = summarize_components(_256_layer_totals)   
    _512_original = summarize_components(_512_layer_totals)  
    _128_ideal = {}
    _256_ideal = {}
    _512_ideal = {}
    
    for k,v in _128_original.items():
        _128_ideal[k] = v / 128

    for k,v in _128_original.items():
        _256_ideal[k] =v / 256
    for k,v in _512_original.items():
        _512_ideal[k] = v / 512

    group1 = [_128_original, _256_original, _512_original]
    group2 = [_128_ideal, _256_ideal, _512_ideal]
    # plot_grouped_stacked([group1, group2])