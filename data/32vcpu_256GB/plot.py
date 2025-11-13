#!/usr/bin/env python3
"""
Interpolate attention/FFN latencies and either plot or print metrics.

Usage examples:

  # 1) Plot per-token latencies (Attention & FFN) for 1..1024
  python plot.py --mode plot --max-tokens 1024 --save --prefix ./out/curve

  # 2) Plot accumulated total time (Attention+FFN) from 1..N
  python plot.py --mode accumulated --max-tokens 1024 --save

  # 3) Print accumulated total time for N tokens (1..N)
  python plot.py --mode totals --tokens 128

  # 4) Print incremental time to generate T new tokens starting at input_len
  python plot.py --mode incremental --input-len 128 --tokens 128

  # 5) Max tokens you can generate within a deadline (minutes) starting at input_len
  python plot.py --mode budget --input-len 256 --deadline-min 60
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Measured data (edit here)
# =========================
IN_TOKENS = np.array([128, 192, 255, 320, 382, 449, 508], dtype=float)
ATTN_S    = np.array([85.359, 142.169, 188.923, 289.429, 377.099, 438.109, 520.918], dtype=float)
FFN_S     = np.array([61.210,  88.472, 113.598, 145.572, 136.105, 189.605, 196.038], dtype=float)

# ================
# Core utilities
# ================
def build_interpolator(x, y):
    """
    Piecewise-linear interpolation with linear extrapolation at both ends.
    Returns a callable f(n_tokens) -> latency (scalar or ndarray).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 2:
        raise ValueError("Need at least two points to interpolate/extrapolate.")

    m_left  = (y[1]  - y[0])   / (x[1]  - x[0])
    m_right = (y[-1] - y[-2])  / (x[-1] - x[-2])

    def f(n):
        xv = np.asarray(n, dtype=float)
        yv = np.empty_like(xv, dtype=float)

        left  = xv <= x[0]
        right = xv >= x[-1]
        mid   = (~left) & (~right)

        # Extrapolation
        yv[left]  = y[0]  + m_left  * (xv[left]  - x[0])
        yv[right] = y[-1] + m_right * (xv[right] - x[-1])

        # Interpolation on interior
        if np.any(mid):
            xm = xv[mid]
            j = np.searchsorted(x, xm, side="right") - 1
            j = np.clip(j, 0, len(x) - 2)
            t = (xm - x[j]) / (x[j + 1] - x[j])
            yv[mid] = y[j] + t * (y[j + 1] - y[j])

        return yv.item() if np.isscalar(n) else yv

    return f

# Build interpolators from measured data
ATTN_LAT = build_interpolator(IN_TOKENS, ATTN_S)
FFN_LAT  = build_interpolator(IN_TOKENS, FFN_S)

def per_token_latency(token_idx: int) -> float:
    """
    Interpolated per-token latency at exactly `token_idx`.
    """
    if token_idx < 1 or token_idx > 1024:
        raise ValueError("token_idx must be between 1 and 1024.")
    return float(ATTN_LAT(token_idx) + FFN_LAT(token_idx))


def total_latency(n_tokens: int, IN_TOKENS, ATTN_S, FFN_S) -> float:
    """
    Accumulated total latency (seconds) to generate `n_tokens`,
    summing per-token Attention + FFN latencies from 1..n using interpolation.
    """
    if not (1 <= n_tokens <= 1024):
        raise ValueError("n_tokens must be between 1 and 1024.")
    n = np.arange(1, n_tokens + 1, dtype=float)
    per_token = ATTN_LAT(n) + FFN_LAT(n)
    return float(np.trapezoid(per_token, n))  # NumPy ≥1.26

def incremental_latency(input_len: int, tokens: int, IN_TOKENS, ATTN_S, FFN_S) -> float:
    """
    Accumulated latency (seconds) for generating `tokens` new tokens
    starting from an existing `input_len` context.
    Example: (128, 128) integrates from 129 → 256 (inclusive).
    """
    if input_len < 1 or tokens < 1:
        raise ValueError("Both input_len and tokens must be >= 1.")
    end = input_len + tokens
    if end > 1024:
        raise ValueError("input_len + tokens must not exceed 1024.")
    n = np.arange(input_len + 1, end + 1, dtype=float)
    per_token = ATTN_LAT(n) + FFN_LAT(n)
    return float(np.trapezoid(per_token, n))

def max_output_tokens(input_len: int, deadline_minutes: float, IN_TOKENS, ATTN_S, FFN_S) -> int:
    """
    Maximum number of *new* tokens that can be generated within the time budget,
    starting from `input_len` context tokens.
    """
    if not (1 <= input_len < 1024):
        raise ValueError("input_len must be in [1, 1023].")

    deadline_sec = deadline_minutes * 60.0
    total = 0.0
    generated = 0
    n = input_len

    # Step per token (accurate and simple; can be optimized via binary search if needed)
    while n < 1024:
        step_latency = float(ATTN_LAT(n + 1) + FFN_LAT(n + 1))
        if total + step_latency > deadline_sec:
            break
        total += step_latency
        n += 1
        generated += 1

    return generated

# ================
# Plotting
# ================
def plot_latency_curves(IN_TOKENS, ATTN_S, FFN_S,
                        max_tokens: int = 1024,
                        accumulated: bool = False,
                        save: bool = False,
                        prefix: str = "./interpolated") -> None:
    """
    Plot either per-token latency (Attention & FFN) or accumulated total vs tokens.
    - If accumulated=False: plots Attention(n) and FFN(n) for n=1..max_tokens
    - If accumulated=True: plots sum_{k=1..n} [Attention(k)+FFN(k)]
    """
    xs = np.arange(1, max_tokens + 1, dtype=float)

    if accumulated:
        totals = np.array([total_latency(int(n), IN_TOKENS, ATTN_S, FFN_S) for n in xs], dtype=float)
        plt.figure(figsize=(9, 5.5))
        plt.plot(xs, totals, label="Accumulated (Attention + FFN)")
        plt.xlabel("Tokens generated (n)")
        plt.ylabel("Accumulated latency (seconds)")
        plt.title("Accumulated Generation Time vs Tokens (1 → n)")
        plt.legend()
        if save:
            np.savetxt(f"{prefix}_accumulated_total_{max_tokens}.csv",
                       np.column_stack([xs.astype(int), totals]),
                       delimiter=",", header="N,Accumulated_s", comments="")
            plt.savefig(f"{prefix}_accumulated_total_{max_tokens}.png", bbox_inches="tight")
    else:
        attn_q = ATTN_LAT(xs)
        ffn_q  = FFN_LAT(xs)
        plt.figure(figsize=(9, 5.5))
        plt.plot(xs, attn_q, label="Attention (s)")
        plt.plot(xs, ffn_q,  label="FFN (s)")
        plt.xlabel("Input tokens (N)")
        plt.ylabel("Latency (seconds)")
        plt.title(f"Interpolated Attention & FFN Latencies (1 → {max_tokens} tokens)")
        plt.legend()
        if save:
            np.savetxt(f"{prefix}_attention_ffn_{max_tokens}.csv",
                       np.column_stack([xs.astype(int), attn_q, ffn_q]),
                       delimiter=",", header="N,Attention_s,FFN_s", comments="")
            plt.savefig(f"{prefix}_attention_ffn_{max_tokens}.png", bbox_inches="tight")

    plt.show()

# ================
# CLI
# ================
def parse_args():
    ap = argparse.ArgumentParser(description="Interpolate & analyze Attention/FFN latencies.")
    ap.add_argument("--mode",
                    choices=["plot", "per_token", "accumulated", "totals", "incremental", "budget"],
                    default="plot",
                    help="Action to perform.")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="Upper x-axis/token bound (1..1024) for plotting modes.")
    ap.add_argument("--save", action="store_true",
                    help="Save CSV and PNG for plot modes.")
    ap.add_argument("--prefix", type=str, default="./interpolated",
                    help="Filename prefix for saved CSV/PNG.")
    ap.add_argument("--tokens", type=int, default=128,
                    help="For --mode totals: total tokens N; for --mode incremental: tokens to generate.")
    ap.add_argument("--input-len", type=int, default=128,
                    help="For --mode incremental/budget: starting context size.")
    ap.add_argument("--deadline-min", type=float, default=60.0,
                    help="For --mode budget: time budget in minutes.")
    ap.add_argument("--token_id", type=int, default=128,
                    help="For --mode per_token: token id.")
    return ap.parse_args()
