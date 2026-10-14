#!/usr/bin/env python3
"""
analyze_W_counter.py

Load and analyze a W-distribution counter dumped by the C++ simulation
(compiled with -DENABLE_W_DUMP=1).

Usage:
    python3 analyze_W_counter.py W_counter.bin [--grid 0.005]

Binary format (little-endian, packed):
    int32 dim_k
    int32 n_unique
    repeat n_unique times:
        int16 key[dim_k]    (quantized W, key[i] = round(W[i] / grid_step))
        uint32 count

Output:
    - Summary statistics: total accesses, unique keys, hit-rate upper bound
    - Σ W quantiles (to help decide the sum-threshold cache filter)
    - max W quantiles
    - Top-K coverage (Zipfian-ness diagnostic)
"""

import sys
import argparse
import numpy as np


def load_counter(path: str):
    """Returns (dim_k, keys_int16[n,dim_k], counts_uint32[n])."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
        dim_k = int(header[0])
        n     = int(header[1])
        if n == 0:
            return dim_k, np.zeros((0, dim_k), dtype=np.int16), \
                          np.zeros(0, dtype=np.uint32)
        # each record: dim_k * int16 + uint32
        rec_dtype = np.dtype([("key",   np.int16, (dim_k,)),
                              ("count", np.uint32)])
        raw = np.frombuffer(f.read(), dtype=rec_dtype, count=n)
        keys   = raw["key"]
        counts = raw["count"]
    return dim_k, keys, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="W_counter.bin path")
    ap.add_argument("--grid", type=float, default=0.005,
                    help="Grid step used when dumping (default 0.005). "
                         "Needed to recover real-valued W from int16 keys.")
    args = ap.parse_args()

    dim_k, keys, counts = load_counter(args.path)
    n_unique = len(counts)
    total    = int(counts.sum()) if n_unique > 0 else 0

    print(f"File:          {args.path}")
    print(f"DIM_K:         {dim_k}")
    print(f"Unique keys:   {n_unique:,}")
    print(f"Total bumps:   {total:,}")
    if total == 0:
        print("Empty counter, nothing to analyze.")
        return

    # Hit-rate upper bound: with infinite cache, any access after the first
    # on the same key is a hit -> (total - unique) / total.
    hit_upper = (total - n_unique) / total * 100.0
    print(f"Hit-rate ceiling (infinite cache): {hit_upper:.2f}%")
    print()

    # Recover real-valued W for quantile analysis
    W = keys.astype(np.float64) * args.grid  # shape (n_unique, dim_k)

    # Weighted quantiles (accesses, not unique keys)
    def weighted_quantile(values, weights, qs):
        # values: 1d, weights: 1d same length, qs: list of quantiles in [0,1]
        order = np.argsort(values)
        v = values[order]
        w = weights[order]
        cum = np.cumsum(w)
        cum = cum / cum[-1]
        return np.interp(qs, cum, v)

    sums = W.sum(axis=1)   # per unique key
    maxs = W.max(axis=1)
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    print("Σ W[i]  quantiles (access-weighted):")
    print("  " + "  ".join(f"p{int(q*100):02d}={v:7.3f}"
                            for q, v in zip(qs, weighted_quantile(sums, counts, qs))))
    print("max W[i] quantiles (access-weighted):")
    print("  " + "  ".join(f"p{int(q*100):02d}={v:7.3f}"
                            for q, v in zip(qs, weighted_quantile(maxs, counts, qs))))
    print()

    # Top-K coverage
    order = np.argsort(counts)[::-1]
    sorted_counts = counts[order].astype(np.int64)
    cum = np.cumsum(sorted_counts)
    print("Top-K access coverage:")
    for k in [100, 1000, 10000, 100000, 500000, 1000000]:
        if k <= n_unique:
            cov = cum[k - 1] / total * 100.0
            print(f"  top-{k:>8,} ({k/n_unique*100:5.1f}% of unique): "
                  f"covers {cov:6.2f}% of accesses")
    # Top-1% and top-10% shorthand
    t1  = max(1, n_unique // 100)
    t10 = max(1, n_unique // 10)
    print(f"  top 1% ({t1:,} keys): covers {cum[t1-1]/total*100:.2f}%")
    print(f"  top 10% ({t10:,} keys): covers {cum[t10-1]/total*100:.2f}%")
    print()

    # Access-count histogram (rough Zipfian check)
    print("Access-count buckets (how many unique keys fall in each bucket):")
    bins  = [1, 2, 3, 5, 10, 50, 100, 1000, 10000, int(sorted_counts.max()) + 1]
    edges = sorted(set(bins))
    hist, _ = np.histogram(sorted_counts, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        pct = c / n_unique * 100
        print(f"  [{lo:>6}, {hi:>6}):  {c:>10,} keys  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
