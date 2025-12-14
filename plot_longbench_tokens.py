#!/usr/bin/env python3
"""
Histogram of TOKEN counts for LongBench-v2 using
mistralai/Mistral-7B-Instruct-v0.3 tokenizer.

Defaults:
- dataset: zai-org/LongBench-v2 (fallback THUDM/LongBench-v2)
- split: train
- field: context
- streaming: on
- saves: longbench_v2_<field>_mistral_token_hist.png

Requirements:
  pip install -U datasets transformers
  # (sometimes also needed) pip install -U sentencepiece
"""

import argparse
import math
import sys
from typing import List, Optional

def try_load_dataset(dataset_ids: List[str], split: str, streaming: bool):
    from datasets import load_dataset

    last_err = None
    for ds_id in dataset_ids:
        try:
            ds = load_dataset(ds_id, split=split, streaming=streaming)
            return ds_id, ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load any of {dataset_ids}. Last error: {last_err}")

def percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return int(round(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--field", default="context")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--linear", action="store_true", help="Use linear bins (otherwise log-spaced).")
    ap.add_argument("--no-streaming", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8, help="Tokenizer batch size (default: 8).")
    ap.add_argument("--tokenizer-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--use-fast", action="store_true", help="Force use_fast=True if available.")
    args = ap.parse_args()

    dataset_candidates = ["zai-org/LongBench-v2", "THUDM/LongBench-v2"]
    streaming = not args.no_streaming

    ds_id, ds = try_load_dataset(dataset_candidates, split=args.split, streaming=streaming)
    print(f"Loaded dataset: {ds_id} (split={args.split}, streaming={streaming})", file=sys.stderr)

    # ---- Load tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=args.use_fast)

    # Avoid accidental truncation/warnings for long sequences
    # (we want true token counts, even if > model context length)
    try:
        tok.model_max_length = 10**30
    except Exception:
        pass

    # ---- Iterate and count tokens
    counts: List[int] = []
    buf_texts: List[str] = []
    n = 0

    def flush_buffer():
        nonlocal buf_texts
        if not buf_texts:
            return
        enc = tok(
            buf_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # enc["input_ids"] is a list[list[int]]
        counts.extend([len(ids) for ids in enc["input_ids"]])
        buf_texts = []

    for ex in ds:
        if args.field not in ex:
            raise KeyError(f"Field '{args.field}' not found. Available keys: {list(ex.keys())}")
        text = ex[args.field] or ""
        buf_texts.append(text)
        n += 1

        if len(buf_texts) >= args.batch_size:
            flush_buffer()

        if args.max_examples is not None and n >= args.max_examples:
            break

    flush_buffer()

    if not counts:
        raise RuntimeError("No examples processed; nothing to plot.")

    counts_sorted = sorted(counts)
    out_path = args.out or f"longbench_v2_{args.field}_mistral_token_hist.png"

    # ---- Plot
    import numpy as np
    import matplotlib.pyplot as plt

    arr = np.array(counts, dtype=np.int64)
    mn, mx = int(arr.min()), int(arr.max())

    plt.figure()
    if args.linear or mn <= 0:
        plt.hist(arr, bins=args.bins)
        plt.xlabel(f"Token count in '{args.field}' (linear bins)")
    else:
        lo = max(1, mn)
        bins = np.logspace(np.log10(lo), np.log10(mx), args.bins)
        plt.hist(arr, bins=bins)
        plt.xscale("log")
        plt.xlabel(f"Token count in '{args.field}' (log scale)")

    plt.ylabel("Number of samples")
    plt.title(
        f"LongBench-v2 token counts ({args.field})\n"
        f"Tokenizer: {args.tokenizer_id}\n"
        f"N={len(arr)}  min={mn}  max={mx}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # ---- Summary stats
    print(f"\nSaved histogram -> {out_path}")
    print(f"N={len(counts)}")
    print(f"min={counts_sorted[0]}")
    print(f"p50={percentile(counts_sorted, 50)}")
    print(f"p90={percentile(counts_sorted, 90)}")
    print(f"p99={percentile(counts_sorted, 99)}")
    print(f"max={counts_sorted[-1]}")

if __name__ == "__main__":
    main()
