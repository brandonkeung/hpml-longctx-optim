import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer


def _unwrap_text(x):
    """NarrativeQA sometimes stores text as a dict like {'text': '...'}."""
    if isinstance(x, dict):
        return (x.get("text", "") or "").strip()
    return (x or "").strip()


def extract_summary_and_fulltext(ex):
    doc = ex.get("document") or {}
    summary = _unwrap_text(doc.get("summary", ""))
    full_text = _unwrap_text(doc.get("text", ""))
    return summary, full_text


def stats(arr):
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def save_hist(data, title, outpath, bins=60):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel("Token count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--n_samples", type=int, default=2000,
                    help="How many examples to sample. Use -1 for ALL examples in the split.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./plots")
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_id)

    ds = load_dataset("narrativeqa", "default", split=args.split)
    n_total = len(ds)

    if args.n_samples == -1 or args.n_samples >= n_total:
        idxs = np.arange(n_total)
    else:
        idxs = np.random.choice(n_total, size=args.n_samples, replace=False)

    summary_ctx_lens = []
    full_ctx_lens = []
    kept_summary = 0
    kept_full = 0

    for idx in idxs:
        ex = ds[int(idx)]
        summary, full_text = extract_summary_and_fulltext(ex)

        if summary:
            summary_ids = tok.encode(summary, add_special_tokens=False)
            summary_ctx_lens.append(len(summary_ids))
            kept_summary += 1

        if full_text:
            full_ids = tok.encode(full_text, add_special_tokens=False)
            full_ctx_lens.append(len(full_ids))
            kept_full += 1

    out_summary = os.path.join(args.outdir, f"narrativeqa_{args.split}_SUMMARY_context_token_hist.png")
    out_full = os.path.join(args.outdir, f"narrativeqa_{args.split}_FULLTEXT_context_token_hist.png")

    save_hist(
        summary_ctx_lens,
        f"NarrativeQA {args.split} | SUMMARY context token counts (kept={kept_summary}, total_split={n_total})",
        out_summary,
        bins=args.bins
    )
    save_hist(
        full_ctx_lens,
        f"NarrativeQA {args.split} | FULL TEXT context token counts (kept={kept_full}, total_split={n_total})",
        out_full,
        bins=args.bins
    )

    print("\n=== Done ===")
    print("Saved:", out_summary)
    print("Saved:", out_full)
    print("\nSummary context token stats:", stats(summary_ctx_lens))
    print("Full-text context token stats:", stats(full_ctx_lens))


if __name__ == "__main__":
    main()
