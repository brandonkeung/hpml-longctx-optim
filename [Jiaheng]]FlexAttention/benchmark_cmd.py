# benchmark.py

import subprocess
import json
import math
import re

from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------
# Config
# ---------------------------
MODEL_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
attn_label = "gpt-fast"

CONTEXTS = [512, 2048, 8192]   # token limits to test
N_SAMPLES = 1                    # how many Hotpot examples per ctx_len
CHECKPOINT_PATH = f"checkpoints/{MODEL_REPO}/model.pth"

# ---------------------------
# Percentile helper
# ---------------------------
def percentile(data, p):
    vals = [v for v in data if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return float("nan")
    vals.sort()
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)

# ---------------------------
# Context builder (like old make_context_text)
# ---------------------------

def format_prompt(question, context):
    return (
        "You are a QA system. Use the context to answer the question.\n"
        "Answer with a short phrase or single entity only. No punctuation. No explanation.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    
def make_context_text(sentences, ctx_len, tok):
    """
    sentences: list[list[str]] from HotpotQA ("context"]["sentences"])
    ctx_len:   max tokens allowed for context part
    tok:       tokenizer
    """
    text = ""
    for s in sentences:
        candidate = text + " ".join(s) + " "
        # token-length check
        if len(tok(candidate)["input_ids"]) > ctx_len:
            break
        text = candidate
    return text

# ---------------------------
# Load tokenizer + dataset
# ---------------------------
print(f"Loading tokenizer for {MODEL_REPO} ...")
tok = AutoTokenizer.from_pretrained(MODEL_REPO)

print(f"Loading HotpotQA validation[:{N_SAMPLES}] ...")
ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{N_SAMPLES}]")

examples = []  # each: (question, sentences, gold_answers)
for ex in ds:
    sentences = ex["context"]["sentences"]
    q = ex["question"]
    golds = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
    examples.append((q, sentences, golds))
print(examples[0])
# ---------------------------
# Main loop over contexts
# ---------------------------
all_summaries = []

for ctx_len in CONTEXTS:
    print(f"\n===== Running ctx_len={ctx_len} =====")
    per_req = []  # list of dicts: one per example

    for i, (q, sentences, golds) in enumerate(examples):
        # 1) build context-limited text
        context_text = make_context_text(sentences, ctx_len, tok)

        # 2) raw prompt (you can change format_prompt style if you want)
        prompt_raw = format_prompt(q, context_text)

        # 3) enforce total prompt token length <= ctx_len (like old pipeline)
        encoded = tok(prompt_raw, truncation=True, max_length=ctx_len)
        prompt_tokens = encoded["input_ids"]
        prompt = tok.decode(prompt_tokens, skip_special_tokens=True)

        # 4) call generate.py
        cmd = [
            "python", "generate.py",
            "--compile",
            "--checkpoint_path", CHECKPOINT_PATH,
            "--prompt", prompt,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        # merge stdout + stderr for robust parsing + echo logs
        raw = ""
        if proc.stdout:
            raw += proc.stdout
        if proc.stderr:
            raw += proc.stderr

        print(f"\n--- Example {i} (ctx_len={ctx_len}) ---")
        print(raw)

        out_lines = raw.splitlines()

        lat = None
        tokps = None
        mem = None

        for line in out_lines:
            # e.g. "Time for inference 3: 4.26 sec total, 46.92 tokens/sec"
            if "Time for inference" in line:
                m1 = re.search(r"([0-9.]+)\s*sec total", line)
                m2 = re.search(r"([0-9.]+)\s*tokens/sec", line)
                if m1:
                    lat = float(m1.group(1)) * 1000.0  # sec -> ms
                if m2:
                    tokps = float(m2.group(1))

            # e.g. "Memory used: 14.55 GB"
            elif "Memory used:" in line:
                m = re.search(r"([0-9.]+)\s*GB", line)
                if m:
                    mem = float(m.group(1))

        if lat is None or tokps is None or mem is None:
            print(f"[WARN] Missing metrics for example {i} at ctx_len={ctx_len}: "
                  f"lat={lat}, tokps={tokps}, mem={mem}")
            continue

        per_req.append({
            "latency_ms": lat,
            "tokens_sec": tokps,
            "peak_mem": mem,
        })

    if not per_req:
        print(f"[WARN] No successful runs for ctx_len={ctx_len}, skipping summary.")
        continue

    # ---------------------------
    # Compute percentiles for this ctx_len
    # ---------------------------
    lat_p50 = round(percentile([r["latency_ms"] for r in per_req], 0.50), 2)
    lat_p95 = round(percentile([r["latency_ms"] for r in per_req], 0.95), 2)

    tok_p50 = round(percentile([r["tokens_sec"] for r in per_req], 0.50), 2)
    tok_p95 = round(percentile([r["tokens_sec"] for r in per_req], 0.95), 2)

    mem_p95 = round(percentile([r["peak_mem"] for r in per_req], 0.95), 2)

    run_id = f"{attn_label}_ctx{ctx_len}_N{len(per_req)}"

    ctx_summary = {
        "run_id": run_id,
        "model": MODEL_REPO,
        "attn": attn_label,
        "context_tokens": ctx_len,
        "n_requests": len(per_req),

        "latency_ms_p50": lat_p50,
        "latency_ms_p95": lat_p95,

        "ttft_ms_p50": None,
        "ttft_ms_p95": None,

        "ms_per_token_p50": None,
        "ms_per_token_p95": None,

        "tok_per_s_p50": tok_p50,
        "tok_per_s_p95": tok_p95,

        "decode_ms_p50": None,
        "decode_ms_p95": None,
        "ms_per_token_decode_p50": None,
        "ms_per_token_decode_p95": None,
        "tok_per_s_decode_p50": None,
        "tok_per_s_decode_p95": None,

        "peak_gpu_mem_gb_p95": mem_p95,

        "em_rate": None,  # you can fill later if you compute EM vs golds
    }

    all_summaries.append(ctx_summary)

# ---------------------------
# Save all results
# ---------------------------
with open("result_cmd.json", "w") as f:
    json.dump(all_summaries, f, indent=2)

print("\nDone. Summaries saved to result_cmd.json")