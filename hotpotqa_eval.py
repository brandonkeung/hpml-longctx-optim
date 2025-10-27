import os, time, json, math, random, datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Config (env-overridable)
# ---------------------------
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
CONTEXTS = [int(x) for x in os.environ.get("CONTEXTS", "512,2048,8192").split(",")]
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
ATTN_IMPL = os.environ.get("ATTN_IMPL", "eager").lower()  # "eager" | "flash2"
LOGDIR = os.environ.get("LOGDIR", "./logs")

# ---------------------------
# Helpers
# ---------------------------
def format_prompt(question, context):
    return f"Answer concisely.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

def make_context_text(sentences, target_tokens, tokenizer):
    text, i = "", 0
    while True:
        s = sentences[i % len(sentences)]
        text += (" " if text else "") + " ".join(s)
        if len(tokenizer.encode(text)) >= target_tokens:
            break
        i += 1
    return text

def percentile(values, p):
    if not values: return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return values[int(k)] if f == c else values[f] * (c - k) + values[c] * (k - f)

# ---------------------------
# Main
# ---------------------------
def main():
    # Logging setup
    os.makedirs(LOGDIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    attn_label = "flash2" if ATTN_IMPL in ("flash2", "flash_attention_2", "fa2") else "eager"
    model_name_tag = MODEL_ID.split("/")[-1]
    run_id = f"{model_name_tag}_{attn_label}_N{N_SAMPLES}_B{BATCH_SIZE}_{ts}"
    jsonl_path = os.path.join(LOGDIR, f"{run_id}.jsonl")         # context summaries (one JSON per line)
    full_json_path = os.path.join(LOGDIR, f"{run_id}_full.json") # EVERYTHING

    random.seed(0)

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model w/ attention toggle
    load_kwargs = dict(dtype=DTYPE, device_map=DEVICE_MAP, use_cache=True)
    if attn_label == "flash2":
        load_kwargs["attn_implementation"] = "flash_attention_2"
    print(f"[info] Loading {MODEL_ID} (attn={attn_label}, dtype={DTYPE}, device_map={DEVICE_MAP})")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

    print("Attention impl in config:", model.config._attn_implementation)
    print(model.model.layers[0].self_attn.forward.__qualname__)


    # Data
    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{N_SAMPLES}]")
    examples = []
    for ex in ds:
        sentences = ex["context"]["sentences"]
        q = ex["question"]
        golds = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
        examples.append((q, sentences, [g.lower() for g in golds]))

    # Metrics collectors
    all_rows = []
    ctx_summaries = []
    oom_count = 0
    em_hits = 0

    # Evaluation loop
    for ctx_len in CONTEXTS:
        per_req_lat, per_tok_lat, per_req_tokps, per_req_peak = [], [], [], []

        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i+BATCH_SIZE]
            prompts = []
            for (q, sentences, _golds) in batch:
                context_text = make_context_text(sentences, ctx_len, tok)
                prompts.append(format_prompt(q, context_text))

            inputs = tok(prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=ctx_len).to(model.device)

            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                end.record()
                torch.cuda.synchronize()

                total_ms = start.elapsed_time(end)
                gen_lens = []
                for bidx in range(len(batch)):
                    gen_len = int(out[bidx].shape[-1] - inputs["input_ids"][bidx].shape[-1])
                    gen_lens.append(gen_len)
                avg_gen = sum(gen_lens)/len(gen_lens) if gen_lens else 0

                per_req_lat.append(total_ms)
                per_tok_lat.append((total_ms / max(avg_gen, 1)))
                per_req_tokps.append((avg_gen / (total_ms/1000.0)))
                per_req_peak.append(torch.cuda.max_memory_allocated()/(1024**3))

                texts = tok.batch_decode(out, skip_special_tokens=True)
                for j, (_q, _sentences, golds) in enumerate(batch):
                    ans = texts[j].split("Answer:")[-1].strip().lower()
                    if any(g in ans for g in golds):
                        em_hits += 1

                all_rows.append({
                    "run_id": run_id,
                    "model": MODEL_ID,
                    "attn": attn_label,
                    "context_tokens": ctx_len,
                    "batch_size": len(batch),
                    "latency_ms_total": round(total_ms, 2),
                    "avg_gen_tokens": round(avg_gen, 2),
                    "tok_per_s": round(per_req_tokps[-1], 2),
                    "ms_per_token": round(per_tok_lat[-1], 2),
                    "peak_gpu_mem_gb": round(per_req_peak[-1], 2),
                    "success": True,
                })

            except RuntimeError as e:
                is_oom = "out of memory" in str(e).lower()
                oom_count += 1 if is_oom else 0
                all_rows.append({
                    "run_id": run_id,
                    "model": MODEL_ID,
                    "attn": attn_label,
                    "context_tokens": ctx_len,
                    "batch_size": len(batch),
                    "error": str(e)[:160],
                    "success": False,
                    "oom": bool(is_oom),
                })
                torch.cuda.empty_cache()
                continue

        # Per-context summary (print AND append to .jsonl)
        ctx_summary = {
            "run_id": run_id,
            "model": MODEL_ID,
            "attn": attn_label,
            "context_tokens": ctx_len,
            "n_requests": math.ceil(len(examples)/BATCH_SIZE),
            "latency_ms_p50": round(percentile(per_req_lat, 0.50), 2),
            "latency_ms_p95": round(percentile(per_req_lat, 0.95), 2),
            "ms_per_token_p50": round(percentile(per_tok_lat, 0.50), 2),
            "ms_per_token_p95": round(percentile(per_tok_lat, 0.95), 2),
            "tok_per_s_p50": round(percentile(per_req_tokps, 0.50), 2),
            "tok_per_s_p95": round(percentile(per_req_tokps, 0.95), 2),
            "peak_gpu_mem_gb_p95": round(percentile(per_req_peak, 0.95), 2),
        }
        print(json.dumps(ctx_summary, indent=2))
        ctx_summaries.append(ctx_summary)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(ctx_summary) + "\n")

    # Final overall summary (print AND write full artifact)
    summary = {
        "run_id": run_id,
        "model": MODEL_ID,
        "attn": attn_label,
        "total_samples": N_SAMPLES,
        "batch_size": BATCH_SIZE,
        "contexts": CONTEXTS,
        "oom_count": oom_count,
        "EM_rough": em_hits / N_SAMPLES if N_SAMPLES else float("nan"),
        "timestamp": ts,
    }
    print(json.dumps(summary, indent=2))

    full = {
        "run_meta": {
            "run_id": run_id, "model": MODEL_ID, "attn": attn_label,
            "dtype": str(DTYPE), "device_map": DEVICE_MAP,
            "contexts": CONTEXTS, "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS,
            "timestamp": ts,
        },
        "ctx_summaries": ctx_summaries,
        "rows": all_rows,
        "summary": summary,
    }
    with open(full_json_path, "w") as f:
        json.dump(full, f, indent=2)

if __name__ == "__main__":
    main()
