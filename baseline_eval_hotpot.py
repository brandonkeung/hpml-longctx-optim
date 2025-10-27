import os, time, json, math, random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))  # for req/s
CONTEXTS = [int(x) for x in os.environ.get("CONTEXTS", "512,2048,8192").split(",")]
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
LOGDIR = os.environ.get("LOGDIR", f"{os.environ.get('SCRATCH','.')}/logs")

def format_prompt(question, context):
    return f"Answer concisely.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

def make_context_text(sentences, target_tokens, tokenizer):
    # concatenate sentences until we exceed target_tokens; if too short, repeat
    ctx = []
    text = ""
    i = 0
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
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)

def main():
    os.makedirs(LOGDIR, exist_ok=True)
    random.seed(0)

    attn = "default" # or "flash_attention_2" or "default"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if attn == "flash_attention_2":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            use_cache=True,
            attn_implementation="flash_attention_2"
        )
    elif attn == "default":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            use_cache=True,
        )
    else:
        raise ValueError(f"Unknown attention implementation: {attn}")


    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{N_SAMPLES}]")

    all_rows = []
    oom_count = 0
    em_hits = 0

    # Preload examples
    examples = []
    for ex in ds:
        sentences = ex["context"]["sentences"]
        q = ex["question"]
        # HotpotQA "answer" can be str; normalize to list[str]
        golds = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
        examples.append((q, sentences, [g.lower() for g in golds]))

    for ctx_len in CONTEXTS:
        per_req_lat = []  # total latency per request (ms)
        per_tok_lat = []  # decode latency per generated token (ms/token)
        per_req_tokps = []  # tokens/sec
        per_req_peak = []  # GB

        # process in batches for req/s if BATCH_SIZE>1
        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i+BATCH_SIZE]
            prompts = []
            for (q, sentences, golds) in batch:
                context_text = make_context_text(sentences, ctx_len, tok)
                prompts.append(format_prompt(q, context_text))

            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=ctx_len).to(model.device)

            try:
                # ---- measure prefill+decode with CUDA events ----
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

                # times
                total_ms = start.elapsed_time(end)  # includes prefill + decode
                # Rough decode tokens = avg across batch
                gen_lens = []
                for bidx in range(len(batch)):
                    gen_len = int(out[bidx].shape[-1] - inputs["input_ids"][bidx].shape[-1])
                    gen_lens.append(gen_len)
                avg_gen = sum(gen_lens)/len(gen_lens) if gen_lens else 0

                # metrics
                per_req_lat.append(total_ms)
                per_tok_lat.append((total_ms / max(avg_gen, 1)))  # ms/token
                per_req_tokps.append((avg_gen / (total_ms/1000.0)))  # tok/s
                per_req_peak.append(torch.cuda.max_memory_allocated()/(1024**3))

                # rough EM on this batch
                texts = tok.batch_decode(out, skip_special_tokens=True)
                for j, (q, sentences, golds) in enumerate(batch):
                    ans = texts[j].split("Answer:")[-1].strip().lower()
                    if any(g in ans for g in golds):  # rough match
                        em_hits += 1

                # per-run row
                all_rows.append({
                    "model": MODEL_ID,
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
                    "model": MODEL_ID,
                    "context_tokens": ctx_len,
                    "batch_size": len(batch),
                    "error": str(e)[:160],
                    "success": False,
                    "oom": bool(is_oom),
                })
                torch.cuda.empty_cache()
                continue

        # After finishing this context length, print a context summary
        ctx_summary = {
            "model": MODEL_ID,
            "context_tokens": ctx_len,
            "n_requests": math.ceil(len(examples)/BATCH_SIZE),
            "latency_ms_p50": round(percentile(per_req_lat, 0.50), 2),          # Median end-to-end latency per request (prefill + decode)
            "latency_ms_p95": round(percentile(per_req_lat, 0.95), 2),          # “tail latency” — 95% of requests finished within
            "ms_per_token_p50": round(percentile(per_tok_lat, 0.50), 2),        # Median decoding speed: it took ~31 ms to generate one token
            "ms_per_token_p95": round(percentile(per_tok_lat, 0.95), 2),        # 95th percentile decoding speed
            "tok_per_s_p50": round(percentile(per_req_tokps, 0.50), 2),         # Equivalent throughput in tokens per second
            "tok_per_s_p95": round(percentile(per_req_tokps, 0.95), 2),         # 95th percentile tokens per second
            "peak_gpu_mem_gb_p95": round(percentile(per_req_peak, 0.95), 2),    # 95th percentile peak GPU memory usage
        }
        print(json.dumps(ctx_summary, indent=2))

    # Final overall summary
    summary = {
        "model": MODEL_ID,
        "total_samples": N_SAMPLES,
        "batch_size": BATCH_SIZE,
        "contexts": CONTEXTS,
        "oom_count": oom_count,
        "EM_rough": em_hits / N_SAMPLES
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(LOGDIR, "baseline_hotpot_extended.json"), "w") as f:
        json.dump({"rows": all_rows, "summary": summary}, f, indent=2)

if __name__ == "__main__":
    main()
