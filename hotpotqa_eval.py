import os, time, json, math, random, datetime, re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

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
ATTN_IMPL = os.environ.get("ATTN_IMPL", "eager").lower()  # eager | sdpa | flash2
LOGDIR = os.environ.get("LOGDIR", "./logs")

# ---------------------------
# Helpers
# ---------------------------
def format_prompt(question, context):
    return (
        "You are a QA system. Use the context to answer the question.\n"
        "Answer with a short phrase or single entity only. No punctuation. No explanation.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def make_context_text(sentences, target_tokens, tokenizer):
    # concatenate full context once instead of repeating
    text = " ".join(" ".join(s) for s in sentences)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=True)
    return text


def percentile(values, p):
    if not values: return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return values[int(k)] if f == c else values[f] * (c - k) + values[c] * (k - f)

# --- EM (Exact Match) normalization for HotpotQA ---
def normalize_answer(s: str) -> str:
    def lower(text): return text.lower()
    def remove_punc(text): return re.sub(r"[^\w\s]", "", text)
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(pred: str, golds) -> bool:
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)

# --- TTFT (Time To First Token) measurer ---
class FirstTokenTimer(StoppingCriteria):
    def __init__(self, start_len: int):
        super().__init__()
        self.start_len = start_len
        self.first_token_ms = None
        self._armed = False
        self._t0 = torch.cuda.Event(enable_timing=True)
        self._t1 = torch.cuda.Event(enable_timing=True)

    def arm(self):
        torch.cuda.synchronize()
        self._t0.record()
        self._armed = True

    def __call__(self, input_ids, scores, **kwargs):
        if not self._armed or self.first_token_ms is not None:
            return False
        # When decoded length exceeds prefill length, first token is out
        if input_ids.shape[1] > self.start_len:
            torch.cuda.synchronize()
            self._t1.record()
            torch.cuda.synchronize()
            self.first_token_ms = self._t0.elapsed_time(self._t1)
        return False  # never actually stop

# ---------------------------
# Main
# ---------------------------
def main():
    # Logging setup
    os.makedirs(LOGDIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name_tag = MODEL_ID.split("/")[-1]

    # Attention impl select
    load_kwargs = dict(dtype=DTYPE, device_map=DEVICE_MAP, use_cache=True)
    if ATTN_IMPL in ("flash2", "flash_attention_2", "fa2"):
        attn_label = "flash2"
        load_kwargs["attn_implementation"] = "flash_attention_2"
    elif ATTN_IMPL == "sdpa":
        attn_label = "sdpa"
        # Prefer native flash kernel if possible
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        load_kwargs["attn_implementation"] = "sdpa"
    elif ATTN_IMPL == "eager":
        attn_label = "eager"
        load_kwargs["attn_implementation"] = "eager"
    else:
        raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

    run_id = f"{model_name_tag}_{attn_label}_N{N_SAMPLES}_B{BATCH_SIZE}_{ts}"
    jsonl_path = os.path.join(LOGDIR, f"{run_id}.jsonl")         # context summaries (one JSON per line)
    full_json_path = os.path.join(LOGDIR, f"{run_id}_full.json") # EVERYTHING

    random.seed(0)

    # Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
        examples.append((q, sentences, golds))

    # Collectors
    all_rows = []
    ctx_summaries = []
    oom_count = 0
    em_hits = 0

    # Eval
    for ctx_len in CONTEXTS:
        per_req_lat, per_tok_lat, per_req_tokps, per_req_peak = [], [], [], []
        per_req_ttft, per_req_em = [], []
        per_req_decode_ms, per_req_mspt_decode, per_req_tokps_decode = [], [], []


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

                # TTFT setup
                start_len = inputs["input_ids"].shape[1]
                ftt = FirstTokenTimer(start_len)
                stoppers = StoppingCriteriaList([ftt])
                ftt.arm()

                start.record()
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=tok.eos_token_id,
                        stopping_criteria=stoppers,  # capture TTFT
                        return_dict_in_generate=False,
                        output_scores=False,
                    )
                end.record()
                torch.cuda.synchronize()

                total_ms = start.elapsed_time(end)
                ttft_ms = ftt.first_token_ms or float("nan")

                # gen lens
                gen_lens = []
                for bidx in range(len(batch)):
                    gen_len = int(out[bidx].shape[-1] - inputs["input_ids"][bidx].shape[-1])
                    gen_lens.append(gen_len)
                avg_gen = sum(gen_lens)/len(gen_lens) if gen_lens else 0

                decode_ms = max(total_ms - ttft_ms, 0.0)
                ms_per_token_decode = decode_ms / max(avg_gen, 1)
                tok_per_s_decode = (avg_gen / (decode_ms/1000.0)) if decode_ms > 0 else float("nan")

                per_req_decode_ms.append(decode_ms)
                per_req_mspt_decode.append(ms_per_token_decode if math.isfinite(ms_per_token_decode) else float("nan"))
                per_req_tokps_decode.append(tok_per_s_decode if math.isfinite(tok_per_s_decode) else float("nan"))


                # metrics
                per_req_lat.append(total_ms)
                per_req_ttft.append(ttft_ms)
                per_tok_lat.append(total_ms / max(avg_gen, 1))
                per_req_tokps.append(avg_gen / (total_ms/1000.0))
                per_req_peak.append(torch.cuda.max_memory_allocated()/(1024**3))

                texts = tok.batch_decode(out, skip_special_tokens=True)
                for j, (_q, _sentences, golds) in enumerate(batch):
                    # Assume format "... Answer: <pred>"
                    pred = texts[j].split("Answer:")[-1].strip()
                    is_em = 1 if exact_match(pred, golds) else 0
                    per_req_em.append(is_em)
                    em_hits += is_em

                # per-request row (aggregate for the batch)
                all_rows.append({
                    "run_id": run_id,
                    "model": MODEL_ID,
                    "attn": attn_label,
                    "context_tokens": ctx_len,
                    "batch_size": len(batch),
                    "latency_ms_total": round(total_ms, 2),
                    "ttft_ms": round(ttft_ms, 2) if isinstance(ttft_ms, float) else ttft_ms,
                    "avg_gen_tokens": round(avg_gen, 2),
                    "tok_per_s": round(per_req_tokps[-1], 2),
                    "ms_per_token": round(per_tok_lat[-1], 2),
                    "peak_gpu_mem_gb": round(per_req_peak[-1], 2),
                    "em_hits_in_batch": int(em_hits),
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

        # Context summary
        ctx_summary = {
            "run_id": run_id,
            "model": MODEL_ID,
            "attn": attn_label,
            "context_tokens": ctx_len,
            "n_requests": math.ceil(len(examples)/BATCH_SIZE),
            "latency_ms_p50": round(percentile(per_req_lat, 0.50), 2),
            "latency_ms_p95": round(percentile(per_req_lat, 0.95), 2),
            "ttft_ms_p50": round(percentile(per_req_ttft, 0.50), 2),
            "ttft_ms_p95": round(percentile(per_req_ttft, 0.95), 2),
            "ms_per_token_p50": round(percentile(per_tok_lat, 0.50), 2),
            "ms_per_token_p95": round(percentile(per_tok_lat, 0.95), 2),
            "tok_per_s_p50": round(percentile(per_req_tokps, 0.50), 2),
            "tok_per_s_p95": round(percentile(per_req_tokps, 0.95), 2),
            "decode_ms_p50": round(percentile(per_req_decode_ms, 0.50), 2),
            "decode_ms_p95": round(percentile(per_req_decode_ms, 0.95), 2),
            "ms_per_token_decode_p50": round(percentile(per_req_mspt_decode, 0.50), 2),
            "ms_per_token_decode_p95": round(percentile(per_req_mspt_decode, 0.95), 2),
            "tok_per_s_decode_p50": round(percentile(per_req_tokps_decode, 0.50), 2),
            "tok_per_s_decode_p95": round(percentile(per_req_tokps_decode, 0.95), 2),
            "peak_gpu_mem_gb_p95": round(percentile(per_req_peak, 0.95), 2),
            "em_rate": round(sum(per_req_em) / max(len(per_req_em), 1), 4),
        }
        print(json.dumps(ctx_summary, indent=2))
        ctx_summaries.append(ctx_summary)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(ctx_summary) + "\n")

        # --- DEBUG: print a few example predictions for manual inspection ---
        print("\n=== Sample outputs for context length", ctx_len, "===")
        for j in range(min(3, len(examples))):  # print up to 3 examples
            q, sentences, golds = examples[j]
            context_text = make_context_text(sentences, ctx_len, tok)
            prompt = format_prompt(q, context_text)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            pred_text = tok.decode(out[0], skip_special_tokens=True)
            pred = pred_text.split("Answer:", 1)[-1].strip().split("\n", 1)[0].split(".")[0].strip()

            print(f"Q: {q}")
            print(f"Pred: {pred}")
            print(f"Gold: {golds}\n")

        print("=====================================================\n")

    # Final overall summary
    summary = {
        "run_id": run_id,
        "model": MODEL_ID,
        "attn": attn_label,
        "total_samples": N_SAMPLES,
        "batch_size": BATCH_SIZE,
        "contexts": CONTEXTS,
        "oom_count": oom_count,
        "EM": em_hits / N_SAMPLES if N_SAMPLES else float("nan"),
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
