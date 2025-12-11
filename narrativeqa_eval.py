import os, time, json, math, random, datetime, re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from kv_compression.kv_l2_dynamic import generate_with_l2_compress
import evaluate
from transformers import BitsAndBytesConfig 
import wandb


# ---------------------------
# Config (env-overridable)
# ---------------------------
# MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")       # Qwen/Qwen1.5-1.8B-Chat
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen1.5-1.8B-Chat")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
CONTEXTS = [int(x) for x in os.environ.get("CONTEXTS", "2048,8192,16384").split(",")]
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
ATTN_IMPL = os.environ.get("ATTN_IMPL", "eager").lower()  # eager | sdpa | flash2
LOGDIR = os.environ.get("LOGDIR", "./logs/narrativeqa")
USE_4BIT = os.environ.get("USE_4BIT", "false").lower() == "true"
ENTITY = "bk2951-columbia-university"
# --- KV compression mode ---
#   none        -> vanilla generate()
#   l2          -> manual decode loop with L2-based pruning of KV values
KV_MODE = os.environ.get("KV_MODE", "none").lower()  # none | l2
# For KV_MODE=l2
KEEP_RATIO = float(os.environ.get("KEEP_RATIO", "0.7"))     # keep top-% by magnitude
PRUNE_AFTER = int(os.environ.get("PRUNE_AFTER", "512"))    # start pruning after this length
SKIP_LAYERS = [int(x) for x in os.environ.get("SKIP_LAYERS", "0,1").split(",") if x != ""]

# ---------------------------
# Helpers
# ---------------------------
def format_prompt(question, context):
    return (
        "You are a QA system. Use the context to answer the question.\n"
        "Answer with a short phrase or single entity only. No punctuation. No explanation.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def make_context_text(text, target_tokens, tokenizer):
    # tokenize and trim to target token budget
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

# --- EM (Exact Match) normalization ---
def normalize_answer(s: str) -> str:
    def lower(text): return text.lower()
    def remove_punc(text): return re.sub(r"[^\w\s]", "", text)
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(pred: str, golds) -> bool:
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)

# --- TTFT measurer ---
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
        if input_ids.shape[1] > self.start_len:
            torch.cuda.synchronize()
            self._t1.record()
            torch.cuda.synchronize()
            self.first_token_ms = self._t0.elapsed_time(self._t1)
        return False

# ---------------------------
# NarrativeQA data helpers
# ---------------------------
def load_narrativeqa_examples(n):
    """
    Load NarrativeQA plain_text validation split and extract tuples:
      (question, context_text, [gold_answers])
    We use the 'summary' for context (shorter, better for QA).
    """
    ds = load_dataset("narrativeqa", "default", split=f"validation[:{n}]")
    examples = []
    for ex in ds:
        # Robust field extraction (dataset has nested structures)
        # Question text
        q = ex.get("question") or ex.get("qtext") or ""
        # Answers: may be list of strings or list of dicts with 'text'
        raw_ans = ex.get("answers", [])
        if isinstance(raw_ans, dict) and "text" in raw_ans:
            golds = raw_ans["text"] if isinstance(raw_ans["text"], list) else [raw_ans["text"]]
        elif isinstance(raw_ans, list) and raw_ans and isinstance(raw_ans[0], dict) and "text" in raw_ans[0]:
            golds = [a["text"] for a in raw_ans]
        elif isinstance(raw_ans, list):
            golds = [str(a) for a in raw_ans]
        else:
            golds = []

        # Context: prefer summary; fall back to document text if needed
        doc = ex.get("document") or {}
        summary = doc.get("summary") or ""
        if isinstance(summary, dict):
            # some configs store {text: "..."}
            summary = summary.get("text", "") or ""
        context_text = summary.strip()
        if not context_text:
            # try full text
            full_text = doc.get("text") or ""
            if isinstance(full_text, dict):
                full_text = full_text.get("text", "") or ""
            context_text = full_text.strip()

        if q and context_text:
            examples.append((q, context_text, golds or [""]))
    return examples


def main():
        # Logging setup
    os.makedirs(LOGDIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name_tag = MODEL_ID.split("/")[-1]

    # Attention impl select
    load_kwargs = dict(torch_dtype=DTYPE, device_map=DEVICE_MAP, use_cache=True)
    if ATTN_IMPL in ("flash2", "flash_attention_2", "fa2"):
        attn_label = "flash2"
        load_kwargs["attn_implementation"] = "flash_attention_2"
    elif ATTN_IMPL == "sdpa":
        attn_label = "sdpa"
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        load_kwargs["attn_implementation"] = "sdpa"
    elif ATTN_IMPL == "eager":
        attn_label = "eager"
        load_kwargs["attn_implementation"] = "eager"
    else:
        raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

    if USE_4BIT:
        print("[info] Activating 4-bit NF4 Quantization")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,  # usage: bfloat16 or float16
            bnb_4bit_quant_type="nf4",     # Best accuracy
            bnb_4bit_use_double_quant=True # Saves an extra 0.4 bits per parameter
        )
    else:
        # Only set torch_dtype explicitly if NOT using 4bit (BnB handles storage type internally)
        load_kwargs["torch_dtype"] = DTYPE


    run_id = f"{model_name_tag}_{attn_label}_N{N_SAMPLES}_B{BATCH_SIZE}_{KV_MODE}_{ts}"
    jsonl_path = os.path.join(LOGDIR, f"{run_id}.jsonl")
    full_json_path = os.path.join(LOGDIR, f"{run_id}_full.json")

    # --- W&B init ---
    wandb.init(
        entity=ENTITY,
        project="Long-Context-Optimization",
        name=run_id,
        config={
            "model_id": MODEL_ID,
            "dtype": str(DTYPE),
            "device_map": DEVICE_MAP,
            "attn_impl": attn_label,
            "kv_mode": KV_MODE,
            "use_4bit": USE_4BIT,
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "contexts": CONTEXTS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "keep_ratio": KEEP_RATIO,
            "prune_after": PRUNE_AFTER,
            "skip_layers": SKIP_LAYERS,
        },
    )

    random.seed(0)
    
    # Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[info] Loading {MODEL_ID} (attn={attn_label}, dtype={DTYPE}, device_map={DEVICE_MAP}, kv_mode={KV_MODE})")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    print("Attention impl in config:", getattr(model.config, "_attn_implementation", None))
    try:
        print(model.model.layers[0].self_attn.forward.__qualname__)
    except Exception:
        pass

    # Data: NarrativeQA
    nq_examples = load_narrativeqa_examples(N_SAMPLES)

    # Load text metrics once
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")

    all_rows = []
    ctx_summaries = []
    oom_count = 0
    em_hits = 0  # can delete if you drop EM entirely

    all_rows = []
    ctx_summaries = []
    oom_count = 0
    em_hits = 0
    
    for ctx_len in CONTEXTS:
        per_req_lat, per_tok_lat, per_req_tokps, per_req_peak = [], [], [], []
        per_req_ttft = []
        per_req_decode_ms, per_req_mspt_decode, per_req_tokps_decode = [], [], []
        
        ctx_preds = []
        ctx_refs = []

        for i in range(0, len(nq_examples), BATCH_SIZE):
            batch = nq_examples[i:i+BATCH_SIZE]
            prompts = []
            golds_list = []
            for (q, sentences, golds) in batch:
                context_text = make_context_text(sentences, ctx_len, tok)
                prompts.append(format_prompt(q, context_text))
                golds_list.append(golds)
                
            inputs = tok(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=ctx_len).to(model.device)
            
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                
                if KV_MODE == "l2":
                    # Use L2 compression: prefill -> compress -> decode
                    with torch.inference_mode():
                        out, ttft_ms = generate_with_l2_compress(
                            model, tok, inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            keep_ratio=KEEP_RATIO,
                            prune_after=PRUNE_AFTER,
                            skip_layers=SKIP_LAYERS,
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                            eos_token_id=tok.eos_token_id,
                        )
                else:
                    ttft_timer = FirstTokenTimer(inputs["input_ids"].shape[1])
                    stoppers = StoppingCriteriaList([ttft_timer])
                    ttft_timer.arm()
                    with torch.inference_mode():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                            eos_token_id=tok.eos_token_id,
                            stopping_criteria=stoppers,
                        )
                    ttft_ms = ttft_timer.first_token_ms or float("nan")
                
                end.record()
                torch.cuda.synchronize()

                total_ms = start.elapsed_time(end)
                
                # Shapes and decoding lens
                if KV_MODE == "l2":
                    # out is a list of tensors (batch)
                    gen_lens = []
                    for bidx in range(len(batch)):
                        gen_len = int(out[bidx].shape[-1] - inputs["input_ids"][bidx].shape[-1])
                        gen_lens.append(gen_len)
                    avg_gen = sum(gen_lens)/len(gen_lens) if gen_lens else 0
                    # decode times
                    decode_ms = max(total_ms - ttft_ms, 0.0)
                    ms_per_token_decode = decode_ms / max(avg_gen, 1)
                    tok_per_s_decode = (avg_gen / (decode_ms/1000.0)) if decode_ms > 0 else float("nan")
                    texts = tok.batch_decode(out, skip_special_tokens=True)
                else:
                    gen_lens = []
                    for bidx in range(len(batch)):
                        gen_len = int(out[bidx].shape[-1] - inputs["input_ids"][bidx].shape[-1])
                        gen_lens.append(gen_len)
                    avg_gen = sum(gen_lens)/len(gen_lens) if gen_lens else 0
                    decode_ms = max(total_ms - ttft_ms, 0.0)
                    ms_per_token_decode = decode_ms / max(avg_gen, 1)
                    tok_per_s_decode = (avg_gen / (decode_ms/1000.0)) if decode_ms > 0 else float("nan")
                    texts = tok.batch_decode(out, skip_special_tokens=True)

                # metrics
                per_req_lat.append(total_ms)
                per_req_ttft.append(ttft_ms)
                per_tok_lat.append(total_ms / max(avg_gen, 1))
                per_req_tokps.append(avg_gen / (total_ms/1000.0))
                per_req_peak.append(torch.cuda.max_memory_allocated()/(1024**3))

                # Collect predictions and references for metric computation
                for j, golds in enumerate(golds_list):
                    pred = (
                        texts[j]
                        .split("Answer:", 1)[-1]   # take text after "Answer:"
                        .strip()
                        .split("\n", 1)[0]         # first line
                        .split(".")[0]             # optionally strip trailing sentence
                        .strip()
                    )
                    ctx_preds.append(pred)
                    ctx_refs.append(golds)  # golds is already a list of reference strings


                # per-request row (aggregate for the batch)
                all_rows.append({
                    "run_id": run_id,
                    "model": MODEL_ID,
                    "attn": attn_label,
                    "kv_mode": KV_MODE,
                    "context_tokens": ctx_len,
                    "batch_size": len(batch),
                    "latency_ms_total": round(total_ms, 2),
                    "ttft_ms": round(ttft_ms, 2) if isinstance(ttft_ms, float) else ttft_ms,
                    "avg_gen_tokens": round(avg_gen, 2),
                    "tok_per_s": round(per_req_tokps[-1], 2),
                    "ms_per_token": round(per_tok_lat[-1], 2),
                    "peak_gpu_mem_gb": round(per_req_peak[-1], 2),
                    "success": True,
                })

                per_req_decode_ms.append(decode_ms)
                per_req_mspt_decode.append(ms_per_token_decode if math.isfinite(ms_per_token_decode) else float("nan"))
                per_req_tokps_decode.append(tok_per_s_decode if math.isfinite(tok_per_s_decode) else float("nan"))

            except RuntimeError as e:
                is_oom = "out of memory" in str(e).lower()
                oom_count += 1 if is_oom else 0
                all_rows.append({
                    "run_id": run_id,
                    "model": MODEL_ID,
                    "attn": attn_label,
                    "kv_mode": KV_MODE,
                    "context_tokens": ctx_len,
                    "batch_size": len(batch),
                    "error": str(e)[:160],
                    "success": False,
                    "oom": bool(is_oom),
                })
                torch.cuda.empty_cache()
                continue
        
        # Compute quality metrics for this context length
        meteor_res = meteor_metric.compute(
            predictions=ctx_preds,
            references=ctx_refs,   # list of list-of-strings
        )
        rouge_res = rouge_metric.compute(
            predictions=ctx_preds,
            references=ctx_refs,
            use_stemmer=True,
        )

        meteor_score = meteor_res["meteor"]
        rouge1 = rouge_res["rouge1"]
        rougeL = rouge_res["rougeL"]


        # Context summary
        ctx_summary = {
            "run_id": run_id,
            "model": MODEL_ID,
            "attn": attn_label,
            "kv_mode": KV_MODE,
            "quantized": USE_4BIT,
            "context_tokens": ctx_len,
            "n_requests": math.ceil(len(nq_examples)/BATCH_SIZE),
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
            "meteor": round(meteor_score, 4),
            "rouge1": round(rouge1, 4),
            "rougeL": round(rougeL, 4),        
        }

        wandb.log(
            {
                "context_tokens": ctx_len,  # x-axis if you want
                "quality/meteor": meteor_score,
                "quality/rouge1": rouge1,
                "quality/rougeL": rougeL,
                "latency/latency_ms_p50": ctx_summary["latency_ms_p50"],
                "latency/latency_ms_p95": ctx_summary["latency_ms_p95"],
                "latency/ttft_ms_p50": ctx_summary["ttft_ms_p50"],
                "latency/ttft_ms_p95": ctx_summary["ttft_ms_p95"],
                "throughput/ms_per_token_p50": ctx_summary["ms_per_token_p50"],
                "throughput/ms_per_token_p95": ctx_summary["ms_per_token_p95"],
                "throughput/tok_per_s_p50": ctx_summary["tok_per_s_p50"],
                "throughput/tok_per_s_p95": ctx_summary["tok_per_s_p95"],
                "memory/peak_gpu_mem_gb_p95": ctx_summary["peak_gpu_mem_gb_p95"],
            }
        )

        print(json.dumps(ctx_summary, indent=2))
        ctx_summaries.append(ctx_summary)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(ctx_summary) + "\n")

        # --- DEBUG: show a few predictions ---
        print("\n=== Sample outputs for context length", ctx_len, "===")
        for j in range(min(3, len(nq_examples))):
            q, context_text, golds = nq_examples[j]
            ctx_trim = make_context_text(context_text, ctx_len, tok)
            prompt = format_prompt(q, ctx_trim)
            inputs_1 = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to(model.device)

            if KV_MODE == "l2":
                out, _ = generate_with_l2_compress(
                    model, tok, inputs_1,
                    max_new_tokens=MAX_NEW_TOKENS,
                    keep_ratio=KEEP_RATIO,          # e.g., 0.9
                    prune_after=PRUNE_AFTER,        # e.g., 1048
                    skip_layers=SKIP_LAYERS,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )
                pred_text = tok.decode(out[0], skip_special_tokens=True)
            else:
                out = model.generate(
                    **inputs_1, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id
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
        "kv_mode": KV_MODE,
        "quantized": USE_4BIT,
        "total_samples": N_SAMPLES,
        "batch_size": BATCH_SIZE,
        "contexts": CONTEXTS,
        "oom_count": oom_count,
        "timestamp": ts,
    }
    print(json.dumps(summary, indent=2))

    full = {
        "run_meta": {
            "run_id": run_id, "model": MODEL_ID, "attn": attn_label,
            "dtype": str(DTYPE), "device_map": DEVICE_MAP,
            "contexts": CONTEXTS, "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS,
            "kv_mode": KV_MODE, "timestamp": ts,
            "quantized": USE_4BIT,
            "kv_l2": {"keep_ratio": KEEP_RATIO, "prune_after": PRUNE_AFTER, "skip_layers": SKIP_LAYERS},
        },
        "ctx_summaries": ctx_summaries,
        "rows": all_rows,
        "summary": summary,
    }
    full_json_path = os.path.join(LOGDIR, f"{run_id}_full.json")
    with open(full_json_path, "w") as f:
        json.dump(full, f, indent=2)

        # Convert per-request rows to a W&B table (optional)
    if all_rows:
        # get consistent columns
        columns = sorted(all_rows[0].keys())
        table = wandb.Table(columns=columns)
        for row in all_rows:
            table.add_data(*(row.get(col) for col in columns))
        wandb.log({"per_request_metrics": table})

    wandb.finish()

if __name__ == "__main__":
    main()
