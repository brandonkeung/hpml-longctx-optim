"""
LongBench Evaluation with KV Cache Compression Support

This script evaluates models on the LongBench dataset with optional L2 KV cache compression.
It follows the same pattern as narrativeqa_eval.py with proper TTFT measurement.

Usage:
    # Vanilla (no compression)
    python longbench_kv_eval.py
    
    # With L2 compression
    KV_MODE=l2 KEEP_RATIO=0.7 PRUNE_AFTER=512 python longbench_kv_eval.py
    
Environment Variables:
    MODEL_ID       - HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.3)
    N_SAMPLES      - Number of samples to evaluate (default: 50)
    CONTEXTS       - Comma-separated context lengths (default: 2048,8192,16384)
    KV_MODE        - none | l2 (default: none)
    KEEP_RATIO     - For L2: fraction to keep (default: 0.7)
    PRUNE_AFTER    - For L2: start pruning after N tokens (default: 512)
    SKIP_LAYERS    - For L2: layers to skip (default: 0,1)
"""

import os, time, json, math, random, datetime, re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from kv_compression.kv_l2_dynamic import generate_with_l2_compress
from transformers import BitsAndBytesConfig 
import wandb

# --- FlexAttention availability check ---
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask  # noqa: F401
    _HAS_FLEX_ATTENTION = True
except Exception:  # pragma: no cover - flex kernel optional
    flex_attention = None
    create_block_mask = None
    _HAS_FLEX_ATTENTION = False

# ---------------------------
# Config (env-overridable)
# ---------------------------
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")     # Qwen/Qwen1.5-1.8B-Chat
# MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen1.5-1.8B-Chat")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
CONTEXTS = [int(x) for x in os.environ.get("CONTEXTS", "2048,4096,8192,16384,32768").split(",")]
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa").lower()  # eager | sdpa | flash2 | flex
LOGDIR = os.environ.get("LOGDIR", "./logs/longbench")

# QUANTIZATION SETTINGS
QUANT_MODE = os.environ.get("QUANT_MODE", "bnb4").lower()   # none | bnb8 | bnb4
BNB_4BIT_TYPE = os.environ.get("BNB_4BIT_TYPE", "nf4").lower()  # nf4 | fp4
BNB_4BIT_DOUBLE_QUANT = os.environ.get("BNB_4BIT_DOUBLE_QUANT", "true").lower() == "true"

BNB_4BIT_COMPUTE_DTYPE_STR = os.environ.get("BNB_4BIT_COMPUTE_DTYPE", "bfloat16").lower()
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16 if BNB_4BIT_COMPUTE_DTYPE_STR in ("bf16", "bfloat16") else torch.float16

LLM_INT8_THRESHOLD = float(os.environ.get("LLM_INT8_THRESHOLD", "6.0"))
ENTITY = "andyyang903"

DATASET_ID = os.environ.get("DATASET_ID", "zai-org/LongBench-v2")
SPLIT = os.environ.get("SPLIT", f"train[:{N_SAMPLES}]")

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
def format_prompt(question, context, choices):
    """Format prompt for multiple-choice QA"""
    labels = ["A", "B", "C", "D"]
    options_str = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, choices))
    
    return (
        "You are a multiple-choice question answering system.\n"
        "Use the context to answer the question.\n"
        "Choose the best option among A, B, C, and D.\n"
        "Answer with a single capital letter: A, B, C, or D. No explanation.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        "Answer:"
    )


def make_context_text(context, target_tokens, tokenizer, reserve_tokens=512):
    """
    Truncate the context so there is room left for instructions, question,
    options, and the 'Answer:' tag. This avoids truncating off the tail.
    """
    target_context_tokens = max(target_tokens - reserve_tokens, 128)
    ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ids) > target_context_tokens:
        ids = ids[:target_context_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return context


def percentile(values, p):
    """Calculate percentile of a list of values"""
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return values[int(k)] if f == c else values[f] * (c - k) + values[c] * (k - f)


def extract_choice_letter(text: str) -> str:
    """
    Extract the first clear choice letter A/B/C/D from generated text.
    Handles various formats: "A", "a", "(A)", "A.", "A:", "Option A", "The answer is A", etc.
    """
    if not text:
        return ""
    part = text.strip()
    
    # Priority 1: First character is A-D (most common case for well-behaved models)
    if part and part[0].upper() in "ABCD":
        return part[0].upper()
    
    # Priority 2: Starts with common patterns like "(A)" or "A." or "A:"
    m = re.match(r"^\s*\(?([A-Da-d])\)?[\.\:\)\s]", part)
    if m:
        return m.group(1).upper()
    
    # Priority 3: "Option A" or "Choice A" or "Answer: A" patterns
    m = re.search(r"(?:option|choice|answer)[:\s]+\(?([A-Da-d])\)?", part, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Priority 4: "The answer is A" or "I choose A" patterns
    m = re.search(r"(?:is|choose|select|pick)\s+\(?([A-Da-d])\)?", part, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Priority 5: Any standalone A-D letter (with word boundary, case-insensitive)
    m = re.search(r"\b([A-Da-d])\b", part)
    if m:
        return m.group(1).upper()
    
    # Priority 6: Parenthesized letter anywhere
    m = re.search(r"\(([A-Da-d])\)", part)
    if m:
        return m.group(1).upper()
    
    return ""


def choice_exact_match(pred_letter: str, gold_letter: str) -> bool:
    """Check if predicted choice matches gold choice"""
    if not pred_letter:
        return False
    return pred_letter.strip().upper()[:1] == gold_letter.strip().upper()[:1]


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
    elif ATTN_IMPL in ("flex", "flex_attention"):
        if not _HAS_FLEX_ATTENTION:
            raise RuntimeError(
                "ATTN_IMPL=flex requested but torch.nn.attention.flex_attention is not available. "
                "Please ensure you are running PyTorch with FlexAttention support."
            )
        attn_label = "flex"
        load_kwargs["attn_implementation"] = "flex_attention"
    else:
        raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")
    
    if QUANT_MODE == "none":
        # Your current baseline
        load_kwargs["torch_dtype"] = DTYPE

    elif QUANT_MODE == "bnb8":
        print("[info] Activating 8-bit (bitsandbytes / LLM.int8)")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=LLM_INT8_THRESHOLD,
            # optional knobs you can add later if you want:
            # llm_int8_enable_fp32_cpu_offload=False,
        )

    elif QUANT_MODE == "bnb4":
        print(f"[info] Activating 4-bit (bitsandbytes) type={BNB_4BIT_TYPE} double_quant={BNB_4BIT_DOUBLE_QUANT} compute={BNB_4BIT_COMPUTE_DTYPE}")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_quant_type=BNB_4BIT_TYPE,  # "nf4" or "fp4"
            bnb_4bit_use_double_quant=BNB_4BIT_DOUBLE_QUANT,
        )

    else:
        raise ValueError(f"Unknown QUANT_MODE: {QUANT_MODE}")
    
    run_id = f"{model_name_tag}_{attn_label}_N{N_SAMPLES}_B{BATCH_SIZE}_KV{KV_MODE}_Q{QUANT_MODE}_{ts}"
    jsonl_path = os.path.join(LOGDIR, f"{run_id}.jsonl")
    full_json_path = os.path.join(LOGDIR, f"{run_id}_full.json")
    
    random.seed(0)

    # --- W&B init ---
    wandb.init(
        entity=ENTITY,
        project="testrun-longbench",
        name=run_id,
        config={
            "model_id": MODEL_ID,
            "dtype": str(DTYPE),
            "device_map": DEVICE_MAP,
            "attn_impl": attn_label,
            "kv_mode": KV_MODE,
            "quant_mode": QUANT_MODE,
            "bnb_4bit_type": BNB_4BIT_TYPE if QUANT_MODE == "bnb4" else None,
            "bnb_4bit_double_quant": BNB_4BIT_DOUBLE_QUANT if QUANT_MODE == "bnb4" else None,
            "bnb_4bit_compute_dtype": str(BNB_4BIT_COMPUTE_DTYPE) if QUANT_MODE == "bnb4" else None,
            "llm_int8_threshold": LLM_INT8_THRESHOLD if QUANT_MODE == "bnb8" else None,
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "contexts": CONTEXTS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "keep_ratio": KEEP_RATIO,
            "prune_after": PRUNE_AFTER,
            "skip_layers": SKIP_LAYERS,
            "dataset_id": DATASET_ID,
            "split": SPLIT,
        },
    )
    
    # Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[info] Loading {MODEL_ID} (attn={attn_label}, dtype={DTYPE}, device_map={DEVICE_MAP}, kv_mode={KV_MODE}, quantization={QUANT_MODE})")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    print("Attention impl in config:", getattr(model.config, "_attn_implementation", None))
    try:
        print(model.model.layers[0].self_attn.forward.__qualname__)
    except Exception:
        pass
    
    # Load LongBench data
    print(f"[info] Loading LongBench dataset: {DATASET_ID}, split={SPLIT}")
    ds = load_dataset(DATASET_ID, split=SPLIT)
    
    examples = []
    for ex in ds:
        q = ex["question"]
        context = ex["context"]
        choices = [ex["choice_A"], ex["choice_B"], ex["choice_C"], ex["choice_D"]]
        gold_letter = ex["answer"]
        examples.append((q, context, choices, gold_letter))
    
    print(f"[info] Loaded {len(examples)} examples")
    
    all_rows = []
    ctx_summaries = []
    oom_count = 0
    
    for ctx_len in CONTEXTS:
        em_hits = 0
        per_req_lat, per_tok_lat, per_req_tokps, per_req_peak = [], [], [], []
        per_req_ttft, per_req_em = [], []
        per_req_decode_ms, per_req_mspt_decode, per_req_tokps_decode = [], [], []
        
        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i+BATCH_SIZE]
            prompts = []
            golds_list = []
            for (q, context, choices, gold_letter) in batch:
                context_text = make_context_text(context, ctx_len, tok)
                prompt = format_prompt(q, context_text, choices)
                prompts.append(prompt)
                # breakpoint()
                golds_list.append(gold_letter)

                # breakpoint()
            
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
                    # out is tensor of shape (batch_size, total_len)
                    input_len = inputs["input_ids"].shape[1]  # Batch-wide padded length
                    gen_lens = []
                    for bidx in range(len(batch)):
                        gen_len = int(out[bidx].shape[-1] - input_len)
                        gen_lens.append(gen_len)
                    avg_gen = sum(gen_lens)/len(gen_lens) if gen_lens else 0
                    # decode times
                    decode_ms = max(total_ms - ttft_ms, 0.0)
                    ms_per_token_decode = decode_ms / max(avg_gen, 1)
                    tok_per_s_decode = (avg_gen / (decode_ms/1000.0)) if decode_ms > 0 else float("nan")
                    
                    # Decode generated tokens only for EM
                    texts = []
                    for bidx in range(len(batch)):
                        gen_seq = out[bidx][input_len:]  # Use batch-wide input_len
                        gen_text = tok.decode(gen_seq, skip_special_tokens=True)
                        texts.append(gen_text)
                else:
                    gen_lens = []
                    input_len = inputs["input_ids"].shape[1]
                    for bidx in range(len(batch)):
                        gen_len = int(out[bidx].shape[-1] - input_len)
                        gen_lens.append(gen_len)
                    avg_gen = sum(gen_lens)/len(gen_lens) if gen_lens else 0
                    decode_ms = max(total_ms - ttft_ms, 0.0)
                    ms_per_token_decode = decode_ms / max(avg_gen, 1)
                    tok_per_s_decode = (avg_gen / (decode_ms/1000.0)) if decode_ms > 0 else float("nan")
                    
                    # Decode generated tokens only for EM
                    texts = []
                    for bidx in range(len(batch)):
                        gen_seq = out[bidx][input_len:]
                        gen_text = tok.decode(gen_seq, skip_special_tokens=True)
                        texts.append(gen_text)
                
                # metrics
                per_req_lat.append(total_ms)
                per_req_ttft.append(ttft_ms)
                per_tok_lat.append(total_ms / max(avg_gen, 1))
                per_req_tokps.append(avg_gen / (total_ms/1000.0))
                per_req_peak.append(torch.cuda.max_memory_allocated()/(1024**3))
                
                # EM scoring
                for j, gold_letter in enumerate(golds_list):
                    pred_letter = extract_choice_letter(texts[j])
                    # breakpoint()
                    is_em = 1 if choice_exact_match(pred_letter, gold_letter) else 0
                    per_req_em.append(is_em)
                    em_hits += is_em
                
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
                
            except (RuntimeError, torch.OutOfMemoryError) as e:
                print(str(e)[:160])
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
        
        # Context summary
        ctx_summary = {
            "run_id": run_id,
            "model": MODEL_ID,
            "attn": attn_label,
            "kv_mode": KV_MODE,
            "quant_mode": QUANT_MODE,
            "bnb_4bit_type": BNB_4BIT_TYPE if QUANT_MODE == "bnb4" else None,
            "bnb_4bit_double_quant": BNB_4BIT_DOUBLE_QUANT if QUANT_MODE == "bnb4" else None,
            "bnb_4bit_compute_dtype": str(BNB_4BIT_COMPUTE_DTYPE) if QUANT_MODE == "bnb4" else None,
            "llm_int8_threshold": LLM_INT8_THRESHOLD if QUANT_MODE == "bnb8" else None,
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
        wandb.log(
            {
                "context_tokens": ctx_len,
                "quality/em_rate": ctx_summary["em_rate"],
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
        for j in range(min(3, len(examples))):
            q, context, choices, gold_letter = examples[j]
            context_text = make_context_text(context, ctx_len, tok)
            prompt = format_prompt(q, context_text, choices)
            inputs_1 = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to(model.device)
            
            try:
                if KV_MODE == "l2":
                    out, _ = generate_with_l2_compress(
                        model, tok, inputs_1,
                        max_new_tokens=MAX_NEW_TOKENS,
                        keep_ratio=KEEP_RATIO,
                        prune_after=PRUNE_AFTER,
                        skip_layers=SKIP_LAYERS,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.eos_token_id,
                    )
                    gen_seq = out[0][inputs_1["input_ids"].shape[1]:]
                    gen_text = tok.decode(gen_seq, skip_special_tokens=True)
                else:
                    out = model.generate(
                        **inputs_1, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id
                    )
                    gen_seq = out[0][inputs_1["input_ids"].shape[1]:]
                    gen_text = tok.decode(gen_seq, skip_special_tokens=True)
                
                pred_letter = extract_choice_letter(gen_text)
                print(f"Q: {q[:100]}...")
                print(f"Pred: {pred_letter}  (gen: {gen_text[:100]!r})")
                print(f"Gold: {gold_letter}\n")
                # breakpoint()
            except (RuntimeError, torch.OutOfMemoryError) as e:
                print(f"[debug] OOM while generating sample output: {str(e)[:160]}")
                torch.cuda.empty_cache()
                continue
        print("=====================================================\n")
    
    # Final overall summary
    summary = {
        "run_id": run_id,
        "quant_mode": QUANT_MODE,
        "bnb_4bit_type": BNB_4BIT_TYPE if QUANT_MODE == "bnb4" else None,
        "bnb_4bit_double_quant": BNB_4BIT_DOUBLE_QUANT if QUANT_MODE == "bnb4" else None,
        "bnb_4bit_compute_dtype": str(BNB_4BIT_COMPUTE_DTYPE) if QUANT_MODE == "bnb4" else None,
        "llm_int8_threshold": LLM_INT8_THRESHOLD if QUANT_MODE == "bnb8" else None,
        "model": MODEL_ID,
        "attn": attn_label,
        "kv_mode": KV_MODE,
        "total_samples": len(examples),
        "batch_size": BATCH_SIZE,
        "contexts": CONTEXTS,
        "oom_count": oom_count,
        "overall_em": round(em_hits / len(examples), 4) if examples else float("nan"),
        "timestamp": ts,
    }
    print(json.dumps(summary, indent=2))
    
    full = {
        "run_meta": {
            "run_id": run_id, "model": MODEL_ID, "attn": attn_label,
            "quant_mode": QUANT_MODE,
            "bnb_4bit_type": BNB_4BIT_TYPE if QUANT_MODE == "bnb4" else None,
            "bnb_4bit_double_quant": BNB_4BIT_DOUBLE_QUANT if QUANT_MODE == "bnb4" else None,
            "bnb_4bit_compute_dtype": str(BNB_4BIT_COMPUTE_DTYPE) if QUANT_MODE == "bnb4" else None,
            "llm_int8_threshold": LLM_INT8_THRESHOLD if QUANT_MODE == "bnb8" else None,
            "dtype": str(DTYPE), "device_map": DEVICE_MAP,
            "contexts": CONTEXTS, "n_samples": len(examples),
            "batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS,
            "kv_mode": KV_MODE, "timestamp": ts,
            "kv_l2": {"keep_ratio": KEEP_RATIO, "prune_after": PRUNE_AFTER, "skip_layers": SKIP_LAYERS},
        },
        "ctx_summaries": ctx_summaries,
        "rows": all_rows,
        "summary": summary,
    }
    with open(full_json_path, "w") as f:
        json.dump(full, f, indent=2)

    # Convert per-request rows to a W&B table (optional)
    if all_rows:
        columns = sorted(all_rows[0].keys())
        table = wandb.Table(columns=columns)
        for row in all_rows:
            table.add_data(*(row.get(col) for col in columns))
        wandb.log({"per_request_metrics": table})

    wandb.finish()


if __name__ == "__main__":
    main()
