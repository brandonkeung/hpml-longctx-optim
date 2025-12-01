import os
import time, json, math, random, datetime, re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

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
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
CONTEXTS = [int(x) for x in os.environ.get("CONTEXTS", "512,2048,8192").split(",")]
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
ATTN_IMPL = os.environ.get("ATTN_IMPL", "eager").lower()  # eager | sdpa | flash2 | flex
LOGDIR = os.environ.get("LOGDIR", "./logs/longbench")

DATASET_ID = os.environ.get("DATASET_ID", "zai-org/LongBench-v2")
SPLIT = os.environ.get("SPLIT", f"train[:{N_SAMPLES}]")

# ---------------------------
# Helpers
# ---------------------------
def format_prompt(question, context, choices):
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
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return values[int(k)] if f == c else values[f] * (c - k) + values[c] * (k - f)


def extract_choice_letter(text: str) -> str:
    """
    Extract the first clear choice letter A/B/C/D from *generated* text.
    """
    if not text:
        return ""
    part = text.strip()

    # Direct single letter
    m = re.search(r"\b([A-D])\b", part)
    if m:
        return m.group(1)

    # Like '(A)'
    m = re.search(r"\(([A-D])\)", part)
    if m:
        return m.group(1)

    # Fallback: first char if in A-D
    if part and part[0] in "ABCD":
        return part[0]

    return ""


def choice_exact_match(pred_letter: str, gold_letter: str) -> bool:
    if not pred_letter:
        return False
    return pred_letter.strip().upper()[:1] == gold_letter.strip().upper()[:1]


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
        torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=False,
        )
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

    run_id = f"{model_name_tag}_{attn_label}_N{N_SAMPLES}_B{BATCH_SIZE}_{ts}"
    jsonl_path = os.path.join(LOGDIR, f"{run_id}.jsonl")
    full_json_path = os.path.join(LOGDIR, f"{run_id}_full.json")

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
    ds = load_dataset(DATASET_ID, split=SPLIT)

    examples = []
    for ex in ds:
        q = ex["question"]
        context = ex["context"]
        choices = [ex["choice_A"], ex["choice_B"], ex["choice_C"], ex["choice_D"]]
        gold_letter = ex["answer"]
        examples.append((q, context, choices, gold_letter))

    all_rows = []
    ctx_summaries = []
    oom_count = 0

    # Eval
    for ctx_len in CONTEXTS:
        em_hits = 0
        per_req_lat, per_tok_lat, per_req_tokps, per_req_peak = [], [], [], []
        per_req_ttft, per_req_em = [], []
        per_req_decode_ms, per_req_mspt_decode, per_req_tokps_decode = [], [], []

        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i+BATCH_SIZE]
            prompts = []
            for (q, context, choices, _gold_letter) in batch:
                context_text = make_context_text(context, ctx_len, tok)
                prompts.append(format_prompt(q, context_text, choices))

            inputs = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ctx_len,
            ).to(model.device)

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
                        stopping_criteria=stoppers,
                        return_dict_in_generate=False,
                        output_scores=False,
                    )
                end.record()
                torch.cuda.synchronize()

                total_ms = start.elapsed_time(end)
                ttft_ms = ftt.first_token_ms or float("nan")

                # gen lens
                gen_lens = []
                input_len = inputs["input_ids"].shape[1]
                for bidx in range(len(batch)):
                    gen_len = int(out[bidx].shape[-1] - input_len)
                    gen_lens.append(gen_len)
                avg_gen = sum(gen_lens)/len(gen_lens) if gen_lens else 0

                decode_ms = max(total_ms - ttft_ms, 0.0)
                ms_per_token_decode = decode_ms / max(avg_gen, 1)
                tok_per_s_decode = (avg_gen / (decode_ms/1000.0)) if decode_ms > 0 else float("nan")

                per_req_decode_ms.append(decode_ms)
                per_req_mspt_decode.append(
                    ms_per_token_decode if math.isfinite(ms_per_token_decode) else float("nan")
                )
                per_req_tokps_decode.append(
                    tok_per_s_decode if math.isfinite(tok_per_s_decode) else float("nan")
                )

                per_req_lat.append(total_ms)
                per_req_ttft.append(ttft_ms)
                per_tok_lat.append(total_ms / max(avg_gen, 1))
                per_req_tokps.append(avg_gen / (total_ms/1000.0))
                per_req_peak.append(torch.cuda.max_memory_allocated()/(1024**3))

                # ----- EM / accuracy: decode ONLY generated tokens -----
                for j, (_q, _context, _choices, gold_letter) in enumerate(batch):
                    full_seq = out[j]
                    gen_seq = full_seq[input_len:]
                    gen_text = tok.decode(gen_seq, skip_special_tokens=True)
                    pred_letter = extract_choice_letter(gen_text)
                    is_em = 1 if choice_exact_match(pred_letter, gold_letter) else 0
                    per_req_em.append(is_em)
                    em_hits += is_em

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
                print(str(e)[:160])
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

        # Debug: show a few sample predictions
        print("\n=== Sample outputs for context length", ctx_len, "===")
        for j in range(min(3, len(examples))):
            q, context, choices, gold_letter = examples[j]
            context_text = make_context_text(context, ctx_len, tok)
            prompt = format_prompt(q, context_text, choices)
            debug_inputs = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=ctx_len,
            ).to(model.device)

            with torch.inference_mode():
                debug_out = model.generate(
                    **debug_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            debug_full = debug_out[0]
            debug_gen = debug_full[debug_inputs["input_ids"].shape[1]:]
            debug_gen_text = tok.decode(debug_gen, skip_special_tokens=True)
            pred_letter = extract_choice_letter(debug_gen_text)

            print(f"Q: {q}")
            print(f"Pred: {pred_letter}  (gen: {debug_gen_text[:200]!r})")
            print(f"Gold: {gold_letter}\n")

        print("=====================================================\n")

    summary = {
        "run_id": run_id,
        "model": MODEL_ID,
        "attn": attn_label,
        "total_samples": len(examples),
        "batch_size": BATCH_SIZE,
        "contexts": CONTEXTS,
        "oom_count": oom_count,
        "EM": em_hits / len(examples) if examples else float("nan"),
        "timestamp": ts,
    }
    print(json.dumps(summary, indent=2))

    full = {
        "run_meta": {
            "run_id": run_id, "model": MODEL_ID, "attn": attn_label,
            "dtype": str(DTYPE), "device_map": DEVICE_MAP,
            "contexts": CONTEXTS, "n_samples": len(examples),
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
