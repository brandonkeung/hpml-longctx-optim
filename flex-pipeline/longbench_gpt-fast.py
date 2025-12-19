# benchmark.py

import json
import time
import re
import math
import os
from datasets import load_dataset
from pathlib import Path
import torch
from transformers import AutoTokenizer
from tokenizer import get_tokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import wandb

import itertools
import sys
import time
from typing import Optional, Tuple, Union

import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

# ---------------------------
# Settingsc
# ---------------------------

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "100"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
CONTEXTS = [int(x) for x in os.environ.get("CONTEXTS", "2048,4096,8192").split(",")]
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
LOGDIR = os.environ.get("LOGDIR", "./logs")
MASK_TYPE = os.environ.get("MASK_TYPE", "prefix_lm").lower()  # causal | sliding | block_local | prefix_lm

DATASET_ID = os.environ.get("DATASET_ID", "zai-org/LongBench-v2")
WARMUP_SIZE = int(os.environ.get("WARMUP_SIZE", "5"))
ENTITY = "jd4136-columbia-university"
SPLIT = os.environ.get("SPLIT", f"train[:{N_SAMPLES}]")

device = "cuda"
DTYPE = precision = torch.bfloat16

# ---------------------------
# Torch configuration
# ---------------------------

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True
torch._dynamo.config.cache_size_limit = 64

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

# ---------------------------
# GPT-fast utils functions
# ---------------------------

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv

def sliding_mask(b, h, q, kv, W=1024):
    return (q >= kv) & ((q - kv) < W)

def block_local_mask(b, h, q, kv, B=128):
    return torch.eq(q // B, kv // B)

def prefix_lm_mask(b, h, q, kv, PREFIX=256):
    return (q < PREFIX) | (q >= kv)

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, mask_fn, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    mask = create_block_mask(mask_fn, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int,mask_fn, callback=lambda _: _, **sampling_kwargs):
    block_mask = create_block_mask(mask_fn, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, block_mask, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

    return new_tokens, new_probs

def model_forward(model, x, input_pos):
    return model(x, input_pos)

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    mask_fn,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)
    
    # Prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, mask_fn=mask_fn, **sampling_kwargs).clone()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    prefill_time_s = t1 - t0
    seq[:, T] = next_token.squeeze()

    # Decode
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    accept_counts = [0] * (speculate_k + 1)

    generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, callback=callback, mask_fn=mask_fn, **sampling_kwargs)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    decode_time_s = t3 - t2    
    seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {
        'accept_counts': accept_counts,
        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "num_prefill_tokens": T,
        "num_decode_tokens": max_new_tokens - 1,  
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

# ---------------------------
# Helper functions
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

# def make_context_text(context, max_total_tokens, prompt_overhead=512):
#     max_ctx_tokens = max_total_tokens - prompt_overhead
#     ctx_ids = sp_tokenizer.encode(context)
#     if len(ctx_ids) > max_ctx_tokens:
#         ctx_ids = ctx_ids[:max_ctx_tokens]

#     return sp_tokenizer.decode(ctx_ids)


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
# Load model + tokenizer
# ---------------------------

if MASK_TYPE == "causal":
    mask_fn = causal_mask
elif MASK_TYPE == "sliding":
    mask_fn = sliding_mask
elif MASK_TYPE == "block_local":
    mask_fn = block_local_mask
elif MASK_TYPE == "prefix_lm":
    mask_fn = prefix_lm_mask
else:
    raise ValueError(f"Unknown MASK_TYPE={MASK_TYPE}")

print(f"[info] Using MASK_TYPE={MASK_TYPE}")

checkpoint_path = Path(f"checkpoints/{MODEL_ID}/model.pth")
tokenizer_path = checkpoint_path.parent / "tokenizer.model"

ts = time.strftime("%Y%m%d_%H%M%S")
safe_model = MODEL_ID.split("/")[-1]
run_id = f"{safe_model}_{MASK_TYPE}_N{N_SAMPLES}_B{BATCH_SIZE}_new{MAX_NEW_TOKENS}_{ts}"

summary_path = Path(LOGDIR) / f"{run_id}.json"
full_json_path = os.path.join(LOGDIR, f"{run_id}_full.json")

wandb.init(
    entity=ENTITY,
    project="Long-Context-Optimization",
    name=run_id,
    config={
        "model_id": MODEL_ID,
        "dtype": str(DTYPE),
        "mask_type": MASK_TYPE,
        "n_samples": N_SAMPLES,
        "batch_size": BATCH_SIZE,
        "contexts": CONTEXTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dataset_id": DATASET_ID,
        "split": SPLIT,
    },
)

print('[info] Compiling')
create_block_mask = torch.compile(create_block_mask)
decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

print("[info] Loading gpt-fast model...")
model = _load_model(
    checkpoint_path,
    device=device,
    precision=precision,
    use_tp=False,
)
model.eval()

print("[info] Loading tokenizers...")
sp_tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# ---------------------------
# Dataset
# ---------------------------

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

# ----------------------------
# 4) Benchmark main loop (per context length)
# ----------------------------

block_size = model.config.block_size
max_ctx_tokens = block_size - MAX_NEW_TOKENS
if max_ctx_tokens <= 0:
    raise ValueError(
        f"MAX_NEW_TOKENS={MAX_NEW_TOKENS} too large for block_size={block_size}"
    )

print(f"[info] block_size={block_size}, max_ctx_tokens={max_ctx_tokens}")

all_rows = []
ctx_summaries = []

device = "cuda" if torch.cuda.is_available() else "cpu"

for ctx_len in CONTEXTS:
    print(f"\n=== ctx_len={ctx_len} ===")

    # ----------------------------
    # Per-request accumulators
    # ----------------------------
    per_req_lat_ms = []
    per_req_tokps = []
    per_req_mspt = []
    per_req_peak_gb = []
    per_req_em = []

    per_req_prefill_ms = []
    per_req_decode_ms = []
    per_req_prefill_tokps = []
    per_req_decode_tokps = []

    # ----------------------------
    # Build prompts
    # ----------------------------
    prompts = []
    golds = []

    for (q, context, choices, gold_letter) in examples:
        ctx_text = make_context_text(context, ctx_len, hf_tokenizer)
        prompt_raw = format_prompt(q, ctx_text, choices)

        ids = sp_tokenizer.encode(prompt_raw)
        ids = ids[:max_ctx_tokens]
        if len(ids) == 0:
            ids = [sp_tokenizer.bos_id()]

        prompt_tensor = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
        prompts.append(prompt_tensor)
        golds.append(gold_letter)
    
    # for (q, context, choices, gold_letter) in examples:
    #     empty_prompt = format_prompt(q, "", choices)
    #     prompt_overhead = len(sp_tokenizer.encode(empty_prompt))

    #     max_ctx_tokens = MAX_NEW_TOKENS - prompt_overhead - 8
    #     max_ctx_tokens = max(max_ctx_tokens, 0)

    #     ctx_ids = sp_tokenizer.encode(context)
    #     ctx_ids = ctx_ids[:max_ctx_tokens]
    #     ctx_text = sp_tokenizer.decode(ctx_ids)

    #     prompt_raw = format_prompt(q, ctx_text, choices)
    #     ids = sp_tokenizer.encode(prompt_raw)
    #     ids = ids[:MAX_NEW_TOKENS]

    #     prompt_tensor = torch.tensor(ids, device=device).unsqueeze(0)
    #     prompts.append(prompt_tensor)
    #     golds.append(gold_letter)
        
    # ----------------------------
    # Warmup
    # ----------------------------
    print("[info] Warming up...")
    torch.cuda.synchronize()
    torch.profiler._utils._init_for_cuda_graphs()

    for i in range(min(WARMUP_SIZE, len(prompts))):
        _ = generate(
            model,
            prompts[-1 - i],
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=1,
            interactive=False,
            draft_model=None,
            temperature=0.0,
            top_k=0,
            mask_fn=mask_fn,
        )

    torch.cuda.synchronize()

    # ----------------------------
    # Timed runs
    # ----------------------------
    for i, prompt_tensor in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        seq, stats = generate(
            model,
            prompt_tensor,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=1,
            interactive=False,
            draft_model=None,
            temperature=0.0,
            top_k=0,
            mask_fn=mask_fn,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # ---- end-to-end ----
        total_s = max(t1 - t0, 1e-9)
        total_ms = total_s * 1000.0

        input_len = int(prompt_tensor.size(-1))
        gen_len = int(seq.size(-1) - input_len)

        tokps = gen_len / total_s
        mspt = total_ms / max(gen_len, 1)

        peak_gb = torch.cuda.max_memory_reserved() / 1e9

        # ---- phase stats ----
        prefill_s = float(stats["prefill_time_s"])
        decode_s = float(stats["decode_time_s"])

        prefill_tok = int(stats["num_prefill_tokens"])
        decode_tok = int(stats["num_decode_tokens"])

        prefill_tokps = prefill_tok / prefill_s if prefill_s > 0 else float("nan")
        decode_tokps = decode_tok / decode_s if decode_s > 0 else float("nan")

        # ---- accuracy ----
        gen_ids = seq[0, input_len:]
        gen_text = sp_tokenizer.decode(gen_ids.tolist())
        pred_letter = extract_choice_letter(gen_text)
        is_em = int(choice_exact_match(pred_letter, golds[i]))

        # ----------------------------
        # Append per-request metrics
        # ----------------------------
        per_req_lat_ms.append(total_ms)
        per_req_tokps.append(tokps)
        per_req_mspt.append(mspt)
        per_req_peak_gb.append(peak_gb)
        per_req_em.append(is_em)

        per_req_prefill_ms.append(prefill_s * 1000.0)
        per_req_decode_ms.append(decode_s * 1000.0)
        per_req_prefill_tokps.append(prefill_tokps)
        per_req_decode_tokps.append(decode_tokps)
        
        all_rows.append(
            {
                "model": MODEL_ID,
                "mask": MASK_TYPE,
                "context_tokens": ctx_len,
                "request_id": i,

                "input_len": input_len,
                "gen_len": gen_len,

                "latency_ms": total_ms,
                "prefill_latency_ms": prefill_s * 1000.0,
                "decode_latency_ms": decode_s * 1000.0,

                "tok_per_s": tokps,
                "ms_per_token": mspt,
                "prefill_tokps": prefill_tokps,
                "decode_tokps": decode_tokps,

                "peak_gpu_mem_gb": peak_gb,

                "pred_letter": pred_letter,
                "gold_letter": golds[i],
                "em": is_em,
            }
        )

        print(
            f"[{i}] total={total_s:.3f}s ({tokps:.2f} tok/s), "
            f"prefill={prefill_s:.3f}s ({prefill_tokps:.2f} tok/s), "
            f"decode={decode_s:.3f}s ({decode_tokps:.2f} tok/s), "
            f"mem={peak_gb:.2f}GB, pred={pred_letter} gold={golds[i]}"
        )

    # ----------------------------
    # Context-level summary
    # ----------------------------
    ctx_summary = {
        "model": MODEL_ID,
        "mask": MASK_TYPE,
        "context_tokens": ctx_len,
        "n_requests": len(prompts),

        # end-to-end
        "latency_ms_p50": round(percentile(per_req_lat_ms, 0.50), 2),
        "latency_ms_p95": round(percentile(per_req_lat_ms, 0.95), 2),
        "tok_per_s_p50": round(percentile(per_req_tokps, 0.50), 2),
        "tok_per_s_p95": round(percentile(per_req_tokps, 0.95), 2),
        "ms_per_token_p50": round(percentile(per_req_mspt, 0.50), 2),
        "ms_per_token_p95": round(percentile(per_req_mspt, 0.95), 2),

        # prefill
        "prefill_latency_ms_p50": round(percentile(per_req_prefill_ms, 0.50), 2),
        "prefill_latency_ms_p95": round(percentile(per_req_prefill_ms, 0.95), 2),
        "prefill_tokps_p50": round(percentile(per_req_prefill_tokps, 0.50), 2),
        "prefill_tokps_p95": round(percentile(per_req_prefill_tokps, 0.95), 2),

        # decode
        "decode_latency_ms_p50": round(percentile(per_req_decode_ms, 0.50), 2),
        "decode_latency_ms_p95": round(percentile(per_req_decode_ms, 0.95), 2),
        "decode_tokps_p50": round(percentile(per_req_decode_tokps, 0.50), 2),
        "decode_tokps_p95": round(percentile(per_req_decode_tokps, 0.95), 2),

        # misc
        "peak_gpu_mem_gb_p95": round(percentile(per_req_peak_gb, 0.95), 2),
        "em_rate": round(sum(per_req_em) / max(len(per_req_em), 1), 4),
    }

    ctx_summaries.append(ctx_summary)
    print("\n[context summary]")
    print(ctx_summary)
    wandb.log(
        {
            "context_tokens": ctx_len,
            "quality/em_rate": ctx_summary["em_rate"],
            "latency/latency_ms_p50": ctx_summary["latency_ms_p50"],
            "latency/latency_ms_p95": ctx_summary["latency_ms_p95"],
            "latency/prefill_latency_ms_p50": ctx_summary["prefill_latency_ms_p50"],
            "latency/prefill_latency_ms_p95": ctx_summary["prefill_latency_ms_p95"],
            "throughput/ms_per_token_p50": ctx_summary["ms_per_token_p50"],
            "throughput/ms_per_token_p95": ctx_summary["ms_per_token_p95"],
            "throughput/tok_per_s_p50": ctx_summary["tok_per_s_p50"],
            "throughput/tok_per_s_p95": ctx_summary["tok_per_s_p95"],
            "memory/peak_gpu_mem_gb_p95": ctx_summary["peak_gpu_mem_gb_p95"],
        }
    )
    
    print(f"\n=== Sample outputs for context length {ctx_len} ===")
    for j in range(min(3, len(examples))):
        q, context, choices, gold_letter = examples[j]

        # build prompt exactly like benchmark
        # ctx_text = make_context_text(context, ctx_len)
        ctx_text = make_context_text(context, ctx_len, hf_tokenizer)
        prompt_raw = format_prompt(q, ctx_text, choices)

        # tokenize with sp tokenizer (same as benchmark)
        ids = sp_tokenizer.encode(prompt_raw)
        ids = ids[:max_ctx_tokens]
        if len(ids) == 0:
            ids = [sp_tokenizer.bos_id()]
        prompt_tensor = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)

        # run gpt-fast generate with deterministic decoding (no sampling)
        torch.cuda.synchronize()
        debug_seq, _debug_stats = generate(
            model,
            prompt_tensor,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=1,
            interactive=False,
            draft_model=None,
            temperature=0.0,  
            top_k=0,
            mask_fn=mask_fn,
        )
        torch.cuda.synchronize()

        input_len = prompt_tensor.size(-1)
        debug_gen_ids = debug_seq[0, input_len:]
        debug_gen_text = sp_tokenizer.decode(debug_gen_ids.tolist())

        pred_letter = extract_choice_letter(debug_gen_text)
        print(f"Q: {q}")
        print(f"Pred: {pred_letter}  (gen: {debug_gen_text[:200]!r})")
        print(f"Gold: {gold_letter}\n")

# ----------------------------
# 6) Save results
# ----------------------------

all_em = []
for cs in ctx_summaries:
    # cs["em_rate"] is already a float in [0,1]
    if "em_rate" in cs and cs["em_rate"] is not None:
        all_em.append(float(cs["em_rate"]))
global_em_rate = round(sum(all_em) / max(len(all_em), 1), 4) if all_em else 0.0

summary = {
    "run_id": run_id,
    "model": MODEL_ID,
    "mask": MASK_TYPE,
    "total_samples": len(examples),
    "batch_size": BATCH_SIZE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "contexts": CONTEXTS,
    "timestamp": ts,
    "overall_em_rate": global_em_rate,

}

print(json.dumps(summary, indent=2))

full = {
    "run_meta": {
        "run_id": run_id,
        "model": MODEL_ID,
        "mask": MASK_TYPE,
        "contexts": CONTEXTS,
        "n_samples": len(examples),
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "timestamp": ts,
    },
    "all_rows": all_rows,
    "ctx_summaries": ctx_summaries,          
    "summary": summary,              
}

with open(full_json_path, "w") as f:
    json.dump(full, f, indent=2)

if all_rows:
    columns = sorted(all_rows[0].keys())
    table = wandb.Table(columns=columns)
    for row in all_rows:
        table.add_data(*(row.get(col) for col in columns))
    wandb.log({"per_request_metrics": table})

wandb.finish()
print(f"[info] wrote {full_json_path}")