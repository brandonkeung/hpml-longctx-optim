import os, time, json, torch, random, string
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
CONTEXTS = [512, 2048, 8192]  # add 32768 later if VRAM allows
DTYPE = torch.bfloat16

def random_text(n_chars: int):
    return " ".join(["".join(random.choices(string.ascii_lowercase, k=6)) for _ in range(n_chars//7)])

def run_once(tok, model, ctx_tokens: int):
    prompt = random_text(max(100, ctx_tokens//4))
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_tokens).to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks_out = out.shape[-1] - inputs["input_ids"].shape[-1]
    return dt, toks_out

def main():
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map="auto", use_cache=True, attn_implementation="eager"
    )
    rows = []
    for ctx in CONTEXTS:
        try:
            dt, tg = run_once(tok, model, ctx)
            tps = tg / dt if dt > 0 else float("nan")
            peak = torch.cuda.max_memory_allocated()/(1024**3)
            rows.append({"context_tokens": ctx, "latency_s": round(dt,4), "gen_tokens": tg,
                         "throughput_tok_per_s": round(tps,2), "peak_gpu_mem_gb": round(peak,2)})
        except RuntimeError as e:
            rows.append({"context_tokens": ctx, "error": str(e)})
            torch.cuda.empty_cache()
    print(json.dumps(rows, indent=2))
    logdir = os.environ.get("LOGDIR", f"{os.environ.get('SCRATCH','.')}/logs")
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "baseline_ctx_sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)

if __name__ == "__main__":
    main()
