# HPML Final Project – Long-Context LLM Optimization

## Description  
Optimizing long-context LLM inference by integrating Flash/FlexAttention and L2 KV-cache compression. Benchmarks up to 32k tokens on latency, throughput, memory usage, and task quality to demonstrate faster, lighter, and scalable inference.

---

## Project Schedule (9 Weeks + Buffer)

### Week 1 – Setup & Planning
- [ ] Insomnia access + conda environments  
- [ ] GitHub repo structure (`src/`, `bench/`, `slurm/`, `env/`)  
- [ ] Hugging Face login & baseline model download (Mistral/Qwen)  
- [ ] Decide on logging format (CSV/JSON metrics)  
- [ ] Assign sub-teams  

### Week 2 – Baseline Benchmarks
- [ ] Run baseline inference with **standard attention + full KV-cache**  
- [ ] Collect metrics: latency/token, throughput, GPU memory, OOM threshold  
- [ ] Set up datasets (NarrativeQA, HotpotQA, Needle-in-a-Haystack)  
- [ ] Verify structured logs + plotting  

### Week 3 – Flash/FlexAttention Integration
- [ ] Integrate FlashAttention-2 / FlexAttention into baseline model  
- [ ] Validate output equivalence on small prompts  
- [ ] Benchmark up to 8k context (sanity check)  

### Week 4 – Full Flash/Flex Benchmarks
- [ ] Scale benchmarks to long contexts (8k → 32k)  
- [ ] Collect latency, throughput, GPU memory usage  
- [ ] Generate comparison plots (baseline vs flash/flex)  

### Week 5 – L2 KV Compression (Implementation)
- [ ] Implement L2 norm truncation for KV-cache  
- [ ] Add configurable thresholds (e.g., 25%, 50%, 75%)  
- [ ] Validate correctness of outputs  

### Week 6 – L2 KV Compression (Experiments)
- [ ] Run compression benchmarks across thresholds  
- [ ] Track memory savings vs accuracy drift (EM/F1/ROUGE)  
- [ ] Add graphs showing trade-offs  

### Week 7 – Combined Optimizations
- [ ] Run **Flash/Flex + L2 KV compression** together  
- [ ] Full context sweep (512 → 32k)  
- [ ] Generate Pareto plots (speed–memory–quality)  

### Week 8 – Buffer / Debugging
- [ ] Resolve cluster delays or OOM issues  
- [ ] Rerun missing experiments  
- [ ] Polish code for reproducibility  

### Week 9 – Final Deliverables
- [ ] Prepare final report + demo (Jupyter or recorded video)  
- [ ] Show baseline run hitting OOM vs optimized run succeeding  
- [ ] Finalize plots and structured logs  
- [ ] Submit report, code, and demo  

---

## Division of Work

- **Sub-Team 1 – Attention (2 people)**  
  - Flash/FlexAttention integration & debugging  
  - GPU profiling + compatibility with Llama/Mistral/Qwen  

- **Sub-Team 2 – Compression (1 person)**  
  - Implement L2 KV compression  
  - Run ablation studies on thresholds  

- **Sub-Team 3 – Benchmarks (1 person)**  
  - Build benchmarking harness (`bench.py`)  
  - Collect logs, generate plots, Pareto curves  

- **Everyone (Weeks 7–9)**  
  - Combined runs  
  - Report writing & demo prep  
