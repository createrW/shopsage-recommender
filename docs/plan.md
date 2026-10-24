# ShopSage Recommender — Plan and Status

**English** | [简体中文](plan.zh-CN.md)

**Updated:** 2026-10-24, after S5 · **Progress:** S0–S5 done (E007 sweep, E008 recall paths, E009 ranker, E010 LLM expansion); S6 (demo) is next · **Code:** `2b05c58` (S4) on `main`, pushed to [createrW/shopsage-recommender](https://github.com/createrW/shopsage-recommender); S5 is uncommitted · **Companion:** [memory.md](memory.md) (facts, rules, decisions)

- **Goal:** a two-stage recommender on MovieLens-20M (Two-Tower dense recall and BM25 sparse recall, a LightGBM re-ranker, optional LLM query expansion, Milvus Lite retrieval, a Streamlit demo) with every reported number measured.
- **Architecture:** offline, PyTorch trains the towers on a cloud GPU, numpy and scipy compute exact candidates for evaluation, and LightGBM re-ranks the union of both recall paths. Online, the demo runs on CPU: a numpy user tower, Milvus Lite (dense, sparse BM25 and RRF hybrid search), LightGBM, and Claude Haiku for query expansion.
- **Stack:** Python 3.11–3.12, Polars, DuckDB, PyTorch 2, LightGBM, Milvus Lite (`pymilvus`), scipy, W&B, Streamlit, Typer, pytest with hypothesis, Claude Haiku 5.5.
- **Spec:** the owner's updated scope of 2026-10-23 (§2). Design rationale is in [design.md](design.md); the sections listed in memory.md §6 are superseded until S7.

> **Working loop (owner's rule, 2026-10-23).** Before a step: read plan.md and memory.md. After it: tick the boxes here, update memory.md and both Chinese twins, then report what the owner needs to do next. **Efficiency first:** take the simplest path that measures the targets honestly; anything off that path is cut (§3). Execution is native, step by step, test first.

Cloud deployment follows the owner's hosting plan, `cloud_hosting_plan_2026-10-17.md` (outside this repo: `/Users/zy/Project/new_sys/cloud_hosting_plan_2026-10-17.md`), project P02. §6 maps it onto this repo.

## 1. Status at a glance

| Area | Done | Next |
|---|---|---|
| Data | ML-20M download; the Liang split, an exact replication of the published counts (E004); the Day-1 split kept | — |
| Evaluation | masked full-corpus top-k, truncated Recall / NDCG / HR, percentile and paired bootstrap; `shopsage evaluate` | S2: the same code scores the Two-Tower |
| Baselines | MostPop and EASE, calibrated against their published figures (E004) | — |
| Models | Two-Tower A0–A4 trained on Modal (E007): A1 Recall@50 0.5361, NDCG@10 0.3351, above EASE; ablation (a) done | S4 re-ranks its candidates |
| Retrieval | BM25 path, RRF fusion, Milvus Lite serving index with exact parity on real data; ablation (b) (E008) | S6: choose the online engine on the host |
| Ranking | LightGBM LambdaRank over dense top-500 ∪ BM25 top-100, 12 features with sparse EASE; 3 seeds (E009) | S6 ships it in the demo |
| LLM | Claude Haiku expansion with structured output, cache and fallbacks; ablation (c): routed expansion +0.022 HR@50 on fresh events (E010) | S6: the demo's free-text box |
| Demo | the probe deployed on Community Cloud: 482 MB peak, hybrid p95 176 ms (E005) | S6: full demo; latency fallback planned (§8) |
| Repository | 150 tests green; lint, types and lock clean; S4 on GitHub | commit and push S5 (you, §7) |

Measured so far, `Liang-SG-strict`, 10,000 test users: MostPop Recall@50 0.2352, NDCG@10 0.1275; EASE Recall@50 0.5210, NDCG@10 0.3280 (E004). Two-Tower A1, 3 seeds (E007): Recall@50 0.5361 ± 0.0005, NDCG@10 0.3351 ± 0.0005, +0.0151 [+0.0119, +0.0184] Recall@50 over EASE. Hard negatives: A3 − A1 = −0.0020 [−0.0033, −0.0007], so the +0.08 target is not met; the logQ correction is worth +0.225 (A1 − A0). BM25 Recall@50 0.2529; RRF hybrid 0.4647, −0.0715 [−0.0748, −0.0685] against dense (E008). **Two-stage system (LightGBM ranker, E009): Recall@50 0.5420 ± 0.0006, NDCG@10 0.3428**, +0.0209 / +0.0148 over EASE; 8.6 ms p95 per user on the laptop CPU. LLM expansion for BM25 tag queries: always-on −0.031 HR@50; routed to unknown tags +0.022 [+0.012, +0.032] on fresh events (E010).

## 2. Scope and targets (owner's spec, 2026-10-23)

- **Data:** MovieLens-20M (138,493 users, 27,278 movies, 20,000,263 ratings). Amazon Reviews 2023 Toys is cut (§3).
- **Recall:** two paths, Two-Tower (dense) and BM25 (sparse), plus their RRF hybrid. **Ranking:** LightGBM to a Top-10. **LLM:** query expansion with Claude Haiku, an optional path. **Serving retrieval:** Milvus Lite. **Demo:** Streamlit. **Tracking:** W&B. **Metrics:** NDCG@10, Recall@50 and HitRate@50 with bootstrap 95% CIs.

Targets, not results; each is reported against its published anchor:

| Target | Value | Measured as | Published anchor | Assessment |
|---|---|---|---|---|
| Recall@50 | ≥ 0.55 | truncated Recall@50 on the 10,000 Liang test users | EASE 0.521; RecVAE 0.553, the highest verified (`Liang-SG-strict`) | **not met:** the system reaches 0.5420 ± 0.0006 (E009), 0.008 short; EASE 0.5210, dense 0.5361 |
| NDCG@10 after the ranker | ≥ 0.32 | same users | EASE 0.329 (`Liang-SG-fit`) | **met:** 0.3428 (E009), +0.0148 [+0.0126, +0.0171] over EASE, +0.0074 over the dense order |
| Latency, retrieve and rank to 10 | ≤ 200 ms p95 | demo host, CPU, LLM excluded | — | on track: the whole path takes 8.6 ms p95 on the laptop through exact numpy / scipy (E009); the host (≈ 4–5× slower, E005) is measured in S6 |
| Hard-negative ablation | Recall@50 +0.08 (A3 − A1) | same users, 3 seeds per arm | — | **not met:** A3 − A1 = −0.0020 [−0.0033, −0.0007] (E007); logQ itself is worth +0.225 |

## 3. Efficiency-first changes (2026-10-23)

| Area | Before | Now | Why |
|---|---|---|---|
| Protocol (D-001) | `LOO-full` and `LOO-asof`; Liang optional, owner's call on Day 4 | `Liang-SG-strict` only, an exact replication of Liang et al.'s split; the default unless you veto it | the targets only mean something there; one protocol to build; a random 80/20 holdout needs no as-of machinery |
| Features | as-of accessor, leakage canary, 61/51-user missing-value rule | features from fold-in rows only, item statistics from training users only, one canary test | the as-of machinery existed for the timestamp split |
| User tower (D-004) | open | inductive: pooled fold-in history → MLP; no user-ID table | strict held-out users are never trained on |
| Negatives (D-005) | round 2 = uniform MNS; mined negatives optional | headline A1 (in-batch + logQ) vs A3 (A1 + mined hard negatives), 3 seeds each; A0, A2 and A4 one seed each | the spec asks for a hard-negative mining ablation |
| Retrieval engine (D-008) | exact search plus a FAISS ANN study | exact numpy offline; Milvus Lite (FLAT; dense, sparse BM25 and RRF) for serving; FAISS no longer a direct dependency (Milvus Lite 3.x runs on faiss-cpu) | Milvus is in the spec; Milvus Lite is FLAT-only, and exact search is right at 20K items |
| BM25 (D-010) | scipy BM25 | scipy offline; the same weights stored as Milvus sparse vectors; no Elasticsearch | no JVM service, and it runs on Streamlit Cloud |
| Ranker (D-006) | dense Top-500 | dense Top-500 ∪ BM25 Top-100; ~11 features, including a sparse-EASE score | two recall paths; the strongest cheap signal |
| LLM | Haiku, with vLLM Qwen as an alternative | Haiku only; ablation (c) is ~1,000 queries, BM25 on the raw query vs on the expansion | no GPU needed; ≈ \$1 |
| Training | many runs, one at a time | one Modal sweep run in parallel, ~25 min wall-clock (estimate) | one hand-off |
| Seeds (D-007) | ≥ 3 for every arm | 3 for each claimed comparison; 1 for context arms, labelled as such | the same rigour wherever a claim is made |
| Cut | — | BGE-M3, the user-ID arm, the `LOO-*` and `Liang-SG-fit` tables, subgroup diagnostics, ItemKNN, the ANN study, vLLM, MLflow, Elasticsearch, Amazon Reviews, building the Docker image | not on the path to the targets |
| Docs | everything at every step | plan.md and memory.md every step; experiment_log.md per run; README, design.md and api_spec.md at S1 (banner and decisions) and S7 (rewrite) | less upkeep |

## 4. Steps

Effort is a rough estimate of working time. Every step ends with the suite green and the working loop above.

### S0. Day 1 ✅ (2026-10-18)

Scaffold, Polars pipeline, md5-checked download, leave-one-out split, EDA notebook, docs, CI, cloud bootstrap script and MPS benchmark; 53 tests; commit `3e0f08f`. Details: memory.md §5 and experiment_log.md E001–E003.

### S1. Liang split, evaluation, baselines, packaging, Milvus probe ✅ (2026-10-23, laptop)

- [x] `src/shopsage_rec/data/liang.py`: `liang_split(positives, *, seed=98765, n_heldout_users=10_000, test_prop=0.2) -> LiangSplit`, call for call the authors' preprocessing; tested against a literal transcription of their notebook (fixed and hypothesis-generated data) and, on the real data, against the published counts: 9,990,682 positives, 136,677 users, 20,720 → 20,108 items. `load_liang()` returns CSR matrices; `partition_val_users()` gives V_A / V_B.
- [x] `configs/data/movielens_20m_liang.yaml` (`split.strategy: liang`) is the default config: `shopsage prepare` writes `data/processed/liang/` in 5 s (2.2 GB peak RSS). The Day-1 split stays one `--config` away.
- [x] `evaluation/metrics.py` (`topk_masked`, `topk_for_users`, `relevance`, `recall_at_k`, `ndcg_at_k`, `hit_rate_at_k`, `evaluate_topk`) and `evaluation/bootstrap.py` (`bootstrap_ci`, `paired_bootstrap`, the api_spec.md §8.4 procedure).
- [x] `models/baselines.py` (MostPop, EASE in float32) and `shopsage evaluate` (`evaluation/runner.py`): λ on V_A, test metrics with CIs, `per_user.parquet` and `summary.json` per run.
- [x] **Calibration gate passed (E004):** MostPop Recall@50 0.2352 (published 0.235), EASE 0.5210 (published 0.521); Recall@20 and NDCG@100 within 0.002 of Steck 2019, Table 1.
- [x] Packaging: no direct FAISS dependency; `pymilvus[milvus-lite]` in the `demo` and `dev` extras; scipy in the core; the `cpu` / `gpu` extras and their conflicts gone; `bootstrap_cloud.sh`, Makefile, CI, `Dockerfile` and `requirements.txt` follow; `uv.lock` re-locked (232 packages).
- [x] Docs sync: experiment_log.md (protocols, TARGETS, decisions D-001 to D-014, run matrix, E004); README rewritten around the Liang protocol with the measured baselines; banners on design.md and api_spec.md, whose §2.7, §6.1 and §9 now describe S1; GitHub handle `createrW` filled in.
- [x] Probe: `serving/probe.py` and `serving/app.py`, with `serving/requirements.txt` from `uv export`. On the laptop at 20,108 items, Streamlit and the app together peak at about 450 MB; hybrid search p95 is 34–44 ms (E004). Milvus Lite must `flush()` after loading: unflushed rows are scanned in Python (dense p50 181 ms before the flush, 9–17 ms after). CI runs the probe on Linux and checks that `serving/requirements.txt` matches `uv.lock`.
- [ ] Deploy the probe on Streamlit Community Cloud and record its memory and latency on the host: needs your commit, push and Community Cloud app (§7).
- **Exit:** met, except the host measurement, which waits for the deploy.

### S2. Two-Tower, negatives, Modal sweep ✅ (2026-10-24; code on the laptop, sweep on Modal)

- [x] `features/items.py`: genre multi-hot, 5-year bucket, genome PCA-64 with a missing flag, cached in `item_features.npz`.
- [x] `models/two_tower.py`: item tower (`item_idx` embedding ⊕ features → MLP → 128-d, L2-normalised); inductive user tower (mean of history-item embeddings over ≤ 50 sampled history items ⊕ log history length → MLP → 128-d); `sample_history` leaves the target out.
- [x] `models/losses.py`: `sampled_softmax` with logQ correction and accidental-hit masking (`PositiveSet`); extra negatives shared (uniform, full) or per row (mined).
- [x] `models/trainer.py`: arms A0–A4, fused AdamW (no decay on embeddings), the bf16/fp16 rule, best epoch on V_A Recall@50, export (`item_embeddings.npy`, `user_tower.npz`, `model.pt`), test metrics with CIs (`per_user.parquet`, `metrics.json`), peak memory, W&B (offline without a key). `models/mining.py`: ranks 101–500 per training user. `models/sweep.py`: the job plan.
- [x] Parity test: the numpy user tower against torch on 1,000 users (< 1e-5).
- [x] `shopsage train / mine / sweep`; `scripts/train.sh`; `scripts/modal_train.py` (prepare on a Volume → 4-run τ/lr check → stage 1 in parallel → mine → stage 2 → `sweep_summary.json`; finished runs are skipped on a re-run).
- [x] MPS smoke runs (A1, A2, A4 at 10% of users) and a 3-epoch full-data A1 check (E006).
- [x] Sweep on Modal (you, 2026-10-24): 13 runs, 2.5 A10-hours ≈ \$2.75; copied back to `artifacts/runs/`; logged as E007 with ablation (a) and paired deltas.

### S3. BM25, Milvus serving index, ablation (b) ✅ (2026-10-24, laptop)

- [x] `retrieval/bm25.py`: item documents (title words, genres, genome top-20 tags, training users' tags; terms in one document dropped); BM25 weights in scipy.sparse; the query counts the fold-in items containing each term (`log1p`); k1, b and the query weighting chosen on V_A.
- [x] `retrieval/milvus_store.py`: one collection (dense FLAT + sparse inverted index), flushed; `search_dense`, `search_sparse`, `search_hybrid` (RRF, k = 60). Seen items are removed by a filter inside the search, so results equal the offline paths exactly (simpler than over-fetching).
- [x] `retrieval/fusion.py`: RRF for offline evaluation, Milvus's formula (ranks from 1).
- [x] `retrieval/evaluate.py`, `shopsage recall`: table (b) with paired deltas, candidate coverage for S4, Milvus parity and latency next to exact numpy / scipy (E008).
- **Results:** BM25 0.2529, dense 0.5361, hybrid 0.4647 Recall@50; Milvus identical to offline on 1,000 users; exact search 16–40× faster than Milvus Lite on the laptop. Dense top-500 covers 89.5% of the targets, + BM25 top-100 89.7%.

### S4. LightGBM ranker ✅ (2026-10-24, laptop CPU)

- [x] `ranking/candidates.py`: dense top-500 ∪ positive-score BM25 top-100 per user, fold-in excluded, with each path's rank.
- [x] `ranking/features.py`: 12 features (dense score and rank, BM25 score and rank, sparse EASE with 100 neighbours per item, item log-popularity and mean rating over training users, genre affinity, genome similarity, year and year gap, log history length). The canary test changes the targets and checks that no feature moves; item statistics are tested to use training users only.
- [x] `ranking/train.py`, `shopsage rank`: LambdaRank on V_B (4,500 users to fit, 500 for early stopping on NDCG@10), 3 seeds, with and without `ease_score`; EASE and dense recomputed as references; paired deltas; feature importance; serving latency per user (E009).
- **Results:** system Recall@50 0.5420 ± 0.0006, NDCG@10 0.3428 ± 0.0000; +0.0209 [+0.0178, +0.0242] and +0.0148 [+0.0126, +0.0171] over EASE; +0.0057 and +0.0074 over the dense order. Without EASE the ranker only reproduces the dense order. Serving path 8.6 ms p95 on the laptop. 134 s for the whole run.

### S5. LLM query expansion, ablation (c) ✅ (2026-10-24, laptop and the Anthropic API, \$0.18)

- [x] `llm/expand.py`: `claude-haiku-5-5`, structured output (`keywords`, `genres`, `titles`), effort `low`, no sampling parameters, a sha256-keyed disk cache, and the raw query as fallback on a missing key, a refusal, `max_tokens`, a timeout, an API error or invalid output; tested with a fake client.
- [x] `llm/benchmark.py`, `shopsage tags`: 1,000 tag events of held-out users (their tags never enter the documents); C0 raw tag vs C1 expansion (weight 0.5, fixed before the run); C1r (routed) added after the first run and confirmed on 1,000 fresh events; cluster bootstrap over users (E010). C2 cut.
- **Results:** C1 − C0 = −0.031 [−0.057, −0.004] HR@50 (pre-declared): expanding a tag the documents already know dilutes it (−0.07), while for unknown tags expansion lifts HR@50 from ~0.05 to ~0.25. Routed (C1r) − C0 = +0.022 [+0.012, +0.032] on fresh events. \$0.14 per 1,000 calls, 1.7 s p50, 0 fallbacks.

### S6. Demo and deployment (~5 h; laptop, then Streamlit Community Cloud)

- [ ] Apply D-003: where the titles come from, and what the bundle may contain.
- [ ] Export the bundle (item embeddings, `user_tower.npz`, sparse EASE, BM25 weights, the LightGBM model, item metadata as permitted) to a GitHub Release; pin its tag and sha256 in `configs/deploy/streamlit.yaml`. The Milvus Lite collection is built at startup from the bundle.
- [ ] `serving/app.py`: pick 5 movies → Top-10 with scores and the path that retrieved each item; a dense / BM25 / hybrid toggle; a free-text query with an LLM toggle that shows the expansion, expanding only queries that are not vocabulary terms (E010) and serving cached expansions without a key; a latency readout and a benchmark button.
- [ ] Serving engine from host measurements: exact numpy / scipy (8.6 ms p95 for the whole path on the laptop, E009) vs Milvus Lite (hybrid p95 48.7 ms on the laptop, E008); Milvus shown with its parity check either way. Import LightGBM before pymilvus on macOS (memory.md §7).
- [ ] Deploy at `yi-shopsage.streamlit.app`; memory within 690 MB; p95 ≤ 200 ms on the host; record the GIF. Log E011.
- **Exit:** live URL; the latency row of §2 filled.

### S7. Write-up and publish (~3 h)

- [ ] README results and ablation tables with CIs, the W&B report link, the GIF and the URL; rewrite the superseded parts of design.md and api_spec.md; a resume line built from measured numbers only.
- [ ] Final review; you push.

## 5. Review focus

Failure modes that no step's main path exercises; each gets a test in the step named.

1. Held-out targets leaking into training or features (held-out users' tags, popularity counted over all users): canary tests in S1 and S4.
2. Fold-in items left unmasked before the top-k cut: masked top-k tests in S1, the Milvus over-fetch test in S3.
3. Milvus results drifting from the exact path (ties, filters, fewer than k results): parity tests in S3.
4. Demo memory or latency on Community Cloud (Milvus Lite on Debian 11, 690 MB): the S1 probe and the S6 host benchmark.
5. LLM path failures (no key, refusal, timeout, titles that ground to nothing): fake-client tests in S5.

## 6. Cloud deployment (follows the hosting plan)

### 6.1 What the hosting plan prescribes for this project

- **Demo, must deploy:** Streamlit Community Cloud, free, deployed from the GitHub repo, at a URL such as `https://yi-shopsage.streamlit.app`; CPU inference only, no PyTorch training there. Alternative: Hugging Face Spaces (free).
- **GPU training:** Modal, for when the machine has no GPU; RunPod is listed as the cheapest GPU option.
- **Budget:** the \$0 tier covers this project; Modal's \$30 per month is the optional item. Rules: a \$5 billing alert, and shut cloud resources down as soon as the work is done.

### 6.2 Workload to platform

| Workload | Platform | In this repo | Cost |
|---|---|---|---|
| Data, EDA, tests, EASE, BM25, LightGBM, LLM calls | laptop | `make` targets, the `shopsage` CLI | \$0 (LLM calls ≈ \$1) |
| Two-Tower sweep | Modal A10; RunPod as the alternative | `scripts/modal_train.py`; `scripts/bootstrap_cloud.sh` and `scripts/train.sh` | ≈ \$3 (estimate from the measured A1 epoch), inside the \$30 monthly credit |
| Demo | Streamlit Community Cloud | `serving/app.py`, `.streamlit/`, `packages.txt`, `configs/deploy/streamlit.yaml` | \$0 |
| Demo fallback | Hugging Face Spaces (CPU basic: 2 vCPU, 16 GB) | `Dockerfile` | \$0 |
| Demo artifacts | sha256-pinned GitHub Release asset | S6 | \$0 |
| Experiment tracking | W&B free tier | `WANDB_API_KEY` | \$0 |

### 6.3 Training hand-off: what you run

**VRAM requirement: 8 GB minimum, 16 GB or more recommended; a 24 GB A10 (Modal's `"A10"`) is the default.** Measured on the laptop (E006, MPS, B = 4,096; MPS reports the driver's allocation, an upper bound on CUDA's): A1 2.1 GB, A2 2.2 GB, A4 (full softmax) 5.0 GB. A3 adds 8 mined negatives per row, about A2's size, plus the mined pool on the GPU (116,677 × 400 as int64 ≈ 0.37 GB; 0.19 GB as int32 on disk).

| Arm | Peak memory (measured on MPS) | s / epoch, M5 MPS (measured) | s / epoch, A10 |
|---|:-:|:-:|:-:|
| A0, A1 (in-batch) | 2.1 GB (A10: 0.98 GB, measured) | 129–146 | **21.5 (measured, Modal trial)** |
| A2 (+2,048 uniform) | 2.2 GB | ~150 (10% run, scaled ×10) | ~25 (estimate) |
| A3 (+8 mined per row) | ≈ A2 | not measured | ~30 (estimate) |
| A4 (full softmax) | 5.0 GB | ~420 (10% run, scaled ×10) | ~75 (estimate) |

- **Sweep estimate:** 4 tuning runs × 10 epochs, then 9 runs × 30 epochs (A4 the longest, ~1 h) ≈ 2.5 A10-hours ≈ \$3 at Modal's A10 rate (about \$1.10 per hour); about 1–1.5 hours of wall-clock because the runs of a stage go in parallel. Estimates; `metrics.json` (`wall_s`, `peak_memory_gb`, `gpu`) replaces them.

**Option A: Modal (the hosting plan's choice).**

```bash
uv tool install modal && modal setup                 # once: the CLI, then a browser login
modal secret create wandb WANDB_API_KEY=<your key>   # once; without W&B: modal secret create wandb WANDB_MODE=offline
cd /Users/zy/Project/new_sys/projects/shopsage-recommender
modal run scripts/modal_train.py                     # data prep + the whole sweep; prints the per-arm summary
modal volume get shopsage-data artifacts/runs artifacts/   # the 13 run directories
```

`modal run` must stay connected (keep the laptop awake); if it drops, run it again: finished runs are skipped. Other GPU: `SHOPSAGE_GPU=L4 modal run scripts/modal_train.py`. Short trial first, if you like: `modal run scripts/modal_train.py --epochs 2 --tune-epochs 1` (then delete `artifacts/runs` on the Volume with `modal volume rm -r shopsage-data artifacts/runs` before the real sweep, since finished run ids are reused).

**Option B: RunPod or any Linux GPU box,** sequential, about 4–5 hours on an A10.

```bash
git clone https://github.com/createrW/shopsage-recommender.git && cd shopsage-recommender
export WANDB_API_KEY=...                             # or WANDB_MODE=offline
bash scripts/bootstrap_cloud.sh                      # uv, train + llm extras, torch check, download, Liang split
bash scripts/train.sh --sweep                        # the whole sweep, one run after another
```

Copy `artifacts/runs/` back, then **terminate** the pod: a stopped pod keeps billing for its volume disk (RunPod: \$0.20 per GB-month idle). A Community Cloud RTX 4090 (24 GB, \$0.34 per hour) is enough.

**Option C: a free T4 on Colab or Kaggle,** for smoke runs only (fp16, about twice as slow).

### 6.4 Demo deployment: Streamlit Community Cloud

The S1 probe and the S6 demo deploy the same way:

1. The repo is public on GitHub (§7).
2. At share.streamlit.io, sign in with GitHub and create an app from the repo: repository `createrW/shopsage-recommender`, branch `main`, main file `src/shopsage_rec/serving/app.py`, App URL `yi-shopsage` (a custom subdomain, changeable later).
3. Advanced settings: Python 3.12 (the default). Secrets: `ANTHROPIC_API_KEY = "..."`, only for the live LLM path from S5 on; the demo runs without it.
4. Community Cloud searches the entrypoint's directory before the repo root, so it installs `src/shopsage_rec/serving/requirements.txt` rather than the root `uv.lock`, plus `libgomp1` from the root `packages.txt`. Pushes to `main` update the app.
5. S6: upload the demo bundle to a GitHub Release, put its tag and sha256 in `configs/deploy/streamlit.yaml`, and push.
6. Checks: no torch import; Milvus Lite starts and answers; memory within the 690 MB guarantee (2.7 GB maximum); p50/p95 latency of a Top-10 request; a cold start after the app sleeps (12 hours idle).

**Fallback:** Hugging Face Spaces (Docker or Streamlit SDK; CPU basic, 2 vCPU and 16 GB, free), for example if Milvus Lite or the bundle does not fit. D-003 applies there too.

### 6.5 Cost guardrails (the hosting plan's rules, applied here)

- **Modal:** set a workspace budget (Modal's FAQ: "If you want to limit your monthly spend, you can set a Workspace budget"); the \$30 monthly Starter credit covers the plan.
- **RunPod:** prepaid credit without auto-recharge; terminate pods once the artifacts are copied out.
- **Nothing runs long-term in the cloud:** training boxes are ephemeral, and the only always-on service is the free Streamlit app (this project's version of the hosting plan's "`terraform destroy` when done").
- **Anthropic:** set a low monthly spend limit in the Console; responses are cached, so reruns cost nothing.
- **Secrets stay out of git:** `.streamlit/secrets.toml` is git-ignored and outside the Docker build context; keys come from environment variables or Streamlit secrets.
- **The demo Docker image stays local,** off public registries, until D-003 is decided.

### 6.6 Where the repo refines the hosting plan

Vendor figures were re-checked on the dates shown; the platform choices are the hosting plan's.

| Hosting plan | Checked fact | What the repo does |
|---|---|---|
| Streamlit Community Cloud: 1 GB RAM | 690 MB guaranteed, up to 2.7 GB; ≤ 2 CPU cores; sleeps after 12 hours idle (Streamlit docs, 2026-10-18) | the demo is designed for 690 MB and never imports torch |
| Modal A10 at \$0.60/h, so 10 h ≈ \$6 | A10 \$0.000306/s ≈ \$1.10/h; T4 \$0.000164/s ≈ \$0.59/h; Starter includes \$30 per month (modal.com/pricing, 2026-10-23) | the sweep ≈ \$1.1–1.7, budget 3 h ≈ \$3.3: \$0 out of pocket within the credit |
| Modal A100 40 GB at \$1.50/h | \$0.000583/s ≈ \$2.10/h | not needed: the worst case peaks at 6.1 GB |
| Hugging Face Spaces GPU at \$0.60/h | T4 small \$0.40/h; CPU basic 2 vCPU, 16 GB, free (huggingface.co/pricing, 2026-10-23) | the demo stays on CPU; Spaces is the fallback host |
| RunPod RTX 4090 at ~\$0.40/h (spot) | \$0.34/h Community Cloud, \$0.89/h Secure Cloud; idle volume disk \$0.20 per GB-month (runpod.io/pricing, 2026-10-23) | the alternative to Modal; terminate pods after use |
| P02 training: 8–12 GB VRAM, 10–15 hours | measured on MPS (E006): ≤ 5.0 GB peak (A4); the sweep needs ~3–5 A10-hours (estimate) | 8 GB minimum, 16 GB or more recommended |
| P02 checkpoints: ~500 MB | ~78 MB of parameters plus 155 MB of AdamW state per checkpoint at the Day-1 shapes (estimate) | keep the best checkpoint per run; artifacts never go into git |
| Demo: interviewers type a user_id | the GroupLens licence forbids redistribution without permission (D-003) | a visitor "pick 5 movies" profile; a user-id picker only with GroupLens's permission |
| Resume line: "<200 ms CPU inference; 50+ test users" | nothing measured yet | latency measured on the live host (S6); a user count only if real, as the hosting plan itself warns in its Part 1 |

## 7. Your action items

**Now**

- [x] Reply "go S1" (2026-10-23; no default vetoed, so D-001 and the other §3 defaults are decided).
- [x] Create the GitHub repo and push Day 1 (`createrW/shopsage-recommender`; CI green).
- [x] Commit and push S1 (`05f74c4`, 2026-10-24; CI green, including the Milvus Lite probe on Linux).

- [x] Deploy the probe on Community Cloud (2026-10-24; E005: 482 MB, hybrid p95 176 ms).
- [x] Commit and push S2 and its results (`fefcf6c`, `c26417e`, 2026-10-24).
- [x] Run the S2 sweep on Modal (2026-10-24; E007). Note: `modal volume get` refuses to overwrite `artifacts/runs/sweep_summary.json`, which the entrypoint already writes locally; download into an empty directory instead.
- [x] Commit and push S3 (`a54bdbe`, 2026-10-24).
- [x] `brew install libomp` (2026-10-24).
- [x] Commit and push S4 (`2b05c58`, 2026-10-24).
- [x] An Anthropic API key in `~/.zshrc` (2026-10-24); S5 spent \$0.18.
- [ ] Commit and push S5:

  ```bash
  git add -A
  git commit -m "S5: Claude Haiku query expansion, routed to unknown tags (E010)"
  git push
  ```

- [ ] Decide D-003 for S6 (below) if GroupLens has not answered: the demo needs a bundle derived from the data.

- [ ] Email GroupLens about D-003 (an answer can take time):

  ```text
  To: grouplens-info@umn.edu
  Subject: Permission request: public non-commercial demo built on MovieLens-20M

  Hello GroupLens team,

  I am a student at the University of Southern California building a non-commercial
  portfolio project, a movie recommender trained on MovieLens-20M. I would like to host
  a public demo on Streamlit Community Cloud and would like to know whether the following
  is permitted:
  1. publishing model artifacts derived from the data (item embeddings, model weights, a
     BM25 index built from titles, genres and tags);
  2. including the titles and genres from movies.csv;
  3. optionally, showing a small sample of anonymised user rating histories.
  The repository does not redistribute the dataset (users download it from grouplens.org),
  cites Harper & Konstan (2015), and implies no endorsement.

  Thank you,
  Yi Zhao
  ```

**Before the sweep** (needed for the step above)

- [ ] A Modal account: `uv tool install modal && modal setup`, then a workspace budget (e.g. \$15) under Settings → Usage.
- [ ] A W&B account and API key, stored for Modal with `modal secret create wandb WANDB_API_KEY=...` (or `WANDB_MODE=offline` to skip W&B).

**Later**

- [ ] Before S6: the GroupLens answer on D-003 (above).

## 8. Risks and cut order

Cut order if late: C2 in ablation (c), then the A0 / A2 / A4 context arms, the ranker variant without EASE, live LLM calls in the demo (cached examples only), Milvus in the demo (a numpy fallback; Milvus stays in the offline parity benchmark), and finally the cloud GPU (train on MPS, ~40 min per run).

| Risk | Mitigation |
|---|---|
| The Liang replication is off (counts differ) | golden and real-data count tests in S1, fixed before S2 |
| EASE misses its calibration figure | metric and split bugs are fixed before any model number is reported |
| The targets are not reached | report against the published anchors; never change the protocol or the metric to chase them |
| macOS only: faiss (under Milvus Lite), torch and LightGBM each bundle libomp; a second OpenMP runtime with its own thread pool segfaults | `shopsage rank` loads only LightGBM's; the tests run LightGBM on one thread; the demo process loads LightGBM before pymilvus or serves without Milvus (S6). Linux uses GNU libgomp |
| The sweep fails or stalls on Modal | finished runs are skipped on a re-run; a 2-epoch trial first; RunPod (Option B) as the fallback |
| Milvus Lite is slow on Community Cloud: probe hybrid p95 176 ms on the host (E005); real queries are heavier (laptop: 48.7 ms vs 1.5 ms for exact numpy / scipy, E008) | S6 measures numpy exact dense top-k and scipy BM25 on the host (same results; FLAT overlap 1.000); if Milvus plus the ranker misses D-013, the demo serves through numpy / scipy and shows Milvus as the alternative engine with its latency |
| The demo exceeds 690 MB | probe peak 482 MB (E005); trim the bundle; Hugging Face Spaces |
| The licence rules out titles or artifacts | the D-003 email now; fallback paths in S6 |

## 9. Definition of done

- Every result comes from a logged run, written `mean ± sd [95% CI]`, under `Liang-SG-strict`.
- Tables (a), (b) and (c) and the ranker table exist, and all four targets are reported against their anchors, met or not.
- The live demo has a measured p95 latency; the README shows the GIF; the W&B report is linked; CI is green on GitHub.
- No claim in the README or the resume line lacks a measurement behind it.
