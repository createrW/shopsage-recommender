# ShopSage Recommender — Project Memory

**English** | [简体中文](memory.zh-CN.md)

**Updated:** 2026-10-24, S2 code done (the sweep is with the owner). Read this first, then [plan.md](plan.md); together they are the source of truth for the next step. This file holds facts, rules and decisions; plan.md holds the steps and their checkboxes.

## 1. The project

- **What:** a two-stage recommender on MovieLens-20M: Two-Tower dense recall and BM25 sparse recall plus their RRF hybrid, a LightGBM re-ranker, optional Claude Haiku query expansion, Milvus Lite as the serving retrieval engine, and a CPU-only Streamlit demo. Every reported number comes from a logged run.
- **Why:** the owner's MLE portfolio flagship; the README is the page reviewers see.
- **Owner:** Yi Zhao (USC), who also does every GitHub push.
- **Where:** `/Users/zy/Project/new_sys/projects/shopsage-recommender`, branch `main`. Commits `3e0f08f` (Day 1) and `05f74c4` (S1) are pushed to `https://github.com/createrW/shopsage-recommender` (CI green); the S2 changes are uncommitted.
- **State:** S0 and S1 done; S2 code done and checked on MPS (E006); the owner runs the Modal sweep, then I log it as E007 and build tables (a); S3 (BM25, Milvus serving index) can proceed meanwhile. Re-planned efficiency-first on 2026-10-23 (plan.md §3).

## 2. Inputs that live outside the repo

| Input | Where | Notes |
|---|---|---|
| Original brief: scope, fixed stack, 10-day plan, targets | pasted into the first working session (2026-10-18) | condensed in §3 |
| Updated scope: Milvus, a BM25 recall path, new targets | pasted on 2026-10-23 | condensed in §3; governs wherever it differs from the original |
| Code standard | `/Users/zy/Project/new_sys/code_standard.md` | layout followed; deviations in design.md §15 |
| Hosting plan | `/Users/zy/Project/new_sys/cloud_hosting_plan_2026-10-17.md` | governs cloud deployment (plan.md §6); a byte-identical copy (same sha256) sits in `/Users/zy/Project/new_sys/cloud_sync/personal_extras/` |

## 3. The specs, condensed

- **Updated scope (2026-10-23), which governs:** MovieLens-20M (Amazon Reviews 2023 Toys optional; cut). Two recall paths, Two-Tower dense and BM25 sparse; a LightGBM ranker; LLM query expansion (Qwen2.5-0.5B via vLLM, GPT-4o-mini or the Claude Haiku API; Haiku chosen). Stack: PyTorch 2 with torchrec or a hand-written two-tower, LightGBM, sentence-transformers BGE-M3, bitsandbytes; FAISS or Milvus Lite, plus Elasticsearch BM25; Polars and DuckDB; W&B or MLflow; Streamlit and a Jupyter EDA notebook; NDCG@10, Recall@50 and HitRate@50 with bootstrap 95% CIs. Targets: Recall@50 ≥ 0.55; NDCG@10 after the ranker ≥ 0.32; end-to-end latency ≤ 200 ms per query (retrieve 50, rank 10); a hard-negative mining ablation showing Recall@50 +0.08 or more. Resources: one RTX 3060/4060 (8–12 GB) for 10–15 h of training, 16 GB of RAM, 15 GB of disk.
- **Original brief (2026-10-18):** the same system with FAISS, a hard-negative round, a Top-500 → LightGBM ranker, LLM expansion, three ablations ((a) negatives, (b) dense vs BM25 vs hybrid, (c) with vs without LLM expansion), a Streamlit demo (a user-id dropdown, that user's last 5 ratings, a Top-10 with scores, an LLM toggle) and a README with tables and CIs. Out of scope: online serving, deep rankers, reinforcement learning, A/B infrastructure. Its targets were Recall@50 0.42 → 0.58 and NDCG@10 0.32.
- **Choices made under efficiency first:** plan.md §3.

## 4. Standing rules

1. **No number without a run.** A metric enters the docs only from a logged run (experiment_log.md Rule 1); result cells read `pending` until then; literature figures are labelled and cited.
2. **Targets are not results.** The owner's targets are reported against their published anchors, met or not.
3. **Licence.** MovieLens data is never committed, because GroupLens forbids redistribution without permission; `data/` is git-ignored, and the public demo ships no per-user rating rows unless D-003 allows it.
4. **The demo never imports torch** (Streamlit Community Cloud: CPU only, 690 MB guaranteed); CI checks this.
5. **The owner pushes.** Agents commit only when asked and never push; git commands run one at a time (a compound `mkdir && git init` was declined on Day 1).
6. **Training is handed off** as one cloud command plus a VRAM table, and the owner runs it.
7. **Cloud deployment follows the hosting plan** (plan.md §6); vendor figures are re-checked before they enter the docs.
8. **Polars, not pandas,** in this project's code (pymilvus imports pandas on its own); DuckDB for SQL over parquet.
9. **Fold-in rule** (Liang protocol): a held-out user's vector and features come only from that user's fold-in items; item statistics, BM25 tags and every trained model use training users only. The Day-1 as-of accessor design applies only to the leave-one-out split, which is no longer used.
10. **Tests first (TDD).** The suite stays green with warnings as errors; ruff, mypy and `uv lock --check` stay clean.
11. **The experiment log is append-only.** Corrections are new entries; a decision row may be revised with a dated note.
12. **Status docs are bilingual.** plan.md and memory.md each have a `.zh-CN.md` twin; update both together.
13. **Working loop.** Read plan.md and memory.md before every step; after it, update both (with their Chinese twins) and tell the owner what to do next.
14. **Efficiency first.** Take the simplest path that measures the targets honestly, and record every cut in plan.md §3. Speed never buys a weaker evaluation: full-corpus ranking, no target leakage, no unmeasured claims.

## 5. Key facts

### The Liang split (measured in S1, E004; the published counts reproduce)

| Fact | Value |
|---|---|
| Preprocessing | rating ≥ 4.0; users with ≥ 5 positives; no item filter: 9,990,682 positives, 136,677 users, 20,720 movies (all as published) |
| Split | 116,677 training users, 10,000 validation and 10,000 test users; 20,108 items in the training catalogue (as published); training rows 8,538,846 |
| Held-out rows | validation fold-in 589,887 / targets 142,514; test fold-in 579,008 / targets 139,775; every held-out user has targets |
| `split_sha256` | `4f93a0e6f6119f20d9aabedc633c82bf37965bf5fbf8a1d2ba72ed94316e4886` |
| Config hash | `2f24b2c594e1` (`configs/data/movielens_20m_liang.yaml`) |
| `meta.json` `created_at` | 2026-10-23T20:06:05+00:00 |
| `prepare` | 5.1 s, 2.2 GB peak RSS on the M5 |

### Results so far (E004, `Liang-SG-strict`, 10,000 test users; mean [95% CI])

| Model | Recall@20 | Recall@50 | NDCG@10 | NDCG@100 | HR@50 |
|---|---|---|---|---|---|
| MostPop | 0.1618 | 0.2352 [0.2310, 0.2396] | 0.1275 [0.1241, 0.1311] | 0.1906 | 0.7425 |
| EASE, λ = 1,000 (chosen on V_A) | 0.3893 | 0.5210 [0.5156, 0.5264] | 0.3280 [0.3231, 0.3331] | 0.4201 | 0.9246 |
| Published (Steck 2019, Table 1): popularity / EASE | 0.162 / 0.391 | 0.235 / 0.521 | — / 0.329 (`Liang-SG-fit`) | 0.191 / 0.420 | — |

- Calibration gate passed: both baselines reproduce the published figures (Recall@50 within 0.0002).
- EASE already meets the NDCG@10 target (0.32); the Recall@50 target (0.55) is 0.029 above EASE. RecVAE's 0.553 is the highest published Recall@50 under this protocol.
- `shopsage evaluate`: 211 s for MostPop plus a three-λ EASE grid, 3.5 GB peak RSS (EASE's Gram and weight matrices are 1.6 GB each in float32). The Recall@50 bootstrap half-width is about ±0.005 at 10,000 users.
- Deployment probe on the laptop (20,108 items, synthetic dense and sparse vectors): hybrid search p95 34–44 ms, dense p95 18–24 ms; Streamlit plus the app peak at about 450 MB.
- **On Streamlit Community Cloud (E005):** import pymilvus 1.9 s, build 7.9 s; dense p50/p95 76.3/135.2 ms, sparse 63.0/94.6 ms, hybrid 118.0/176.3 ms; overlap 1.000; peak RSS 482 MB. Memory fits 690 MB; the hybrid p95 alone takes 176 of the 200 ms budget, so S6 measures a numpy / scipy serving path and falls back to it if needed.
- **Two-Tower check (E006, MPS, not a result):** A1 after 3 epochs, test Recall@50 0.5145 [0.5088, 0.5199], NDCG@10 0.3163, still rising; 129–146 s per epoch on the M5; peak memory A1 2.1 GB, A2 2.2 GB, A4 5.0 GB.

### Day-1 run (raw data, and the leave-one-out split, kept but not used for results)

| Fact | Value |
|---|---|
| Raw data | 20,000,263 ratings (0.5–5.0 stars); 138,493 users (each with ≥ 20 ratings; no demographics); 27,278 movies (26,744 rated); 465,564 tag applications; 1995-01-09 to 2015-03-31 |
| Archive md5 | `cd245b17a1ae2cc31bb14903e1204af3` |
| Day-1 split | 9,977,455 positives; 136,674 users × 13,681 items; leave-one-out with a splitmix64 tie-break; `split_sha256` `5fa6cbe8f702…` (unchanged by an EDA re-run on 2026-10-23); config hash `aec7ddd4dc98` |
| `prepare` | 3.3 s warm / 9.1 s cold, 2.3 GB peak RSS on the M5 (E001) |
| MPS (M5, B = 4,096, bf16) | 37.25 ms per step → 88 s per epoch in-batch; peak 2.30 GB with 4,096 sampled negatives (1.16 GB in-batch); 30 epochs ≈ 44 min (E002) |
| Popularity | the top-100 movies hold 23.5% of positives; Gini 0.90 over the 26,744 rated movies |
| Local disk (2026-10-23) | `data/raw` 1.0 GB, `data/processed` 138 MB, `data/processed/liang` 146 MB, `.venv` 946 MB |
| Tests | 122 passing (2026-10-24), the slow real-data test included |

## 6. Decisions

Current decisions; experiment_log.md's Decisions table holds the full text and revision notes. The protocol in use is `Liang-SG-strict` (experiment_log.md, Protocols).

| ID | Topic | Status | Decision |
|---|---|---|---|
| D-001 | Evaluation protocol | DECIDED 2026-10-23 (the owner started S1 without a veto) | `Liang-SG-strict` is the only headline protocol: an exact replication of Liang et al.'s split (seed 98765, 10,000 validation and 10,000 test users, 80/20 fold-in and targets). Calibration passed (E004). The leave-one-out split and `Liang-SG-fit` are cut. |
| D-002 | Item filter | REVISED 2026-10-23 | No item filter (`min_item_interactions: 1`), Liang's preprocessing. The 5-positive filter applies only to the unused Day-1 split. |
| D-003 | Demo data licence | OPEN, before S6 | default: a visitor "pick 5 movies" profile, titles obtained from GroupLens at startup, no per-user rows; the owner emails GroupLens (plan.md §7) |
| D-004 | User tower | DECIDED 2026-10-23, follows D-001 | inductive: mean of history-item embeddings over ≤ 50 fold-in items plus log history length → MLP → 128-d; no user-ID table |
| D-005 | Hard-negative ablation | REVISED 2026-10-23 | headline: A1 (in-batch + logQ, "without") vs A3 (A1 + per-row negatives mined from ranks 101–500 of A1's seed-42 model, "with"), 3 seeds each; A0 (no logQ), A2 (A1 + 2,048 uniform negatives, MNS) and A4 (full softmax) one seed each, as context |
| D-006 | Ranker data | REVISED 2026-10-23 | V_A / V_B = the 10,000 Liang validation users split 50/50 (seed 42); candidates = dense Top-500 ∪ BM25 Top-100; labels = each user's held-out 20%; ~11 features including a sparse-EASE score; LambdaRank, 3 LightGBM seeds, with and without the EASE feature |
| D-007 | Metrics and CIs | REVISED 2026-10-23 | headline truncated Recall@50 and NDCG@10, plus Recall@20, NDCG@100 and HR@50, on the 10,000 test users; percentile bootstrap (2,000 resamples, seed 42), paired between arms; 3 seeds for each claimed comparison, 1 for context arms, labelled |
| D-008 | Retrieval engine | REVISED 2026-10-23 | offline: exact numpy top-k with fold-in items masked before the cut; serving: Milvus Lite (FLAT; dense, sparse BM25 vectors and RRF hybrid search), flushed after loading, parity-tested against the exact path; FAISS is no longer a direct dependency (Milvus Lite 3.x runs on faiss-cpu) and the ANN study is cut |
| D-009 | Baselines | REVISED 2026-10-23 | MostPop (popularity over training users) and EASE (dense B, λ on V_A) in every table; ItemKNN cut |
| D-010 | BM25 and LLM | REVISED 2026-10-23 | BM25 over title, genres, top genome tags and training-user tags, in scipy (no Elasticsearch); the user query is the fold-in items' text; LLM: `claude-haiku-5-5` with structured output, effort `low`, cached; ablation (c) = ~1,000 tag queries, C0 BM25 raw vs C1 BM25 + expansion (C2 hybrid if time); vLLM, Qwen and GPT-4o-mini not used |
| D-011 | Timestamps | N/A since 2026-10-23 | applied only to the leave-one-out split |
| D-012 | Schedule | REPLACED 2026-10-23 | by steps S1–S7 (plan.md §4) |
| D-013 | Latency target | DECIDED 2026-10-23 | ≤ 200 ms p95 for retrieval (dense + BM25) plus LightGBM re-ranking to 10, measured on the demo host, LLM excluded and reported separately |
| D-014 | Efficiency-first scope | DECIDED 2026-10-23 | cuts listed in plan.md §3; one Modal sweep instead of runs one at a time |

**Superseded until the S7 rewrite:** design.md §4.1, §4.3, §5–§7, §9, §11, §16 and §17, plus the protocol-specific parts of §4.4, §8, §10, §12 and §13 (its banner says so); api_spec.md §5.6, §6.2 and §7 (its §2.7, §6.1 and §9 describe S1). The README was rewritten in S1 and is current. Where an old section disagrees with this one, this one wins.

## 7. Environment and gotchas

- **Laptop:** Apple M5, 16 GB of unified memory, no NVIDIA GPU. uv 0.12.22; the project venv runs Python 3.12.15 with the core and `dev` extras (scipy, pymilvus, milvus-lite, pandas and faiss-cpu included; no torch, no streamlit).
- **Extras since S1:** `train`, `llm`, `demo`, `eda`, `dev`; the `cpu` / `gpu` FAISS extras and their conflicts are gone. `uv sync` is exact: name every extra in one command. `uv run --extra demo ...` adds Streamlit for a one-off run (`make demo`), and a later `uv sync --extra dev` removes it again.
- **Milvus Lite (3.2.2, with pymilvus 3.0.2):** pure Python over faiss-cpu, inside the calling process. FLAT index only; sparse vectors and hybrid search work. **Call `client.flush(collection)` after inserting**: unflushed rows are scanned in Python on every query (dense p50 181 ms at 20,108 items, 9–17 ms after the flush). pymilvus imports pandas (about 1 s warm, 10 s on a cold first import). The docs list Ubuntu ≥ 20.04 and macOS ≥ 11, not Debian 11, so the host is checked by the probe deploy.
- **Headless Streamlit:** `streamlit.testing.v1.AppTest.from_file("src/shopsage_rec/serving/app.py").run()` executes the app in-process; used to measure the probe locally.
- **LightGBM on macOS** needs `brew install libomp` (not installed as of 2026-10-23); Linux and Streamlit Cloud use `libgomp1`.
- **torch on Linux** comes from the cu126 index (NVIDIA driver ≥ 525); Blackwell GPUs need the cu130 index.
- **AMP:** bf16 only when `torch.cuda.is_bf16_supported(including_emulation=False)` (compute capability ≥ 8.0); a T4 uses fp16 with `GradScaler`; MPS ran bf16 autocast.
- **MPS has no float64:** tensors moved to MPS must be float32 (the item-frequency vector for logQ hit this).
- **Modal 1.6:** `Image.uv_sync` installs dependencies only (no project), so the source is added with `add_local_dir` and `PYTHONPATH`; GPU names are `"T4"`, `"L4"`, `"A10"`, `"L40S"`, `"A100"`, … (no `"A10G"` in the docs as of 2026-10-24); Volumes need `commit()` after writes and `reload()` before reads.
- **Tests that check `sys.modules`** must run in a subprocess: other tests in the session import torch.
- **MPS timing:** fewer than 30 timed steps is too noisy (a 3-step run looked 30% faster than it is).
- **Polars `is_in`:** testing a column against a Series of the same dtype is deprecated; use `.is_in(series.implode())`.
- **Docker:** the daemon was not running on 2026-10-18 or on 2026-10-23, so the image has never been built locally; only static checks have passed.
- **Git:** no global identity is configured; the Day-1 commit set the author per command. `gh` is authenticated for `createrW/shopsage-recommender` (CI status via `gh run list`).
- **Streamlit Community Cloud** installs from the first dependency file it finds, the entrypoint's directory before the repo root, in the order `uv.lock` > `Pipfile` > `environment.yml` > `requirements.txt` > `pyproject.toml`; hence `src/shopsage_rec/serving/requirements.txt` (regenerate with `make demo-requirements`; CI fails when it drifts from `uv.lock`).
- **`claude-haiku-5-5`:** \$0.10 / \$0.50 per million input / output tokens for prompts up to 100K tokens; structured output through `output_config.format`; adaptive thinking on by default and effort `medium` by default, so set `low`; since 2026-10-21 any `temperature`, `top_p` or `top_k` returns a 400; no server-side refusal fallback, so a `refusal` stop reason is handled in client code (claude-api skill, cached 2026-10-06).
- **Vendor figures, as checked:** Streamlit Community Cloud (2026-10-18): 690 MB guaranteed, 2.7 GB maximum, ≤ 2 cores, Python 3.12 by default, Debian 11, sleeps after 12 h idle. Modal (2026-10-23): Starter includes \$30 of credit per month; A10 \$1.10/h, T4 \$0.59/h, L4 \$0.80/h, A100 40 GB \$2.10/h; workspace budgets. Hugging Face (2026-10-23): CPU basic, 2 vCPU and 16 GB, free; T4 small \$0.40/h. RunPod (2026-10-23): RTX 4090 \$0.34/h on Community Cloud, \$0.89/h on Secure Cloud; idle volume disk \$0.20 per GB-month.

## 8. Where things are

| Path | Contents |
|---|---|
| `README.md` | front page: results with the measured baselines, architecture, quickstart, cloud training and VRAM, dataset, experiments, demo, design rationale, limitations, roadmap |
| `docs/experiment_log.md` | rules, protocols, TARGETS, decisions D-001 to D-014, run matrix, entries E001–E006 |
| `docs/design.md`, `docs/api_spec.md` | the Day-1 design and the artifact / CLI contract (v0.4), partly superseded (§6) |
| `docs/plan.md`, `docs/memory.md` | the step plan and status; this memory (each with a `.zh-CN.md` twin) |
| `src/shopsage_rec/cli.py`, `config.py` | the Typer CLI (`prepare`, `stats`, `evaluate`; defaults follow the Liang split); `DataConfig`, `SplitConfig` (`leave_one_out` or `liang`) |
| `src/shopsage_rec/data/` | `loader`, `preprocess`, `split` (Day-1), `liang` (`liang_split`, `load_liang`, `partition_val_users`), `pipeline` (`run_prepare` for both strategies) |
| `src/shopsage_rec/evaluation/` | `metrics` (masked top-k, Recall / NDCG / HR), `bootstrap`, `runner` (`evaluate_baselines`) |
| `src/shopsage_rec/models/baselines.py` | `MostPop`, `gram_matrix`, `EASE` |
| `src/shopsage_rec/serving/` | `app.py` (Streamlit entry point: the probe), `probe.py` (`run_probe`), `requirements.txt` (generated) |
| `src/shopsage_rec/features/items.py` | item-tower inputs: genres, year bucket, genome PCA-64, cached in `<processed>/item_features.npz` |
| `src/shopsage_rec/models/` | `baselines.py`; S2: `two_tower.py` (`TwoTower`, `sample_history`), `losses.py` (`sampled_softmax`, `PositiveSet`), `numpy_tower.py` (the torch-free user tower), `trainer.py` (`TrainConfig`, `ARMS`, `train_two_tower`), `mining.py`, `sweep.py` (the job plan, pure Python) |
| `src/shopsage_rec/{retrieval,ranking,llm}/` | placeholders holding only a docstring (S3–S5) |
| `configs/data/` | `movielens_20m_liang.yaml` (default, hash `2f24b2c594e1`) and `movielens_20m.yaml` (Day-1, hash `aec7ddd4dc98`) |
| `configs/train/lgbm_ranker.yaml` | ranker draft (S4); the Two-Tower has no YAML: `TrainConfig` and CLI options, snapshotted in each `metrics.json` |
| `configs/deploy/streamlit.yaml` | demo config draft: bundle, artifacts, Milvus Lite retrieval, LLM expansion |
| `scripts/` | `download_data.sh`, `prepare_data.sh` (default: the Liang config), `bootstrap_cloud.sh` (train + llm extras, the Liang split), `bench_two_tower_step.py`, `train.sh` (sequential sweep), `modal_train.py` (the Modal sweep: Volume `shopsage-data`, GPU `A10`) |
| `tests/` | 122 tests, among them `test_two_tower.py`, `test_trainer.py`, `test_sweep.py`, `test_item_features.py`, `test_liang.py` (with the real-data count test), `test_liang_pipeline.py`, `test_metrics.py`, `test_baselines.py`, `test_probe.py` |
| `notebooks/01_eda.ipynb` | the EDA on the Day-1 split, executed; figures in `docs/figures/` |
| `Makefile` | 22 targets (`make help`), including `evaluate` and `demo-requirements` |
| `.github/workflows/ci.yml` | ruff, ruff format, mypy, pytest; the demo requirements without torch; `serving/requirements.txt` against `uv.lock`; the Milvus Lite probe on Linux |
| `Dockerfile`, `.dockerignore` | CPU demo image (never built yet); the Hugging Face Spaces fallback |
| `.streamlit/`, `packages.txt` | Community Cloud settings, secrets template, `libgomp1` |
| `data/`, `artifacts/` (git-ignored) | `data/raw/ml-20m/`, `data/processed/` (Day-1), `data/processed/liang/`; `artifacts/eval/local-20261023-1306/` (E004); `artifacts/runs/` (E006 checks: `smoke-A1-mps`, `check-A1-full3-mps`) |

## 9. Resuming

1. Read plan.md and this file (rule 13); continue at the first unchecked step in plan.md §4.
2. Check the workspace:

```bash
cd /Users/zy/Project/new_sys/projects/shopsage-recommender
git status && git log --oneline -3
uv sync --extra dev                              # core + dev; add --extra train for torch on MPS
make check                                       # ruff, mypy and the fast tests
uv run shopsage stats | grep split_sha256        # the Liang split: 4f93a0e6f611...
```

3. After the step: tick plan.md, update this file (facts, decisions, history), mirror both into the `.zh-CN.md` twins, log runs in experiment_log.md, then report the owner's next actions.

## 10. History

| Date | Event |
|---|---|
| 2026-10-17 | the owner writes the hosting plan |
| 2026-10-18 | Day 1 built: scaffold, data pipeline, split, EDA, docs, CI; MPS benchmark (E002); a final review (104 findings) resolved, with D-005 and D-006 revised; commit `3e0f08f` (21:13 PDT) |
| 2026-10-23 | plan.md and memory.md written in English and Chinese; the hosting plan referenced at its new path; vendor prices re-checked |
| 2026-10-23 | the owner's updated scope (Milvus, a BM25 recall path, new targets) and two rules (efficiency first; the working loop); re-planned into steps S1–S7 with the Liang protocol as the default |
| 2026-10-23 | the owner pushed Day 1 to `createrW/shopsage-recommender` (CI green) and said "go S1" |
| 2026-10-24 | the owner pushed S1 as `05f74c4`; CI green, including the Milvus Lite probe on Linux |
| 2026-10-24 | the owner deployed the probe on Community Cloud (E005) and said "go s2"; S2 code done: Two-Tower A0–A4, mining, sweep plan, Modal script; MPS checks (E006); 122 tests green |
| 2026-10-23 | S1 done: the Liang split reproduces the published counts; `shopsage evaluate`; MostPop and EASE reproduce their published Recall@50 (E004); FAISS out, Milvus Lite in; the Milvus Lite probe measured on the laptop; README and experiment_log synced |
