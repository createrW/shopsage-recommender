# Experiment Log

Append-only record of ShopSage runs and decisions. The design rationale is in [design.md](design.md); artifact schemas, metric definitions and the environment are in [api_spec.md](api_spec.md).

## Rules

1. **No number without a run.** A metric enters this log only with its run id, config hash and artifact path. Literature figures live in design.md §9; the TARGETS table quotes only the anchors behind each label, and decision rows cite the Day-1 EDA facts they rest on.
2. **Targets are labelled.** The owner's targets appear only in the TARGETS table. They are not results.
3. **Placeholders.** Every result cell reads `pending` until its run has finished and been checked.
4. **Append, don't edit.** A correction is a new entry that names the entry it supersedes. The old entry gains one line, "superseded by E0xx", and nothing else in it changes. An entry whose outputs are still `pending` is completed in place. The rule governs entries (E0xx); the Decisions table records the current decision, and a revised row says so in its status and ends with a dated revision note.
5. **Seeds and CIs.** Every trained arm runs at least 3 training seeds. Metrics are written `mean ± sd [lo, hi]` with the number of users: mean ± sd across seeds, and [lo, hi] the 95% percentile bootstrap over users (2,000 resamples, numpy Generator seeded 42). Arms are compared only through the paired bootstrap Δ, with the same resample indices for both arms. A difference is claimed only if the paired user-bootstrap CI excludes 0 and the per-seed paired Δs (seed k of arm A vs seed k of arm B, same users) all have the CI's sign; tables also report the per-seed Δ range. Definitions and procedure: api_spec.md §8 (D-007).
6. **Protocol named.** Every result table states its protocol: `LOO-full`, `LOO-asof`, `Liang-SG-strict` or `Liang-SG-fit` ([Protocols](#protocols), D-001), or, for ablation (c), the tag-as-query benchmark defined with it.
7. **Test once.** Model selection uses validation users only: V_A for retriever settings, V_B for ranker labels (D-006). Test metrics are computed once per final configuration, and every test run is logged, whatever it shows.

## Protocols

The evaluation code (`shopsage evaluate`, `src/shopsage_rec/evaluation/`) landed in S1; metric definitions are in api_spec.md §8. Since the 2026-10-23 re-plan only `Liang-SG-strict` is in use (D-001): the `LOO-*` labels describe the Day-1 split, which is kept but not evaluated, and `Liang-SG-fit` is cut.

| Label | Split | Evaluated users | Ranked against | Headline metrics |
|---|---|---|---|---|
| `LOO-full` | Day-1 leave-one-out by (`timestamp`, `tie_key`), `splits/` | every user, at the val event (V_A or V_B) or at the test event | all items except the user's history positives (train; train + val at test) and anything the user rated before the cutoff | HR@{10, 50, 100, 500}, NDCG@10, MRR@10 |
| `LOO-asof` | same | same | as `LOO-full`, and also without items whose first rating in `ratings.parquet` is later than the cutoff (the catalogue as of the event) | same |
| `Liang-SG-strict` | strong generalisation (Liang et al. 2018, §4.3): held-out users are never trained on; models train on the training users' full histories; `min_item_interactions: 1`. Built in S1 (`configs/data/movielens_20m_liang.yaml`, E004) | held-out users; a random 20% of each one's positives are the targets | all items except the user's fold-in positives | Recall@20 and Recall@50 (truncated), NDCG@100, NDCG@10; the Recall@20/50 and NDCG@100 anchors (design.md §9.2) are from this variant |
| `Liang-SG-fit` | strong generalisation in the fold-in-in-training variant: the held-out users' fold-in 80% is part of the training data (Milogradskii et al., RecSys 2024); `min_item_interactions: 1`. Cut on 2026-10-23 (D-014); kept as the source of the NDCG@10 anchors | same as `Liang-SG-strict` | same | same; the NDCG@10 anchors (design.md §9.2) are from this variant |

- `LOO-asof` exists because a per-user split admits global look-ahead (Ji et al., TOIS 2023): averaged over test users, 53.7% of all train rows are newer than that user's own test event, and for the median test event 56.8% of the kept catalogue had no rating yet in `ratings.parquet` (Day-1 measurement). Both `LOO-*` variants are reported (D-001).
- Accepted mild global leaks, documented rather than removed: the tag genome (computed over all time) and the global item filter (counted over all positives).
- The leave-one-out literature counts every rating as an interaction over 26,744 items. `LOO-full` ranks positives over the filtered catalogue, so those figures are context, not like-for-like (design.md §9).

## Recording an entry

| Field | Content |
|---|---|
| Entry | `E` + three digits, sequential |
| Date | YYYY-MM-DD |
| Run id | W&B run id, or `local-YYYYMMDD-HHMM` for runs without W&B |
| W&B | run URL, or `n/a` |
| Code | git commit SHA once the repository is under version control; until then `n/a (pre-git)` |
| Config | path(s) and config hash (below) |
| Data | `created_at`, `split_sha256` (first 12 hex characters), `n_users` and `n_items` from the `splits/meta.json` used |
| Protocol | `LOO-full`, `LOO-asof`, `Liang-SG-strict` or `Liang-SG-fit` (Rule 6), and the user subset (V_A, V_B or test) |
| Hardware | laptop (M5, 16 GB; MPS for training), or GPU model and VRAM |
| Seeds | training seeds (at least 3 per trained arm); bootstrap seed 42 |
| Cost | wall-clock, GPU-hours, API tokens and cost where relevant |
| Peak memory | `torch.cuda.max_memory_allocated()` for CUDA runs; the peak of `torch.mps.driver_allocated_memory()` for MPS runs; peak RSS for CPU runs |
| Results | metric table, `mean ± sd [lo, hi]` and n |
| Δ vs reference | paired bootstrap Δ with its 95% CI against a named reference entry and arm, and the per-seed paired Δ range (Rule 5) |
| Notes / decision | what changed, what was learned, what happens next |

**Config hash:** the first 12 hex characters of the SHA-256 of the YAML's parsed content, serialised as JSON with sorted keys, so that comments and key order do not change it. PyYAML is a core dependency, so `uv run` provides it (a bare system Python may lack it). If the CLI later prints its own hash, that value takes precedence.

```bash
uv run python -c "import hashlib, json, sys, yaml; print(hashlib.sha256(json.dumps(yaml.safe_load(open(sys.argv[1])), sort_keys=True).encode()).hexdigest()[:12])" configs/data/movielens_20m.yaml
```

### Entry template

```markdown
### E0xx — YYYY-MM-DD — <title>

| Field | Value |
|---|---|
| Run id / W&B | pending |
| Code | <git SHA, or `n/a (pre-git)`>; `shopsage_rec` <version> |
| Config | configs/<group>/<name>.yaml @ <hash> |
| Data | meta.json created_at <...>; split_sha256 <12 hex>; n_users <...>; n_items <...> |
| Protocol / users | LOO-full / test |
| Hardware / seeds | <...> / training seeds <s1, s2, s3>; bootstrap seed 42 |
| Cost / peak memory | pending |

| Metric | mean ± sd [95% CI] | n users |
|---|---|---|
| HR@10 | pending | pending |
| HR@50 (= Recall@50 under LOO) | pending | pending |
| HR@100 | pending | pending |
| HR@500 | pending | pending |
| NDCG@10 | pending | pending |
| MRR@10 | pending | pending |

Δ vs E0yy <arm> (paired bootstrap, same resample indices; per-seed Δ range): pending
Notes:
```

`Liang-SG-strict` and `Liang-SG-fit` entries replace the metric rows with Recall@20, Recall@50 (truncated), NDCG@100 and NDCG@10.

## TARGETS (targets, not results)

The owner's updated scope (2026-10-23) set four targets, which replace the brief's Recall@50 0.42 → 0.58 and NDCG@10 0.32. Each is compared with published figures under `Liang-SG-strict` and, from E004 on, with in-repo measurements. Nothing here is a result of this project's models until a logged run measures it.

| Metric | Target | Published anchor (`Liang-SG-strict` unless noted) | In-repo reference | Assessment |
|---|---|---|---|---|
| Recall@50 (truncated) | ≥ 0.55 | EASE 0.521, Mult-VAE 0.537, RecVAE 0.553 (the highest verified) | EASE 0.5210 [0.5156, 0.5264] (E004) | **not met:** system (ranker) 0.5420 ± 0.0006 (E009), dense A1 0.5361 (E007); 0.008 short |
| NDCG@10 after the ranker | ≥ 0.32 | EASE 0.329 (`Liang-SG-fit`, Milogradskii et al. 2024) | EASE 0.3280 [0.3231, 0.3331] (E004) | **met:** 0.3428 (E009), +0.0148 [+0.0126, +0.0171] over EASE and +0.0074 over the dense order |
| Latency, retrieval and ranking to 10 | ≤ 200 ms p95 on the demo host | — | S1 probe: Milvus Lite hybrid p95 34 ms on the laptop (E004), 176 ms on the host (E005); real queries: Milvus hybrid p95 48.7 ms vs exact numpy / scipy 1.5 ms on the laptop (E008) | at risk with Milvus as the online path; comfortable with the exact path; decided on the host in S6 |
| Hard-negative ablation | Recall@50 +0.08 (A3 − A1) | — | — | ambitious, since A1 already has logQ and accidental-hit masking; **measured −0.0020 [−0.0033, −0.0007]: not met** (E007) |

Sources: [Liang et al. 2018](https://arxiv.org/abs/1802.05814) (Mult-VAE); [Steck 2019](https://arxiv.org/abs/1905.03375), Table 1 (popularity, EASE); Shenbin et al., WSDM 2020 (RecVAE); Milogradskii et al., RecSys 2024 (NDCG@10, fold-in-in-training variant).

## Decisions

Revised on 2026-10-23 by the efficiency-first re-plan, which followed the owner's updated scope; the owner started S1 without vetoing its defaults. docs/memory.md §6 mirrors this table.

| ID | Topic | Status | Due | Decision or recommendation |
|---|---|---|---|---|
| D-001 | Evaluation protocol | DECIDED (2026-10-23) | — | `Liang-SG-strict` is the only headline protocol: an exact replication of Liang et al.'s split (seed 98765; 10,000 validation and 10,000 test users; a random 20% of each held-out user's items as targets; items limited to the 20,108 in the training catalogue). Truncated Recall@20/50, NDCG@10/100 and HR@50 on the 10,000 test users with paired bootstrap CIs. In-repo MostPop and EASE reproduce their published Recall@50 (E004). The Day-1 leave-one-out split is kept but not evaluated; `Liang-SG-fit` is cut. Decided 2026-10-23: the targets only mean something under this protocol, and a random holdout needs no as-of machinery. |
| D-002 | Item filter | DECIDED (Day 1; revised 2026-10-23) | — | No item filter (`min_item_interactions: 1`), Liang's preprocessing: 9,990,682 positives from 136,677 users over 20,720 movies, 20,108 of them in the training catalogue (E004). The single-pass ≥ 5 filter remains only in the unused Day-1 split. Revised 2026-10-23: follows D-001. |
| D-003 | Demo data licence | OPEN | before S6 | MovieLens may not be redistributed. Until GroupLens grants permission the public demo ships no per-user rating rows (visitors build a "pick 5 movies" profile) and obtains titles from GroupLens at startup. The delivery path is decided (a sha256-pinned GitHub Release asset, api_spec.md §9.5); what the asset may contain is not. |
| D-004 | User tower | DECIDED (2026-10-23) | — | Inductive: the mean of history-item embeddings over ≤ 50 fold-in items plus log history length → MLP → 128-d, L2-normalised; no user-ID table, since held-out users are never trained on. Visitor profiles in the demo go through the same path. Decided 2026-10-23: follows D-001. |
| D-005 | Hard-negative ablation | DECIDED (Day 1; revised 2026-10-23) | — | Headline: A1 (in-batch + logQ, "without") vs A3 (A1 + per-row negatives mined from ranks 101–500 of A1's seed-42 model, "with"), 3 seeds each, paired bootstrap on Recall@50. Context arms with one seed each: A0 (no logQ), A2 (A1 + 2,048 uniform negatives, MNS), A4 (full softmax over 20,108 items). Accidental hits masked; popularity and masks from training users only. Revised 2026-10-23: the updated scope asks for a hard-negative mining ablation (target +0.08 Recall@50), so mined negatives become the headline and MNS a context arm. |
| D-006 | Ranker data | DECIDED (Day 1; revised 2026-10-18 and 2026-10-23) | — | V_A / V_B = the 10,000 Liang validation users split 50/50 by seed 42 (V_A: model selection; V_B: ranker labels, 10% for early stopping). Candidates = dense Top-500 ∪ BM25 Top-100; labels = each user's held-out items; ~11 features including a sparse-EASE score (top-100 neighbours per item); LambdaRank on NDCG@10, 3 LightGBM seeds, reported with and without the EASE feature. Missed positives are never injected. Revised 2026-10-23: two recall paths (the updated scope) and the Liang protocol. |
| D-007 | Metrics and CIs | DECIDED (Day 1; revised 2026-10-23) | — | Headline truncated Recall@50 and NDCG@10, plus Recall@20, NDCG@100 and HR@50, over the 10,000 test users (all have targets, E004). Percentile bootstrap over users, 2,000 resamples, seed 42, chunks of 100 (`evaluation/bootstrap.py`), paired between arms; the Recall@50 half-width is ~±0.005 at 10,000 users (E004). 3 seeds for each claimed comparison, 1 for context arms, labelled. Revised 2026-10-23: Liang protocol, efficiency first. |
| D-008 | Retrieval engine | DECIDED (Day 1; revised 2026-10-23) | — | Offline: exact top-k with fold-in items set to −inf before the cut, ties to the lower item index (`evaluation/metrics.py`). Serving: Milvus Lite (FLAT index; dense vectors, sparse BM25 vectors and RRF hybrid search), flushed after loading, parity-tested against the exact path (S3). FAISS as a direct dependency and the ANN study are dropped: Milvus Lite 3.x is FLAT-only and runs faiss-cpu internally. Revised 2026-10-23: Milvus is in the updated scope; the S1 probe measured dense and hybrid p95 of 18 and 34 ms at 20,108 items on the laptop (E004). Note 2026-10-24 (E008): on real data Milvus Lite returns exactly the offline answers on all three paths (1,000 users), but exact numpy / scipy search is 16–40× faster on the laptop (hybrid p95 1.5 vs 48.7 ms); the demo's online path is chosen in S6 from host measurements, Milvus kept at least as the shown, parity-checked engine. |
| D-009 | Baselines | DECIDED (Day 1; revised 2026-10-23) | — | MostPop (popularity over training users) and EASE (λ chosen on V_A) in every table; ItemKNN cut. EASE at 20,108 items: Gram and weight matrices 1.6 GB each in float32; the λ grid ran at 3.5 GB peak RSS on the laptop (E004). Revised 2026-10-23: the Liang protocol replaces as-of popularity. |
| D-010 | BM25 and LLM evaluation | DECIDED (Day 1; revised 2026-10-23) | — | BM25 item documents: title, genres, top genome tags and tags from training users only; the user query is the fold-in items' terms; scipy.sparse BM25 offline, the same weights as Milvus sparse vectors online; no Elasticsearch. Hybrid: RRF (k = 60). LLM: `claude-haiku-5-5`, structured output, effort `low`, no sampling parameters, responses cached by sha256(model id + prompt + params). Ablation (c): ~1,000 tag-as-query events, C0 BM25 on the raw tag vs C1 on the expansion (C2 hybrid if time). Cut: vLLM / Qwen, GPT-4o-mini, the convex-mix hybrid. Revised 2026-10-23: efficiency first. |
| D-011 | Timestamps | NOT APPLICABLE (2026-10-23) | — | The session-timestamp findings (51.6% of users rated their whole positive history within an hour; 24.2% have a tie at a split boundary) concern the Day-1 leave-one-out split, which is no longer evaluated. The Liang split is a random holdout and ignores time, a stated limitation. |
| D-012 | Schedule | REPLACED (2026-10-23) | — | Replaced by steps S1–S7 in docs/plan.md §4. |
| D-013 | Latency target | DECIDED (2026-10-23) | — | ≤ 200 ms p95 for retrieval (dense + BM25) plus LightGBM re-ranking to 10, measured on the demo host; the LLM path is reported separately. |
| D-014 | Efficiency-first scope | DECIDED (2026-10-23) | — | Cut: the as-of accessor for the leave-one-out split, the `LOO-*` and `Liang-SG-fit` tables, BGE-M3, the user-ID arm, subgroup diagnostics, ItemKNN, the ANN study, vLLM, MLflow, Elasticsearch, Amazon Reviews, building the Docker image. All training runs go through one Modal sweep (docs/plan.md §3). |

## Planned run matrix

Every result cell reads `pending` until its run is logged. All runs use `Liang-SG-strict` on the 10,000 test users, with selection on V_A. Arms in a claimed comparison run 3 seeds; context arms run 1 seed and are labelled. The MostPop and EASE rows come from E004.

### Milestones

| Tag | Step | Run | Readout | Status |
|---|---|---|---|---|
| P1-calib | S1 | MostPop and EASE (λ on V_A) | Recall@20/50, NDCG@10/100, HR@50 against the published figures | done (E004) |
| P2-sweep | S2 | Two-Tower sweep on Modal: a τ/lr check on A1, then A0–A4 | test metrics per arm and seed; peak VRAM | planned |
| P3-ablations | S3 | tables (a) and (b) | paired Δs with CIs | planned |
| P4-rank | S4 | LightGBM over dense Top-500 ∪ BM25 Top-100 (D-006) | NDCG@10, system Recall@50 and HR@50; offline CPU latency | planned |
| P5-llm | S5 | ablation (c) | HR@10, HR@50, grounding rate, latency, cost | planned |
| P6-demo | S6 | the demo on Streamlit Community Cloud | p50 / p95 latency and memory on the host | planned |

### Ablation (a): hard-negative mining (S2–S3, D-005)

| Arm | Negatives | Seeds | Recall@50 | NDCG@10 | Δ Recall@50 vs A1 [95% CI] |
|---|---|---|---|---|---|
| MostPop | anchor | — | 0.2352 | 0.1275 | — |
| EASE | anchor | — | 0.5210 | 0.3280 | — |
| A0 | in-batch, no logQ | 1 | 0.3111 | 0.1413 | −0.2252 [−0.2297, −0.2203] (E007) |
| A1 (ref, "without") | in-batch + logQ | 3 | 0.5361 ± 0.0005 | 0.3351 ± 0.0005 | — (E007) |
| A2 | A1 + 2,048 uniform negatives (MNS) | 1 | 0.5342 | 0.3332 | −0.0021 [−0.0045, +0.0003] (E007) |
| A3 ("with") | A1 + negatives mined from ranks 101–500 of A1 seed 42 | 3 | 0.5341 ± 0.0008 | 0.3351 ± 0.0011 | −0.0020 [−0.0033, −0.0007] (E007) |
| A4 | full softmax over 20,108 items | 1 | 0.5333 | 0.3335 | −0.0030 [−0.0053, −0.0007] (E007) |

### Ablation (b): recall paths (S3, D-010)

| Arm | Retriever | Recall@50 | NDCG@10 | Δ Recall@50 vs B1 [95% CI] |
|---|---|---|---|---|
| MostPop | anchor | 0.2352 | 0.1275 | — |
| EASE | anchor | 0.5210 | 0.3280 | — |
| B0 | BM25 over item documents | 0.2529 | 0.1556 | −0.2833 [−0.2891, −0.2772] (E008) |
| B1 (ref) | dense, best arm of (a): A1, 3 seeds | 0.5361 ± 0.0005 | 0.3351 ± 0.0005 | — (E008) |
| B2 | hybrid: RRF of B0 and B1 (k = 60) | 0.4647 ± 0.0002 | 0.2789 ± 0.0007 | −0.0715 [−0.0748, −0.0685] (E008) |

### Ablation (c): LLM query expansion (S5, D-010)

Tag-as-query known-item retrieval: ~1,000 (user, tag) events, relevant = the movies that user tagged with the tag; the user's own tags are excluded from item documents; cluster bootstrap over users.

| Arm | Query path | HR@10 | HR@50 | Δ HR@50 vs C0 [95% CI] | Grounding rate | Latency p50 / p95 | Cost per 1K queries |
|---|---|---|---|---|---|---|---|
| C0 (ref) | BM25 on the raw tag | pending | pending | — | — | pending | — |
| C1 | BM25 on the `claude-haiku-5-5` expansion | pending | pending | pending | pending | pending | pending |
| C2 (if time) | hybrid of C1 and the dense path | pending | pending | pending | pending | pending | pending |

## Entries

### E001 — 2026-10-18 — Day 1: data preparation (MovieLens-20M)

Status: **recorded**. Outputs below were copied from `splits/meta.json` of the final Day-1 run and from the DuckDB queries; every sanity check was executed and ticked. The run reproduces bit-for-bit (`split_sha256`) across a second seed-42 run, a scratch-directory run and the notebook's own `run_prepare` call.

| Field | Value |
|---|---|
| Run id / W&B | `local-20261018-1938` (local time of `created_at`) / n/a: no W&B for data prep |
| Code | `n/a (pre-git)`; `shopsage_rec` 0.1.0 |
| Config | `configs/data/movielens_20m.yaml` @ `aec7ddd4dc98` (snapshot below; `meta.json` stores the resolved `DataConfig` under `config`, with absolute `raw_dir` and `processed_dir`) |
| Data | `meta.json` `created_at` 2026-10-19T02:38:18+00:00; `split_sha256` `5fa6cbe8f702`; `n_users` 136,674; `n_items` 13,681 |
| Commands | `uv run shopsage prepare --config configs/data/movielens_20m.yaml`, then `uv run shopsage stats --processed-dir data/processed`; the final `meta.json` (`created_at` above) was written by the notebook's `run_prepare` call (`make eda`), with the same `split_sha256` |
| Split | `leave_one_out`, seed 42; ties broken by `tie_key` (api_spec.md §5.4) |
| Hardware | laptop: Apple M5, 16 GB |
| Wall-clock / peak RSS | 3.3 s warm cache (9.1 s cold), 2.3 GB peak RSS (`/usr/bin/time -l`, Polars 2.0.0) |

**Inputs** (verified at download; they agree with the GroupLens README). Archive md5 `cd245b17a1ae2cc31bb14903e1204af3`.

| File | Rows | Content |
|---|---|---|
| `ratings.csv` | 20,000,263 | userId, movieId, rating (0.5–5.0, half-star), timestamp; 1995-01-09 to 2015-03-31 |
| `movies.csv` | 27,278 | movieId, title, pipe-separated genres; "(no genres listed)" sentinel |
| `tags.csv` | 465,564 | userId, movieId, tag, timestamp |
| `links.csv`, `genome-scores.csv` (11.7M rows), `genome-tags.csv` | — | not converted by `prepare`; the genome is used from Day 2 (item features, BM25 documents; D-010) |

Raw users: 138,493. Distinct rated movies: 26,744.

**Config snapshot**

| Key | Value |
|---|---|
| `min_rating` | 4.0 |
| `min_item_interactions` | 5 |
| `min_user_interactions` | 5 |
| `split.strategy` | `leave_one_out` |
| `split.seed` | 42 |

**Outputs.** Every key the code writes to `splits/meta.json` (meaning in api_spec.md §3), plus the query results.

| Key or quantity | Value | Source |
|---|---|---|
| `strategy`, `seed` | `leave_one_out`, 42 | meta.json |
| `tie_break` | `splitmix64(((user << 32) \| item) xor splitmix64(seed=42)); key from precomputed column 'tie_key'` | meta.json |
| `dataset`, `config` | `movielens-20m`; config matches the snapshot above | meta.json |
| `n_ratings_raw` (expected 20,000,263) | 20,000,263 | meta.json; query 1 |
| `n_movies` (expected 27,278), `n_tags` (expected 465,564) | 27,278; 465,564 | meta.json |
| Positives (rating ≥ 4.0) before filtering | 9,995,410 (49.98% of ratings) | query 2 |
| `n_users` | 136,674 (98.7% of 138,493) | meta.json |
| `n_items` | 13,681 (51.2% of the 26,744 rated movies) | meta.json |
| `n_interactions` | 9,977,455 (the filters remove 0.18% of the positives) | meta.json |
| `n_train` | 9,704,107 (median 36 per user, p10 10, p90 169) | meta.json |
| `n_val`, `n_test` (each = `n_users` by construction) | 136,674; 136,674 | meta.json |
| `density` | 0.005336 (0.5336%) | meta.json |
| `users_with_boundary_tie`, `boundary_tie_fraction` | 20,580; 0.1506 | meta.json; query 4 |
| `users_with_val_train_tie` | 21,139 | meta.json; query 4 |
| `users_with_any_boundary_tie`, `any_boundary_tie_fraction` | 33,053; 0.2418 | meta.json; query 4 |
| `cold_test_items`, `cold_val_items` | 0; 0 | meta.json; query 5 |
| `split_sha256` | `5fa6cbe8f7027e8c4724d592857a0c4134821b33eef982e98d849aa88898c22e`; snippet prints `True` | meta.json; recomputed by the snippet below |
| `created_at`, `versions` | 2026-10-19T02:38:18+00:00; shopsage_rec 0.1.0, polars 2.0.0, python 3.12.15 | meta.json |
| Items left below 5 positives by the single pass | 1 | query 3 |
| Users with more than 500 train positives, and the maximum (D-008) | 1,293; max 3,148 | query 8 |

```sql
-- DuckDB, run from the repository root. Each comment states what the result must equal.
-- 1. raw rows kept: n_ratings_raw
SELECT count(*) AS n_ratings_raw FROM 'data/processed/ratings.parquet';

-- 2. positives before filtering
SELECT count(*) AS positives_before_filter FROM 'data/processed/ratings.parquet' WHERE rating >= 4.0;

-- 3. items left below 5 positives by the single-pass filter
SELECT count(*) AS items_below_5 FROM (
  SELECT item_idx FROM (
    SELECT item_idx FROM 'data/processed/splits/train.parquet'
    UNION ALL SELECT item_idx FROM 'data/processed/splits/val.parquet'
    UNION ALL SELECT item_idx FROM 'data/processed/splits/test.parquet'
  ) GROUP BY item_idx HAVING count(*) < 5
);

-- 4. boundary ties and per-user order: the three tie counts in meta.json,
--    users_in_all_three = n_users, order_violations = 0, min_train_rows >= 3
WITH b AS (
  SELECT t.user_idx, t.timestamp AS t_test, v.timestamp AS t_val, tr.t_train, tr.n_train
  FROM 'data/processed/splits/test.parquet' AS t
  JOIN 'data/processed/splits/val.parquet' AS v USING (user_idx)
  JOIN (
    SELECT user_idx, max(timestamp) AS t_train, count(*) AS n_train
    FROM 'data/processed/splits/train.parquet' GROUP BY user_idx
  ) AS tr USING (user_idx)
)
SELECT
  count(*)                                                  AS users_in_all_three,
  count(*) FILTER (WHERE t_test = t_val)                    AS users_with_boundary_tie,
  count(*) FILTER (WHERE t_val = t_train)                   AS users_with_val_train_tie,
  count(*) FILTER (WHERE t_test = t_val OR t_val = t_train) AS users_with_any_boundary_tie,
  count(*) FILTER (WHERE t_test < t_val OR t_val < t_train) AS order_violations,
  min(n_train)                                              AS min_train_rows
FROM b;

-- 5. cold held-out items: cold_test_items, cold_val_items
SELECT
  (SELECT count(DISTINCT item_idx) FROM 'data/processed/splits/test.parquet' AS h
     ANTI JOIN 'data/processed/splits/train.parquet' AS tr USING (item_idx)) AS cold_test_items,
  (SELECT count(DISTINCT item_idx) FROM 'data/processed/splits/val.parquet' AS h
     ANTI JOIN 'data/processed/splits/train.parquet' AS tr USING (item_idx)) AS cold_val_items;

-- 6. pairs in more than one split, and rows below the positive threshold: both 0
WITH s AS (
  SELECT user_idx, item_idx, rating FROM 'data/processed/splits/train.parquet'
  UNION ALL SELECT user_idx, item_idx, rating FROM 'data/processed/splits/val.parquet'
  UNION ALL SELECT user_idx, item_idx, rating FROM 'data/processed/splits/test.parquet'
)
SELECT
  (SELECT count(*) FROM (SELECT user_idx, item_idx FROM s GROUP BY ALL HAVING count(*) > 1)) AS pairs_in_two_splits,
  (SELECT count(*) FROM s WHERE rating < 4.0) AS rows_below_min_rating;

-- 7. id maps contiguous from 0 and ascending in the raw id: true, true, 0, 0
SELECT
  (SELECT min(user_idx) = 0 AND max(user_idx) = count(*) - 1
     FROM 'data/processed/mappings/user_map.parquet') AS users_contiguous,
  (SELECT min(item_idx) = 0 AND max(item_idx) = count(*) - 1
     FROM 'data/processed/mappings/item_map.parquet') AS items_contiguous,
  (SELECT count(*) FROM (SELECT userId, lag(userId) OVER (ORDER BY user_idx) AS prev
     FROM 'data/processed/mappings/user_map.parquet') WHERE userId <= prev) AS user_order_violations,
  (SELECT count(*) FROM (SELECT movieId, lag(movieId) OVER (ORDER BY item_idx) AS prev
     FROM 'data/processed/mappings/item_map.parquet') WHERE movieId <= prev) AS item_order_violations;

-- 8. users with more than 500 train positives, and the maximum (D-008)
SELECT count(*) FILTER (WHERE n > 500) AS users_over_500_train, max(n) AS max_train_positives
FROM (SELECT user_idx, count(*) AS n FROM 'data/processed/splits/train.parquet' GROUP BY user_idx);

-- 9. split schema: exactly user_idx INTEGER, item_idx INTEGER, rating FLOAT, timestamp BIGINT
DESCRIBE SELECT * FROM 'data/processed/splits/train.parquet';
```

`split_sha256` recomputed from the parquet files, with the byte layout of api_spec.md §3 (prints `True`):

```bash
uv run python - <<'PY'
import hashlib, json
import numpy as np, polars as pl
d = hashlib.sha256()
for name in ("val", "test"):
    pairs = pl.read_parquet(f"data/processed/splits/{name}.parquet", columns=["user_idx", "item_idx"])
    arr = pairs.cast(pl.Int64).sort("user_idx", "item_idx").to_numpy()
    d.update(name.encode()); d.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
print(d.hexdigest() == json.load(open("data/processed/splits/meta.json"))["split_sha256"])
PY
```

**Sanity checks** (tick after the run)

- [x] `ratings.parquet` has 20,000,263 rows and every file matches the dtypes in api_spec.md §2; the split files hold exactly `user_idx` Int32, `item_idx` Int32, `rating` Float32 and `timestamp` Int64, with no `tie_key` (query 9).
- [x] Every split row has rating ≥ 4.0, and no (`user_idx`, `item_idx`) pair is in two splits (query 6).
- [x] `n_val == n_test == n_users`, and every user has ≥ 3 train rows (query 4).
- [x] Per user: max train timestamp ≤ val timestamp ≤ test timestamp (query 4, `order_violations` = 0).
- [x] `user_idx` and `item_idx` are contiguous from 0 and ascend with the raw id (query 7).
- [x] The three tie counts and the two cold-item counts in `meta.json` equal queries 4 and 5.
- [x] `split_sha256` recomputed from the parquet files equals `meta.json` (snippet above).
- [x] A second run with seed 42 reproduces `split_sha256` (CLI run, scratch-directory run and the notebook's `run_prepare` all give the value above).
- [x] A re-split with another seed changes `split_sha256`, moves items only for users counted in `users_with_any_boundary_tie`, and leaves every count except `cold_test_items` and `cold_val_items` unchanged (seed 43: 20,607 users moved, all among the 33,053 tie users; every count identical; cold items still 0/0).
- [x] The golden tie-break test in `tests/test_split.py` passes (fixed 20-row table, pinned `split_sha256`).
- [x] EDA figures from `notebooks/01_eda.ipynb` exported to `docs/figures/` (4 PNGs; notebook executed in place, 0 errors).

**Observations** (Day-1 EDA, `notebooks/01_eda.ipynb`): 50.0% of raw ratings are ≥ 4.0 (28% are exactly 4.0). Ratings per user: min 20, median 68, p90 334, max 9,254. Item popularity is extreme: Gini 0.90, the top 1% of movies carry 33.5% of ratings and the top 10% carry 86.1%; 566 movies make up half of all ratings, 3,972 movies have a single rating and 8,399 have fewer than 5. Genres by rating volume: Drama (8.9M), Comedy (7.5M), Action (5.6M), Thriller (5.3M); 19 titles have no parseable year. Tags: 465,564 applications, 38,644 distinct (35,170 after lower-casing and trimming), by 7,801 users (5.6%); 19,545 movies have at least one tag (19,011 of them rated: 71.1% of the 26,744 rated movies). Ties: 15.06% of users have test and val in the same second, 24.18% have a tie at either boundary. Global look-ahead (Day-1 measurement): averaged over test users, 53.7% of train rows post-date the user's test event, and at the median test event 56.8% of the kept catalogue had not been rated yet.

**Decisions recorded:** positives at rating ≥ 4.0; single-pass filter, items (≥ 5) then users (≥ 5) (D-002); all raw ratings retained; leave-one-out by (`timestamp`, `tie_key`), where `tie_key` = splitmix64(((userId << 32) | movieId) xor splitmix64(42)) over raw ids. It replaces the affine-mod-prime hash, which favoured the higher of two adjacent item ids ~81% of the time.

**Next:** Day 2 (D-012): the as-of accessor `features.asof.history(stage=...)` with its leakage canary, metrics and the paired bootstrap, MostPop and EASE (P2-anchors), BM25 item documents and a hello-world Streamlit deploy. Features are built only through the accessor (api_spec.md §5.6), because `ratings.parquet` and `tags.parquet` are raw: they hold every val/test positive, 36.7% of users have ratings after their test event, and 711 users tagged their own test movie.

### E002 — 2026-10-18 — MPS throughput and peak-memory benchmark (synthetic shapes)

Status: **recorded**. One Two-Tower training step on synthetic tensors of the ML-20M shapes, timed on the laptop; no metric is computed.

| Field | Value |
|---|---|
| Run id / W&B | `local-20261018-1830` (approximate: the script prints one JSON line per run; the runs were not logged to a file) / n/a |
| Code | `n/a (pre-git)`; script `scripts/bench_two_tower_step.py`, sha256 `7c6cd1aaf20f2a51bf5c1236adfab23023f57386e608f47917cfd3618ad66d7c` (the committed, linted copy of the scratch script that produced the numbers; same towers, loss and optimiser) |
| Config | no YAML; CLI flags `--B 4096 --K 0 --dtype bf16 --opt fused --device mps` (timing) and `--K 4096` (peak memory) |
| Data | none: random tensors of the exact shapes (136,674 users, 13,681 items, 9,704,107 positives, 25 user and 20 item dense features, 12 year buckets, d = 128, MLP 256) |
| Protocol / users | n/a (no metrics) |
| Hardware / seeds | laptop: Apple M5, 16 GB unified memory, MPS; PyTorch 2.14.1 / `torch.manual_seed(0)`; 5 warm-up steps, 30 timed steps |
| Cost / peak memory | minutes of laptop time / peak `torch.mps.driver_allocated_memory()`: 2.30 GB at B = 4,096 with K = 4,096 sampled negatives; 1.16 GB at K = 0; 2.43 GB at B = 8,192, K = 0. Torch-allocated static 0.41 GB (matches the 0.40 GB allocation-tracker figure in design.md §13) |

**Results** (ms per step → s per epoch; an epoch is 2,370 steps at B = 4,096)

| Run | ms per step | s per epoch |
|---|---|---|
| B = 4,096, K = 0, bf16 autocast, fused AdamW (the timing command) | 37.25 | 88 |
| B = 4,096, K = 0, fp16 | 31.0 | 74 |
| B = 4,096, K = 0, fp32 | 39.7 | 94 |
| B = 4,096, K = 0, foreach AdamW | 37.5 | — |
| B = 4,096, K = 4,096 | 78.7 | 187 |
| B = 2,048 | 13.5 | 64 |
| B = 8,192 | 151 | 179 |

A 30-epoch B = 4,096 in-batch run is therefore ~44 min on the laptop (~37 min with fp16). The M5 cell "2.3 GB / 88 s" in the README and design.md §13 pairs two of these runs: 2.3 GB is the peak with K = 4,096 sampled negatives (1.16 GB in-batch only), 88 s the in-batch-only epoch.

**Reproduction** (2026-10-18 21:10 PDT, the committed script at the sha256 above, 30 timed steps each): `--K 0` gave 35.45 ms per step → 84 s per epoch, peak 1.16 GB; `--K 4096` gave 73.5 ms → 174 s, peak 2.30 GB. Within 5% of the table, so the figures above stand (a 3-step run earlier the same evening gave 25.9 ms; fewer than ~30 timed steps is too noisy on MPS).

**Observations:** the slowness is MPS kernel quality, not bandwidth (log_softmax 13 GB/s, its backward 24 GB/s, nll_loss backward 7 GB/s vs 99 GB/s for a plain bf16→fp32 cast; M5 peak ~153 GB/s). Caveat: synthetic data and towers of the planned shape, not the real training loop.

**Next:** P3-smoke (Day 3) re-measures E002 with the real training loop.

### E003 — 2026-10-18 — Data pipeline re-run through `make eda` (notebook text fixes)

Status: **recorded**. Re-execution of `notebooks/01_eda.ipynb` after the final review reworded five of its cells (the single-pass item filter, the tag share over tagged ∩ rated movies, the MNS wording, the cold-item and HR@K take-aways); the notebook's `run_prepare` call rewrote `data/processed/`. Same code and config as E001.

| Field | Value |
|---|---|
| Run id / W&B | `local-20261018-2059` (local time of `created_at`) / n/a |
| Code | `n/a (pre-git)`; `shopsage_rec` 0.1.0 |
| Config | `configs/data/movielens_20m.yaml` @ `aec7ddd4dc98` (unchanged: the comment edit of the same day does not change the parsed-content hash) |
| Data | created_at 2026-10-19T03:59:50+00:00; split_sha256 `5fa6cbe8f702`; n_users 136,674; n_items 13,681 |
| Protocol / users | n/a (data preparation) |
| Hardware / seeds | laptop (M5, 16 GB) / split seed 42 |
| Cost / peak memory | one notebook execution (`make eda`, a few minutes) / not re-measured (E001: 2.3 GB peak RSS) |
| Commands | `uv run --no-sync jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 01_eda.ipynb` from `notebooks/` (what `make eda` runs) |

**Results:** `meta.json` is identical to E001's except `created_at`: every count, both tie fractions, `cold_test_items` = 0 and `split_sha256` `5fa6cbe8f7027e8c4724d592857a0c4134821b33eef982e98d849aa88898c22e` reproduce. The notebook executed with 0 errors and 0 stderr outputs; the four figures in `docs/figures/` were regenerated with identical content. Regenerated take-aways now print "19,545 movies have at least one tag (19,011 of them rated: 71.1% of the 26,744 rated movies)" and "35,170 lower-cased + trimmed" tags, matching E001's corrected Observations.

**Next:** nothing; E001 remains the entry of record for the Day-1 data facts, and the on-disk `data/processed/` now carries E003's `created_at`.

### E004 — 2026-10-23 — S1: the Liang split and the MostPop / EASE calibration

Status: **recorded**. The strong-generalisation split of Liang et al. (2018), rebuilt from the raw data, and the two calibration baselines under `Liang-SG-strict`. Both reproduce their published figures, which checks the split and the metric code end to end (D-001).

| Field | Value |
|---|---|
| Run id / W&B | `local-20261023-1306` / n/a: no W&B for the baselines |
| Code | working tree on top of `3e0f08f`, S1 changes not yet committed; `shopsage_rec` 0.1.0 |
| Config | `configs/data/movielens_20m_liang.yaml` @ `2f24b2c594e1` |
| Data | meta.json created_at 2026-10-23T20:06:05+00:00; split_sha256 `4f93a0e6f611`; 136,677 users (116,677 training, 10,000 validation, 10,000 test); 20,108 items |
| Protocol / users | `Liang-SG-strict` / λ chosen on V_A (5,000 validation users); test metrics on all 10,000 test users |
| Hardware / seeds | laptop (Apple M5, 16 GB) / deterministic models; split seed 98765; V_A / V_B seed 42; bootstrap seed 42 |
| Commands | `uv run shopsage prepare --config configs/data/movielens_20m_liang.yaml`, then `uv run shopsage evaluate` (λ grid 250, 500, 1000) |
| Artifacts | `data/processed/liang/` (split); `artifacts/eval/local-20261023-1306/` (`per_user.parquet`, `summary.json`) |
| Cost / peak memory | prepare 5.1 s at 2.2 GB peak RSS; evaluate 211 s (EASE 204 s: three float32 inversions of the 20,108 × 20,108 Gram matrix plus scoring) at 3.5 GB peak RSS |

**Split.** Counts from `meta.json`; published counts in parentheses (Liang et al. 2018, Table 1, and the authors' notebook).

| Quantity | Value |
|---|---|
| Positives (rating ≥ 4.0, users with ≥ 5) | 9,990,682 (9,990,682) |
| Users / movies before the split | 136,677 / 20,720 (136,677 / 20,720) |
| Training users / training catalogue | 116,677 / 20,108 items (20,108) |
| Training rows | 8,538,846 |
| Validation fold-in / targets | 589,887 / 142,514 |
| Test fold-in / targets | 579,008 / 139,775 |
| Held-out users without targets | 0 validation, 0 test |

**Results** (test, 10,000 users; mean [95% CI], percentile bootstrap over users, 2,000 resamples, seed 42). Published rows: Steck 2019, Table 1 (standard errors about 0.002), and Milogradskii et al. 2024 for NDCG@10 in the fold-in-in-training variant.

| Model | Recall@20 | Recall@50 | NDCG@10 | NDCG@100 | HR@50 |
|---|---|---|---|---|---|
| MostPop | 0.1618 [0.1581, 0.1657] | 0.2352 [0.2310, 0.2396] | 0.1275 [0.1241, 0.1311] | 0.1906 [0.1873, 0.1938] | 0.7425 [0.7342, 0.7510] |
| popularity, published | 0.162 | 0.235 | — | 0.191 | — |
| EASE, λ = 1,000 | 0.3893 [0.3843, 0.3946] | 0.5210 [0.5156, 0.5264] | 0.3280 [0.3231, 0.3331] | 0.4201 [0.4159, 0.4243] | 0.9246 [0.9192, 0.9296] |
| EASE, published (λ ≈ 500) | 0.391 | 0.521 | 0.329 (`Liang-SG-fit`) | 0.420 | — |

EASE's λ on V_A (Recall@50): 250 → 0.5145, 500 → 0.5165, 1,000 → 0.5166, chosen; the three lie within 0.002. Paired Δ, EASE − MostPop: Recall@50 +0.2858 [0.2805, 0.2909], NDCG@10 +0.2005 [0.1960, 0.2051].

**Calibration gate: passed.** On Recall@50, |MostPop − 0.235| = 0.0002 and |EASE − 0.521| < 0.0001, inside both the 0.01 gate and the paper's standard errors; Recall@20 and NDCG@100 agree within 0.002.

**Deployment probe** (S1; laptop only, since the Community Cloud deploy needs the push). Synthetic catalogue of 20,108 items with a 128-d dense vector and a sparse vector of 20 terms over a 5,000-term vocabulary; 100 queries per path, top-50; Milvus Lite 3.2.2 with pymilvus 3.0.2.

| Run | Dense p50 / p95 | Sparse p50 / p95 | Hybrid (RRF) p50 / p95 | Dense overlap with exact | Peak RSS |
|---|---|---|---|---|---|
| `run_probe()` in a Python process | 13.5 / 17.9 ms | 12.5 / 18.6 ms | 23.8 / 33.9 ms | 1.000 | 442 MB |
| The Streamlit app through `AppTest` (Streamlit and the app in one process) | 17.2 / 24.3 ms | 15.3 / 21.9 ms | 30.0 / 43.8 ms | 1.000 | 451 MB |

Without a `flush()` after inserting, every query scanned the unflushed rows in Python (dense p50 181 ms, hybrid 727 ms); the probe flushes before querying, and the demo will too. A second `AppTest` run after a tidy-up of `app.py` gave dense 9.4 / 20.7 ms, sparse 8.5 / 20.3 ms, hybrid 16.8 / 34.8 ms and 456 MB, within the run-to-run spread of a shared laptop.

**Observations:** EASE alone already reaches the NDCG@10 target (0.32) under this protocol, so the ranker counts only if it beats EASE; the Recall@50 target (0.55) sits 0.029 above EASE. The bootstrap half-width on Recall@50 is about ±0.005 at 10,000 users. pymilvus 3.0.2 imports pandas, and milvus-lite 3.2.2 runs on faiss-cpu.

**Next:** S2, the Two-Tower and the Modal sweep.

### E005 — 2026-10-24 — S1 probe on Streamlit Community Cloud (the demo host)

Status: **recorded** (the owner's screenshot of the deployed probe page). Same probe as E004 (synthetic 20,108 items, 128-d dense, 20-term sparse vectors, 100 queries per path, top-50; Milvus Lite 3.2.2, pymilvus 3.0.2), run at page load on the Community Cloud host from `05f74c4`.

| Measure | Value |
|---|---|
| `import pymilvus` | 1.9 s |
| Build the collection (insert, index, flush) | 7.9 s |
| Dense p50 / p95 | 76.3 / 135.2 ms |
| Sparse p50 / p95 | 63.0 / 94.6 ms |
| Hybrid (RRF) p50 / p95 | 118.0 / 176.3 ms |
| Dense overlap with exact | 1.000 |
| Peak RSS | 482 MB |

**Observations:** memory fits the 690 MB guarantee with ~200 MB to spare. Latency is 4–5× the laptop's (E004): the hybrid p95 alone uses 176 of the 200 ms budget (D-013) before the user tower and the LightGBM re-rank, so Milvus Lite as the demo's online path puts the latency target at risk. **Plan for S6:** measure the numpy exact dense top-k (one 20,108 × 128 matvec) and a scipy BM25 matvec on the host; if Milvus cannot meet D-013 with the ranker, the demo serves through numpy / scipy (identical results: FLAT overlap 1.000) and shows Milvus as the alternative engine with its measured latency. One shared host, one page load; S6 repeats with more queries.

### E006 — 2026-10-24 — S2: Two-Tower smoke and a 3-epoch A1 check on the laptop (MPS)

Status: **recorded; not a reported result.** A check that the S2 trainer runs end to end on real data and how fast; the reported Two-Tower numbers come from the Modal sweep (30 epochs, τ / lr tuned). `shopsage train`, `Liang-SG-strict`, epoch kept on V_A Recall@50, test on all 10,000 test users through the numpy user tower.

| Field | Value |
|---|---|
| Code | working tree on top of `05f74c4`, S2 changes not yet committed; `shopsage_rec` 0.1.0, torch 2.14.1 |
| Data | split_sha256 `4f93a0e6f611`; 20,108 items; genome PCA-64 (`item_features.npz`) |
| Hardware / seed | Apple M5 16 GB, MPS, bf16 autocast / seed 42; bootstrap seed 42 |
| Settings | B = 4,096, τ = 0.05, lr = 2e-3, ≤ 50 history items sampled per row, 128-d; no W&B |

| Run | Users | Epochs | s / epoch | Peak memory | Test Recall@50 [95% CI] | NDCG@10 |
|---|---|---|---|---|---|---|
| `smoke-A1-mps` (A1) | 10% of training users | 1 | 12.3 | 1.18 GB | 0.4207 [0.4151, 0.4267] | 0.2392 |
| `smoke-A2-mps` (A2, M = 2,048 uniform) | 10% | 1 | 14.9 | 2.22 GB | 0.4090 [0.4033, 0.4149] | 0.2312 |
| `smoke-A4-mps` (A4, full softmax) | 10% | 1 | 41.8 | 4.97 GB | 0.4150 [0.4094, 0.4207] | 0.2350 |
| `check-A1-full3-mps` (A1) | all 116,677 | 3 | 129–146 | 2.12 GB | 0.5145 [0.5088, 0.5199] | 0.3163 |

`check-A1-full3-mps`: 8,538,846 rows, 2,085 steps per epoch; V_A Recall@50 0.5027 after epoch 2 and 0.5117 after epoch 3 (still rising); test Recall@20 0.3800, NDCG@100 0.4127, HR@50 0.9248; 418 s wall-clock. The A3 path (mining then training on mined negatives) is exercised by the test suite only.

**Observations:** three epochs already put the Two-Tower within 0.007 of EASE's Recall@50 (0.5210, E004). Peak memory is set by the item tower over the full catalogue (A4: every item each step), not by the user side; every arm fits an 8 GB GPU. MPS peak memory is the driver's allocation, an upper bound on what CUDA reports.

### E007 — 2026-10-24 — S2: the Two-Tower sweep on Modal (ablation (a))

Status: **recorded.** The owner ran `modal run scripts/modal_train.py` (after a 2-epoch trial, whose runs were deleted from the Volume). All numbers below come from `artifacts/runs/<run_id>/metrics.json` and `per_user.parquet`, copied back from the `shopsage-data` Volume.

| Field | Value |
|---|---|
| Run id / W&B | `tune-A1-*` (4), `A0-s42`, `A1-s42/43/44`, `A2-s42`, `A3-s42/43/44`, `A4-s42` / W&B project `shopsage-recommender` (owner's account) |
| Code | working tree on top of `05f74c4` (S2 not yet committed), with the plain-JSON summary fix; `shopsage_rec` 0.1.0, torch 2.14.1+cu126, Python 3.12.10 |
| Data | split_sha256 `4f93a0e6f611` (rebuilt on Modal from the raw archive, md5-checked; identical to the laptop's) |
| Protocol / users | `Liang-SG-strict`; epoch chosen on V_A (5,000 validation users, Recall@50, evaluated every 2 epochs); test on all 10,000 test users through the numpy user tower |
| Hardware / seeds | Modal, one NVIDIA A10 (24 GB) per run, bf16 autocast / seeds 42, 43, 44 for A1 and A3, 42 for the rest; bootstrap seed 42, 2,000 resamples |
| Settings | B = 4,096, ≤ 50 history items, 128-d, 30 epochs, fused AdamW (no decay on embeddings); τ = 0.05, lr = 3e-3 chosen by the tuning runs; A2 M = 2,048; A3 k = 8 per row from ranks 101–500 of `A1-s42` |
| Cost / peak memory | 9,017 GPU-seconds (2.50 A10-hours ≈ \$2.75 at \$1.10/h, plus CPU jobs; the Modal dashboard has the bill); peak GPU memory 0.98 GB (A0, A1), 1.30 GB (A2), 1.38 GB (A3), 3.21 GB (A4) |

**Tuning** (A1, seed 42, 10 epochs; V_A Recall@50): τ 0.05 / lr 1e-3 0.52690; τ 0.05 / lr 3e-3 **0.52695** (chosen; a near tie); τ 0.1 / lr 1e-3 0.51391; τ 0.1 / lr 3e-3 0.51702.

**Results** (test, 10,000 users). Three-seed arms: mean ± sd over seeds; single runs: mean [95% CI].

| Arm | Seeds | Best epoch | Recall@20 | Recall@50 | NDCG@10 | NDCG@100 | HR@50 | s / epoch |
|---|---|---|---|---|---|---|---|---|
| A0, in-batch, no logQ | 1 | 30 | 0.2030 | 0.3111 [0.3049, 0.3171] | 0.1413 | 0.2294 | 0.7803 | 19.0 |
| A1, in-batch + logQ (ref) | 3 | 28, 30, 28 | 0.3997 ± 0.0017 | **0.5361 ± 0.0005** | **0.3351 ± 0.0005** | 0.4324 ± 0.0001 | 0.9371 ± 0.0016 | 21.5 |
| A2, + 2,048 uniform | 1 | 24 | 0.3975 | 0.5342 [0.5287, 0.5398] | 0.3332 | 0.4306 | 0.9354 | 30.7 |
| A3, + 8 mined per row | 3 | 22, 24, 26 | 0.3995 ± 0.0003 | 0.5341 ± 0.0008 | 0.3351 ± 0.0011 | 0.4317 ± 0.0008 | 0.9346 ± 0.0005 | 26.6 |
| A4, full softmax | 1 | 30 | 0.4001 | 0.5333 [0.5277, 0.5388] | 0.3335 | 0.4309 | 0.9340 | 75.5 |
| EASE (E004) | — | — | 0.3893 | 0.5210 | 0.3280 | 0.4201 | 0.9246 | — |

Per-seed A1 Recall@50: 0.5363 [0.5306, 0.5417], 0.5356 [0.5300, 0.5413], 0.5365 [0.5309, 0.5420]; A3: 0.5343, 0.5349, 0.5332.

**Paired Δ** (same users, same resample indices; seed-averaged per user for the 3-seed arms):

| Comparison | Δ Recall@50 [95% CI] | Δ NDCG@10 [95% CI] | Per-seed Δ Recall@50 |
|---|---|---|---|
| **A3 − A1 (the hard-negative ablation)** | **−0.0020 [−0.0033, −0.0007]** | −0.0001 [−0.0011, +0.0010] | −0.0020, −0.0007, −0.0033 |
| A1 − A0 (logQ correction) | +0.2252 [+0.2203, +0.2297] (seed 42) | +0.1942 [+0.1896, +0.1986] | — |
| A2 − A1 (seed 42) | −0.0021 [−0.0045, +0.0003] | −0.0023 [−0.0040, −0.0006] | — |
| A4 − A1 (seed 42) | −0.0030 [−0.0053, −0.0007] | −0.0019 [−0.0037, −0.0002] | — |
| A1 − EASE | +0.0151 [+0.0119, +0.0184] | +0.0071 [+0.0045, +0.0099] | +0.0152, +0.0146, +0.0155 |

**Against the targets.** Recall@50 ≥ 0.55: not met by retrieval alone (A1 0.5361; RecVAE's published 0.553 is the frontier); the ranker (S4) is the remaining route. NDCG@10 ≥ 0.32: already met by the retriever (0.3351), and A1 beats EASE on both metrics. Hard-negative ablation +0.08: **not met**; mined negatives give no gain (−0.0020, CI below zero).

**Observations:** the logQ correction is the decisive negative-sampling choice here (+0.225 Recall@50 over plain in-batch, whose popularity bias leaves it near MostPop). With logQ and accidental-hit masking in place, every extra source of negatives (uniform, mined, all items) lands within 0.003 of A1 or slightly below it: the in-batch softmax at B = 4,096 already sees 4,095 negatives per row, and the correction already removes the sampling bias that extra negatives would otherwise fix. Mined ranks 101–500 also hold unlabelled positives (a held-out user's 20% targets are, by construction, items like those). A3 peaks earlier (epochs 22–26) than A1 (28–30). The A0 curve was still rising at epoch 30, but its gap to A1 (0.23) is far outside any plausible remaining gain.

### E008 — 2026-10-24 — S3: recall paths (ablation (b)) and the Milvus Lite serving index on real data

Status: **recorded.** `uv run shopsage recall --run-id recall-e008` (defaults: dense runs `A1-s42/43/44` from E007, depth 200, RRF k = 60, 1,000 parity users).

| Field | Value |
|---|---|
| Run id / W&B | `recall-e008` / n/a |
| Code | working tree on top of `fefcf6c` (S3 not yet committed); `shopsage_rec` 0.1.0; Milvus Lite 3.2.2, pymilvus 3.0.2 |
| Data | split_sha256 `4f93a0e6f611`; 20,108 items |
| Protocol / users | `Liang-SG-strict`; BM25 settings chosen on V_A (5,000 users) by Recall@50; test on all 10,000 test users |
| Hardware | laptop (Apple M5, 16 GB), CPU only |
| Artifacts | `artifacts/eval/recall-e008/` (`per_user.parquet`, `summary.json`) |
| Cost / peak memory | 129 s wall-clock; 4.1 GB peak RSS (offline evaluation, not the demo) |

**BM25 documents.** 17,283 terms after dropping terms found in a single document: 6,115 title words, 19 genres, 11,149 tags (genome top-20 per movie and training users' tags); 404,876 non-zeros; 6 items with an empty document (stored in Milvus with a placeholder term). Grid on V_A (Recall@50), k1 × b × query weighting:

| k1 | b | binary | log |
|---|---|---|---|
| 0.9 | 0.4 | 0.2232 | **0.2526** (chosen) |
| 0.9 | 0.75 | 0.1996 | 0.2211 |
| 1.5 | 0.4 | 0.2202 | 0.2504 |
| 1.5 | 0.75 | 0.1923 | 0.2159 |

**Results** (test, 10,000 users; dense and hybrid: mean ± sd over the three A1 seeds, then seed 42's 95% CI).

| Arm | Recall@20 | Recall@50 | NDCG@10 | NDCG@100 | HR@50 |
|---|---|---|---|---|---|
| MostPop (E004) | 0.1618 | 0.2352 | 0.1275 | 0.1906 | 0.7425 |
| EASE (E004) | 0.3893 | 0.5210 | 0.3280 | 0.4201 | 0.9246 |
| B0, BM25 | 0.1678 | 0.2529 [0.2489, 0.2571] | 0.1556 | 0.2146 | 0.7682 |
| B1, dense (A1) | 0.3997 ± 0.0017 | **0.5361 ± 0.0005** [0.5306, 0.5417] | **0.3351 ± 0.0005** | 0.4324 | 0.9371 |
| B2, hybrid (RRF of B0 and B1) | 0.3291 ± 0.0006 | 0.4647 ± 0.0002 [0.4596, 0.4703] | 0.2789 ± 0.0007 | 0.3737 | 0.9182 |

The dense rows reproduce E007 to the fourth decimal (computed again from the exported files), a check that the export and the evaluation agree.

**Paired Δ** (seed-averaged per user; per-seed ranges in brackets after):

| Comparison | Δ Recall@50 [95% CI] | Δ NDCG@10 [95% CI] | Per-seed Δ Recall@50 |
|---|---|---|---|
| B2 − B1 (hybrid − dense) | −0.0715 [−0.0748, −0.0685] | −0.0562 [−0.0591, −0.0531] | −0.0709 to −0.0720 |
| B0 − B1 (BM25 − dense) | −0.2833 [−0.2891, −0.2772] | −0.1796 [−0.1840, −0.1753] | −0.2827 to −0.2836 |

**Candidate coverage for the ranker (S4)** (mean share of a test user's targets in the set; dense = A1-s42): dense top-500 **0.8950**; BM25 top-100 0.3448; their union **0.8971** (+0.0021).

**Milvus Lite serving index** (one collection: A1-s42 item vectors, FLAT IP; BM25 weights, sparse inverted index; built and flushed in 2.56 s). 1,000 random test users, top-50, seen items removed by a filter inside the search, hybrid = RRF of depth-200 lists:

| Path | Same scores rank by rank as offline | Item overlap | Milvus p50 / p95 | Exact numpy / scipy p50 / p95 (one user) |
|---|---|---|---|---|
| dense | 1.000 | 1.000 | 6.44 / 6.93 ms | 0.38 / 0.42 ms |
| sparse (BM25) | 1.000 | 1.000 | 31.15 / 41.75 ms | 0.79 / 1.03 ms |
| hybrid (RRF) | 1.000 | 1.000 | 38.25 / 48.68 ms | 1.27 / 1.53 ms |

The numpy user tower takes 0.19 / 0.29 ms per user.

**Observations:**
- BM25 over item text barely beats popularity (+0.018 Recall@50). Titles, genres and tags describe what a movie is, not who watches it, so the dense model, trained on co-consumption, is far ahead.
- Equal-weight RRF lets the weak path displace about 7 points of the strong one's Recall@50. The second path earns its place as ranker input instead: features plus the candidate union. Even there the union adds only +0.002 coverage, so the S4 ranker will rely on the dense candidates.
- The chosen BM25 setting sits at the low edge of the k1 and b grid. A wider grid might add a little, but not the 0.28 it trails by.
- Milvus Lite returns exactly the offline answers on all three paths.
- On the laptop, exact search in numpy / scipy is 16–40× faster than Milvus Lite for the same results. The real BM25 queries (hundreds of terms) make the sparse path the slow one; the S1 probe's 5-term queries hid that.
- Projected to the demo host with E005's slowdown of about 4–5×, Milvus hybrid search would take roughly 150–250 ms p95 before re-ranking, against the 200 ms budget (D-013). S6 decides the serving path from host measurements (plan.md §8).

### E009 — 2026-10-24 — S4: the LightGBM LambdaRank re-ranker and the two-stage system

Status: **recorded.** `uv run shopsage rank --run-id rank-e009` (defaults below; settings fixed before the run, not tuned on test).

| Field | Value |
|---|---|
| Run id / W&B | `rank-e009` / n/a |
| Code | working tree on top of `a54bdbe` (S4 not yet committed); `shopsage_rec` 0.1.0; LightGBM 4.7.0 |
| Data | split_sha256 `4f93a0e6f611`; 20,108 items |
| Protocol / users | `Liang-SG-strict`. Training: V_B (5,000 validation users not used for any model selection), 4,500 for fitting and 500 (seed 0) for early stopping on NDCG@10. Test: all 10,000 test users |
| Hardware / seeds | laptop (Apple M5, 16 GB), CPU / LightGBM seeds 42, 43, 44, `deterministic`; bootstrap seed 42 |
| Settings | candidates: dense `A1-s42` top-500 ∪ BM25 top-100 (E008 settings), fold-in excluded; 12 features (`ranking/features.py`); sparse EASE from λ = 1,000 (E004), pruned to 100 neighbours per item; LambdaRank, `lambdarank_truncation_level` 50, 63 leaves, learning rate 0.05, `min_data_in_leaf` 50, feature and bagging fraction 0.8, ≤ 1,000 rounds, early stopping 50 |
| Artifacts | `artifacts/rank/rank-e009/` (`models/*.txt`, `per_user.parquet`, `summary.json`) |
| Cost / peak memory | 133 s wall-clock for everything (EASE fit, features, 6 LightGBM fits, test, latency); 4.0 GB peak RSS |

**Training data.** 2,416,372 candidate rows for the 4,500 fitting users, 2.2% positive; 33 V_B groups had no target among their candidates and were dropped. The candidates hold 84.2% of V_B's targets and 83.9% of the test targets (total hits / total targets). E008's 0.897 is the mean of per-user shares, the same sets weighted per user. Test users have 502 to ~600 candidates, 536.7 on average.

**Results** (test, 10,000 users; rankers: mean ± sd over 3 LightGBM seeds, then the 95% CI of the per-user seed mean; references: mean [95% CI]).

| Arm | Recall@20 | Recall@50 | NDCG@10 | NDCG@100 | HR@50 | Rounds |
|---|---|---|---|---|---|---|
| EASE (recomputed here; = E004) | 0.3893 | 0.5210 [0.5156, 0.5264] | 0.3280 [0.3231, 0.3331] | 0.4201 | 0.9246 | — |
| dense A1-s42 (= E007) | 0.4016 | 0.5363 [0.5306, 0.5417] | 0.3354 [0.3304, 0.3405] | 0.4323 | 0.9381 | — |
| **ranker** (all 12 features) | 0.4079 ± 0.0011 | **0.5420 ± 0.0006** [0.5365, 0.5474] | **0.3428 ± 0.0000** [0.3378, 0.3478] | 0.4391 ± 0.0001 | 0.9358 ± 0.0009 | 89, 86, 102 |
| ranker-no-ease (without `ease_score`) | 0.3996 ± 0.0003 | 0.5356 ± 0.0008 [0.5300, 0.5410] | 0.3307 ± 0.0010 [0.3259, 0.3358] | 0.4282 ± 0.0006 | 0.9362 ± 0.0010 | 14, 21, 43 |

Per seed, ranker: Recall@50 0.5413 / 0.5420 / 0.5426, NDCG@10 0.34284 / 0.34281 / 0.34288.

**Paired Δ** (per-user seed means):

| Comparison | Δ Recall@50 [95% CI] | Δ NDCG@10 [95% CI] |
|---|---|---|
| ranker − EASE | +0.0209 [+0.0178, +0.0242] | +0.0148 [+0.0126, +0.0171] |
| ranker − dense | +0.0057 [+0.0040, +0.0074] | +0.0074 [+0.0057, +0.0091] |
| ranker − ranker-no-ease | +0.0064 [+0.0049, +0.0079] | +0.0121 [+0.0105, +0.0136] |
| ranker-no-ease − dense | −0.0007 [−0.0019, +0.0005] | −0.0047 [−0.0062, −0.0032] |
| ranker-no-ease − EASE | +0.0145 [+0.0112, +0.0180] | +0.0027 [−0.0000, +0.0055] |

**Feature importance** (share of total gain at the best iteration, mean over seeds): `dense_rank` 0.709, `dense_score` 0.103, `ease_score` 0.081, `log_hist_len` 0.020, `bm25_score` 0.015, `item_log_pop` 0.013, `genome_sim` 0.013, `year_gap` 0.011, `genre_affinity` 0.010, `item_year` 0.009, `item_mean_rating` 0.009, `bm25_rank` 0.007.

**Serving latency on the laptop** (500 random test users, one at a time; exact numpy / scipy retrieval, the same feature code, LightGBM on one thread, top 10): retrieval plus features 5.87 / 7.21 ms p50 / p95; LightGBM 1.27 / 1.65 ms; **total 7.21 / 8.56 ms**.

**Against the targets.**
- **NDCG@10 after the ranker ≥ 0.32: met.** The system scores 0.3428, above EASE (+0.0148, CI above zero) and above the dense order (+0.0074).
- **Recall@50 ≥ 0.55: not met.** The system reaches 0.5420, 0.008 short of the target. It is still the best of every model here, and the highest published figure under this protocol is RecVAE's 0.553.
- **Latency ≤ 200 ms:** 8.6 ms p95 offline on the laptop through the exact path. The demo host is measured in S6; E005 showed it running about 4–5× slower than the laptop.

**Observations:**
- The ranker's gain comes from adding EASE's item-item signal to the dense order. Without `ease_score`, LightGBM stops within 14–43 rounds and only reproduces the dense ranking (Recall@50 Δ −0.0007), so the content features (BM25, genome, genres, year) add nothing measurable on their own.
- The candidate set caps the system: 16% of test targets are never retrieved, which bounds what any re-ranker can recover.
- The sparse EASE (100 neighbours per item, about 2M weights) is small enough for the demo bundle, unlike the dense 20,108² matrix.
