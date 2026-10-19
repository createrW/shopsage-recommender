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

The evaluation code is PLANNED (Day 2, D-012); metric definitions are in api_spec.md §8.

| Label | Split | Evaluated users | Ranked against | Headline metrics |
|---|---|---|---|---|
| `LOO-full` | Day-1 leave-one-out by (`timestamp`, `tie_key`), `splits/` | every user, at the val event (V_A or V_B) or at the test event | all items except the user's history positives (train; train + val at test) and anything the user rated before the cutoff | HR@{10, 50, 100, 500}, NDCG@10, MRR@10 |
| `LOO-asof` | same | same | as `LOO-full`, and also without items whose first rating in `ratings.parquet` is later than the cutoff (the catalogue as of the event) | same |
| `Liang-SG-strict` | strong generalisation (Liang et al. 2018, §4.3): held-out users are never trained on; models train on the training users' full histories; `min_item_interactions: 1`. PLANNED, only if D-001 adopts the Liang protocol | held-out users; a random 20% of each one's positives are the targets | all items except the user's fold-in positives | Recall@20 and Recall@50 (truncated), NDCG@100, NDCG@10; the Recall@20/50 and NDCG@100 anchors (design.md §9.2) are from this variant |
| `Liang-SG-fit` | strong generalisation in the fold-in-in-training variant: the held-out users' fold-in 80% is part of the training data (Milogradskii et al., RecSys 2024); `min_item_interactions: 1`. PLANNED, only if D-001 adopts the Liang protocol | same as `Liang-SG-strict` | same | same; the NDCG@10 anchors (design.md §9.2) are from this variant |

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

The owner's brief set Recall@50 0.42 → 0.58 and NDCG@10 0.32 before any model was trained. They were re-labelled on Day 1 against published ML-20M figures, each verified against the paper's table (full list in design.md §9). Nothing here is a result until a logged run measures it.

| Metric | Label | Value | Under the Liang protocol (recommended headline `Liang-SG-strict`, D-001; Recall@50 anchors: `Liang-SG-strict`; NDCG@10 anchors: `Liang-SG-fit`) | Under `LOO-full` (the Day-1 split) |
|---|---|---|---|---|
| Recall@50 | round-1 TARGET | ≥ 0.42 | plausible: WMF 0.498, EASE 0.521 and Mult-VAE 0.537 all clear it | 0.42 is above the only non-sequential anchor quoted (BPR-MF: HR@10 0.081, HR@100 0.337) in a non-comparable setup; in-repo EASE under `LOO-full` (P2-anchors) shows whether a non-sequential model reaches it |
| Recall@50 | stretch TARGET | 0.50–0.52 | realistic stretch: the WMF–EASE band (0.498–0.521) | among the published anchors only sequential models reach it (SASRec, 2023 recipe: HR@50 0.550); no non-sequential ML-20M figure under this protocol is quoted |
| Recall@50 | brief's upper endpoint, not adopted | 0.58 | exceeds every published Liang-protocol Recall@50 (highest verified: RecVAE 0.553) | reached only by HSTU (0.589) and HSTU-large (0.615) |
| NDCG@10 | stretch TARGET | 0.32 | at EASE's level (0.329 in the fold-in-in-training variant) | out of reach: the best verified figure is 0.211 (HSTU-large) |

Under `LOO-full` Recall@K equals HR@K, since each user has one relevant item. Sources: [Klenitskiy & Vasilev 2023](https://arxiv.org/abs/2309.07602) and [Petrov & Macdonald 2022](https://arxiv.org/abs/2207.07483) (BPR-MF under full-corpus leave-one-out); [Zhai et al. 2024](https://arxiv.org/abs/2402.17152) (SASRec 2023 recipe, HSTU); [Liang et al. 2018](https://arxiv.org/abs/1802.05814) (WMF, Mult-VAE); [Steck 2019](https://arxiv.org/abs/1905.03375), Table 1 (EASE 0.521); Shenbin et al., WSDM 2020 (RecVAE); Milogradskii et al., RecSys 2024 (NDCG@10, fold-in-in-training variant).

## Decisions

| ID | Topic | Status | Due | Decision or recommendation |
|---|---|---|---|---|
| D-001 | Evaluation protocol | OPEN | Day 4 (owner) | Recommended: the Liang protocol with `Liang-SG-strict` as the headline, because that is where the Recall@50 anchors live and the inductive user tower (D-004) scores users it never trained on; Recall@20/50, NDCG@100, NDCG@10 and paired bootstrap CIs over users, with NDCG@10 compared with the `Liang-SG-fit` anchors and that caveat stated; the Day-1 split as a secondary table under both `LOO-full` and `LOO-asof`; MostPop and EASE run in-repo as calibration anchors, under both Liang variants in P4-liang (design.md §9). |
| D-002 | Item filter: 5 vs Liang's 0 | DECIDED (Day 1) | — | `min_item_interactions: 5` for the leave-one-out split: a single-pass filter (not an iterated 5-core), so that every item ID embedding gets training signal; the ML-20M leave-one-out papers in design.md §9.2 keep all 26,744 items. It departs from Liang et al., whose released preprocessing has no item filter (136,677 users, 20,720 items, 9,990,682 positives); both Liang variants (`Liang-SG-strict`, `Liang-SG-fit`) run with `1`, which reproduces it (design.md §8). |
| D-003 | Demo data licence | OPEN | before Day 9 | MovieLens may not be redistributed. Until GroupLens grants permission the public demo ships no per-user rating rows (visitors build a "pick 5 movies" profile) and downloads titles from GroupLens at startup. The delivery path is decided (a sha256-pinned GitHub Release asset, api_spec.md §9.5); what the asset may contain is not. |
| D-004 | User tower (ML-20M has no demographics) | OPEN | Day 3 | Recommended: an inductive tower, an MLP over mean-pooled item-tower embeddings of the user's last ≤ 50 as-of positives plus behavioural features (log activity, mean rating, genre affinity, recency), with the `user_id` embedding optional under ID-dropout 0.3 and kept as an ablation arm. It works under leave-one-out, strong-generalisation and temporal protocols and is ~15 lines of numpy in the demo (13,681 × 128 item embeddings ≈ 7 MB plus MLP weights, no 70 MB user table); numpy vs torch forward on 1,000 users, max abs diff < 1e-5. It serves visitor profiles through the missing-feature path (design.md §7); their quality is not measured. |
| D-005 | Negatives (the brief's "hard negatives" round) | DECIDED (Day 1; revised 2026-10-18) | — | Mixed negative sampling (MNS, Yang et al. 2020): in-batch negatives plus M = 2,048 uniformly sampled negatives per batch, logQ over the mixed proposal log(B·p_batch(j) + M·q(j)) with q(j) = 1/13,681 (`sampling: popularity_power` with `alpha: 0.0`). Every off-diagonal column that is a train positive of the row's user is masked; the row's own positive column never is (~660 masked in-batch columns per row at B = 4,096, row-weighted mean; median 571, p90 1,298; ~4 of them duplicates of the row's own positive). A 136,674 × 13,681 bitset is 234 MB; sorted int64 keys also work. Optional: true hard negatives mined from ranks 101–500 of the round-1 model. Popularity and masks come from train only; val/test items stay in the negative pool. Arms: ablation (a). Revised 2026-10-18 after review: the sampler changed from popularity^0.75 to uniform, because on the train split the bottom tercile's share of negative columns at M = 2,048 is 1.0% under f^0.75 and 11.5% under uniform (0.5% in-batch only). |
| D-006 | Ranker training data | DECIDED (Day 1; revised 2026-10-18) | — | Val users split into V_A (retriever selection: best epoch, τ, hybrid α) and V_B (ranker labels, 10% of V_B held out for LightGBM early stopping), a disjoint split seeded by `val_partition.seed` = 42 (`frac_va: 0.5`), fixed across training seeds, so V_A and V_B are the same users in every run. Top-500 candidates with as-of seen items masked, label 1 = the val item; a group that misses it is dropped, never given an injected positive, and the drop rate (1 − V_B HR@500) is logged; all 500 candidates per group at a 20,000-user cap (10M rows ≈ 2 GB); fallback if RAM is tight: ranks 1–50 plus 50 random from 51–500 with row weight 9 on the random ones. Low ratings feed features only, never labels. The candidates come from one retriever run, referenced by its run id: the seed-42 run of B1 (the best arm of ablation (a), chosen by mean V_A HR@50 over seeds; the seed is pre-declared, not the best seed). The ranker's 3 seeds (Rule 5) are LightGBM seeds on that one retriever run; retriever-seed variance is measured in ablation (a). Test protocol: P6-rank. Revised 2026-10-18 after review: the default changed from the 50 + 50 subsample to all 500 candidates, because LambdaRank learns the positive-to-negative odds at each rank, and keeping 11% of the rank-51–500 negatives in training but all of them at test inflates those odds about 9× for ranks 51–500, which over-promotes deep candidates at test with `retrieval_rank` as a feature; the fallback's row weight of 9 (450/50) is an approximate inverse-sampling correction, because lambdarank applies row weights after the pairwise lambdas are accumulated. |
| D-007 | Metrics and CIs | DECIDED (Day 1) | — | Under leave-one-out Recall@K = HR@K, so tables report HR@{10, 50, 100, 500} (HR@500 caps the ranker), NDCG@10 and MRR@10. Paired bootstrap over users on the difference between arms (same resample indices, 2,000 resamples, percentile CIs, seeded numpy Generator); ≥ 3 training seeds per arm, mean ± sd. At n = 136,674 the HR@50 CI half-width is ~±0.27 points, so seed variance matters more (api_spec.md §8); under the Liang protocol (10,000 test users) the half-width is ~±1.0 point, so user sampling also matters. |
| D-008 | Retrieval | DECIDED (Day 1) | — | Exact scoring is the default: batched matmul → `masked_fill(M_u, −inf)` → `torch.topk` / `np.argpartition`, in chunks of ≤ 8,192 users (seconds on CPU at 13,681 × 128), with M_u the user's seen items. Seen items are set to −inf before top-k: 1,293 users have more than 500 train positives (max 3,148), so "top-500, then drop seen" is a bug. Where IndexFlatIP is used instead: one `search()` per user with `IDSelectorNot(IDSelectorBatch(M_u))`, or k = 500 + max_u \|M_u\| per chunk followed by dropping the masked ids, which is equivalent to masking first. FAISS IVF and HNSW are reported as an ANN study (P4-ann). Embeddings are L2-normalised at export, with a unit-norm test. |
| D-009 | Baselines | DECIDED (Day 1) | — | MostPop (as-of popularity, design.md §10) and EASE (Steck 2019; minutes on the laptop) join every results table; ItemKNN if time permits. EASE memory: Gram 749 MB fp32 at 13,681 items; peak ~2.2 GB (G, its inverse and LAPACK's working copy), ~5 GB at 20,720 items; build XᵀX and invert in float32. BM25 is a content baseline, not a CF baseline. |
| D-010 | BM25 and LLM evaluation | DECIDED (Day 1) | — | Item document = `title_clean` + genres + top-k genome tags + train-period user tags, excluding the evaluated user's own tags (the genome covers 74.4% of kept items and 99.2% of test events); query = the same fields over the user's last ≤ 20 as-of positives; scoring by a scipy.sparse BM25 matrix product; fusion by RRF (k = 60) and a per-user z-scored convex mix with α tuned on V_A. An offline "CTR delta" is not measurable, so ablation (c) is tag-as-query known-item retrieval. LLM responses are cached by sha256(model_id + prompt + params). |
| D-011 | Timestamps | DECIDED (Day 1) | — | For 51.6% of users the whole positive history spans ≤ 1 hour (63.5% within a day), and 24.2% have a tie at some split boundary: "most recent" mostly means "last rated in a sign-up session". The task is never described as next-movie prediction; metrics are also split by tie / no-tie and by session span; recency features rank behind popularity, genre, genome and history-similarity features. |
| D-012 | Schedule | DECIDED (Day 1) | — | Day 2 = features + as-of accessor + leakage canary + metrics/bootstrap + MostPop/EASE + BM25 documents + a hello-world Streamlit deploy. Days 7 and 8 swap: BM25 and hybrid (ablation (b)) on Day 7, LLM expansion (ablation (c)) on Day 8. Cut order if late: BGE-M3 (genome PCA-64 instead), mined hard negatives, hybrid tuning, the live LLM path, the cloud GPU (train on MPS). |

## Planned run matrix

Nothing below has run. Every result cell is `pending` and will read `mean ± sd [lo, hi]`; "ref" marks the arm that paired Δs are computed against. Trained arms run 3 seeds; MostPop, EASE and BM25 are deterministic and run once. Until D-001 closes, the tables are planned under `LOO-full` and `LOO-asof` on test users, with selection on V_A; if D-001 adopts the Liang protocol, each table gains `Liang-SG-strict` rows with that protocol's metrics.

### Pipeline milestones

| Tag | Day | Run | Readout | Status |
|---|---|---|---|---|
| P2-anchors | 2 | MostPop (as-of popularity) and EASE, λ tuned on V_A (D-009) | HR@{10, 50, 100, 500}, NDCG@10 and MRR@10 under `LOO-full` and `LOO-asof`; EASE wall-clock and peak RSS | planned |
| P3-smoke | 3 | Two-tower round 1 on the laptop (MPS), B = 4,096 | the training loop end to end; re-measure E002 with the real training loop (s/epoch and peak memory, design.md §13) | planned |
| P3-base | 3 | Two-tower round 1: in-batch softmax + τ + logQ, B = 4,096, user tower per D-004 | V_A HR@50 with exact search; peak VRAM against design.md §13 | planned |
| P3-b16k | 3 | Same at B = 16,384 with 4,096 sampled negatives (the VRAM worst case) | V_A HR@50; peak VRAM | optional, cloud GPU |
| P4-liang | 4 | If D-001 adopts the Liang protocol: build that split (`min_item_interactions: 1`) and run MostPop and EASE under both `Liang-SG-strict` and `Liang-SG-fit` | Recall@20/50, NDCG@100 and NDCG@10 under each variant, each compared with its own variant's published Popularity and EASE figures (EASE Recall@50 0.521 under `Liang-SG-strict`, EASE NDCG@10 0.329 under `Liang-SG-fit`; design.md §9), as a check of the evaluation code that also measures the variant gap for two cheap models | conditional |
| P4-ann | 4 | ANN study: FAISS IVF and HNSW against exact search (D-008) | overlap@500 with the exact top-500; HR@500 Δ; p50 / p95 latency; build time; index size | planned |
| P6-rank | 6 | LightGBM LambdaRank over the Top-500 (D-006): same retriever run (the seed-42 run of B1), train + val masked, features as of the test cutoff, all 500 re-ranked | test NDCG@10 and HR@10 end to end over all test users (misses score 0), paired against NDCG@10 in retrieval order on the same candidates; V_B HR@500 and the group drop rate | planned |

### Ablation (a): negatives (Day 5, D-005)

Varied: the negatives and their logQ correction. Fixed: towers and features (D-004); B = 4,096; same 30-epoch budget per arm, best epoch selected on V_A HR@50; masking of every off-diagonal column that is a train positive of the row's user (~660 in-batch columns per row at B = 4,096, ~4 of them duplicates of the row's own positive; the row's own positive column is never masked); popularity counts from train only; users; protocol. τ is tuned on V_A for each arm. Three seeds per arm: 12 training runs, 15 with A3. The MostPop and EASE rows come from P2-anchors.

| Arm | Negatives | logQ correction | HR@50 | HR@500 | NDCG@10 (retrieval order) | HR@50, tail tercile | Δ HR@50 vs A1 [95% CI] |
|---|---|---|---|---|---|---|---|
| MostPop | anchor | — | pending | pending | pending | pending | pending |
| EASE | anchor | — | pending | pending | pending | pending | pending |
| A0 | in-batch | none | pending | pending | pending | pending | pending |
| A1 (ref) | in-batch | log p_batch(j) | pending | pending | pending | pending | — |
| A2 | MNS: in-batch + M = 2,048 uniform | log(B·p_batch(j) + M/N), N = 13,681 items | pending | pending | pending | pending | pending |
| A3 (optional) | A2 + negatives mined from ranks 101–500 of the round-1 model | as A2; the mined columns' correction is fixed on Day 5 | pending | pending | pending | pending | pending |
| A4 | full softmax over all items (4,096 × 13,681 logits: 224 MB fp32, ~0.73 GB at the backward peak) | none (no sampling) | pending | pending | pending | pending | pending |

Readings: A1 − A0 is the logQ correction, A2 − A1 the uniformly sampled negatives (the brief's question), A3 − A2 the mining, A4 − A1 the gap to the exact softmax. The tail tercile is the target item's train-popularity tercile (design.md §10).

### Ablation (b): dense vs BM25 vs hybrid (Day 7, D-009, D-010)

Varied: the candidate generator. Fixed: candidate depth 500, seen items masked before top-k, users, protocol. B1 and the hybrids carry the 3 seeds of the best arm of (a), chosen by mean V_A HR@50 over seeds.

| Arm | Retriever | HR@50 | HR@500 | NDCG@10 (retrieval order) | Δ HR@50 vs B1 [95% CI] |
|---|---|---|---|---|---|
| B-pop | MostPop, as-of popularity (anchor) | pending | pending | pending | pending |
| B-ease | EASE (anchor) | pending | pending | pending | pending |
| B0 | BM25, documents and queries per D-010 | pending | pending | pending | pending |
| B1 (ref) | dense two-tower, best arm of (a), exact search | pending | pending | pending | — |
| B2-rrf | hybrid: reciprocal rank fusion of B0 and B1, k = 60 | pending | pending | pending | pending |
| B2-cvx | hybrid: per-user z-scored convex mix of B0 and B1, α tuned on V_A | pending | pending | pending | pending |

The ranker (P6-rank) trains on the candidates of B1's seed-42 run (pre-declared, not the best seed; D-006); whether it is re-trained on a hybrid's candidates is decided from this table.

### Ablation (c): LLM query expansion on tag-as-query known-item retrieval (Day 8, D-010)

Benchmark: ~2,000 (user, tag) query events from `tags.parquet` (final count `pending` until built). Query = the tag text; relevant = every movie that user tagged with it; history as of the tag event; the user's own tags are excluded from item documents. Candidate masking is fixed when the benchmark is built: the relevant movie is usually one the user has already rated, so the `LOO-*` seen-item mask cannot apply unchanged. The bootstrap is a cluster bootstrap: it resamples users together with all their (user, tag) query events, since one user contributes many events. Responses are cached under `artifacts/llm_cache/`; `claude-haiku-5-5` rejects non-default sampling parameters, so the cache, not a sampling setting, makes the LLM arms reproducible.

| Arm | Query path | HR@10 | HR@50 | Δ HR@50 vs C0 [95% CI] | Grounding rate | Latency p50 / p95 | Cost per 1K queries |
|---|---|---|---|---|---|---|---|
| C-pop | MostPop as of the event (anchor; ignores the query) | pending | pending | pending | — | pending | — |
| C-ease | EASE over the as-of history (anchor; ignores the query) | pending | pending | pending | — | pending | — |
| C0 (ref) | BM25 over the raw tag | pending | pending | — | — | pending | — |
| C1 | BM25 over the `claude-haiku-5-5` expansion of the tag | pending | pending | pending | pending | pending | pending |
| C2 | dense, query-conditioned (definition fixed on Day 8; design.md §12) | pending | pending | pending | pending | pending | pending |
| C3 | hybrid of C1 and C2: RRF and convex, as in (b) | pending | pending | pending | pending | pending | pending |

C1 − C0 isolates the expansion. The grounding rate is the share of expansions that resolve to catalogue items.

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
