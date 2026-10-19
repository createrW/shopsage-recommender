# ShopSage Recommender — Technical Design

**Status:** Day 1, revised 2026-10-18 · **Owner:** Yi Zhao (USC) · **Companions:** [api_spec.md](api_spec.md) (artifacts, metric definitions, environment), [experiment_log.md](experiment_log.md) (protocol labels, runs, decisions)

> **Numbers policy.** No model-quality result appears here. Every number is a dataset fact from the Day-1 run and its EDA, an engineering figure marked **measured**, a cited literature figure (§9.2), an arithmetic or roofline figure marked **estimate**, a vendor price or platform limit as the vendor publishes it (Modal, Streamlit Community Cloud, Anthropic), or a **TARGET** from the owner's brief. Result cells read `pending` until a logged run in experiment_log.md produces them.

**Decisions.** D-001 protocol (§9) · D-002 item filter (§8) · D-003 demo licence (§16) · D-004 user tower (§4.1) · D-005 negatives (§4.3) · D-006 ranker data (§6) · D-007 metrics and CIs (§10) · D-008 retrieval (§5) · D-009 baselines (§10) · D-010 BM25 and LLM evaluation (§11, §12) · D-011 timestamps (§8.3) · D-012 schedule (§17). Open: D-001 (due Day 4), D-003 (before Day 9), D-004 (due Day 3); the rest are decided (D-005 and D-006 revised 2026-10-18).

## 1. Problem framing

- **Task.** Implicit-feedback top-N recommendation. A positive is a rating ≥ 4.0. Given a user's history as of a cutoff, rank the full filtered catalogue (13,681 items) and return a Top-10. The held-out "most recent" positive is usually the last rating of a sign-up session (§8.3), so the task is never described as next-movie prediction.
- **Why MovieLens-20M.** A public stand-in for implicit-feedback e-commerce: timestamped, with titles, genres, user tags and a tag genome for lexical baselines and LLM grounding, and large enough (20,000,263 ratings, 138,493 users) that a ranker cannot score every user-item pair.
- **Scope.** A production-style offline pipeline: two-tower retrieval served by exact search (with an ANN study), a LightGBM re-ranker, MostPop, EASE and BM25 baselines, an optional LLM query path and a CPU-only demo. Out of scope: online serving (no FastAPI/gRPC), deep rankers (DIN/MMoE), RL, A/B infrastructure.
- **Metrics.** HR@K, NDCG@10 and MRR@10 under leave-one-out; Recall@20/50, NDCG@100 and NDCG@10 under the Liang et al. protocol; paired bootstrap CIs over users and ≥ 3 training seeds per arm (§10). Which protocol is the headline is open (§9).
- **TARGETS (owner's brief, not results).** Recall@50 0.42 → 0.58; NDCG@10 0.32. §9.4 checks them against the literature and re-labels them: under the Liang protocol (whose Recall@50 anchors are strict-variant, `Liang-SG-strict`), Recall@50 ≥ 0.42 for round 1 with 0.50–0.52 as the stretch, and NDCG@10 0.32 as a stretch; 0.58 exceeds every published figure under that protocol.

## 2. Architecture

```text
ratings.csv movies.csv tags.csv genome-*.csv            data/raw/ml-20m (never committed)
     |  shopsage prepare                               Polars on the laptop CPU: 3-9 s, 2.3 GB peak RSS (measured)
     v
data/processed/  ratings | movies | tags (raw: unsafe for features, §8.2) | mappings/ | splits/ + meta.json
     |  features.asof.history(stage)                   Day 2: the only path from raw tables to features (§7)
     v
Two-Tower training                                     PyTorch: MPS smoke runs on the laptop; B >= 8,192 and MNS on a cloud GPU
  round 1: in-batch softmax + temperature + logQ + accidental-hit masking
  round 2: mixed negative sampling, in-batch + M = 2,048 uniformly sampled negatives (the brief's "hard negatives", D-005)
     |  export, L2-normalised
     v
item_embeddings.npy --> exact top-k (matmul / IndexFlatIP) --+    FAISS IVF / HNSW: ANN study only (D-008)
user tower (inductive, D-004) -------------------------------+--> Top-500 --> LightGBM LambdaRank --> Top-10
BM25 over item documents (content baseline, D-010) ----------+    (dense | BM25 | hybrid)
MostPop and EASE: calibration baselines in every results table (D-009)

query path (optional): text --> Claude Haiku expansion --> BM25 grounding --> mean item embedding --> exact top-k
demo: Streamlit over precomputed artifacts (numpy, faiss-cpu, LightGBM; no torch, no GPU)
```

## 3. Why two stages

| | Retrieval | Ranking |
|---|---|---|
| Scope | full catalogue (13,681 items) | Top-500 candidates per user |
| Scorer | dot product of independently computed user and item vectors | LightGBM over user, item and user × item features |
| Cost per user | one 13,681 × 128 matrix-vector product | 500 rows × trees |
| Feature constraint | must factorise into a user side and an item side | none: cross features and explicit negatives allowed |
| Optimised for | recall (HR@50; HR@500 caps the ranker) | precision at the top (NDCG@10) |

- Cross features cannot be computed for every pair (136,674 users × 13,681 items ≈ 1.87 × 10⁹ per scoring pass), so retrieval cuts the set to 500 per user and the ranker only reorders it; HR@500 is reported next to every NDCG@10. The split mirrors industrial candidate-generation and ranking stacks (Covington et al. 2016), which is the system this project demonstrates.

## 4. Retriever: Two-Tower

### 4.1 Towers (user tower: D-004, OPEN, due Day 3)

| Tower | Inputs (features in §7) | Output |
|---|---|---|
| User, recommended (D-004) | mean of the item-tower embeddings of the user's last ≤ 50 as-of positives ⊕ behavioural features (log activity, mean rating, genre affinity, recency) ⊕ optional `user_idx` embedding under ID-dropout 0.3 | MLP → 128-d → L2-normalise |
| User, the brief's design (ablation arm) | `user_idx` embedding ⊕ the same behavioural features | MLP → 128-d → L2-normalise |
| Item | `item_idx` embedding ⊕ genre multi-hot ⊕ year bucket ⊕ genome PCA-64 and missing-genome flag ⊕ optional BGE-M3 vector (frozen 1024-d, linear projection) | MLP → 128-d → L2-normalise |

- **No demographics.** ML-20M ships none (ML-100K and ML-1M do), so the user side is history and behaviour, with or without an ID.
- **Why an inductive tower (recommendation).** It works unchanged under leave-one-out, the Liang protocol (whose held-out users have no trained ID row in `Liang-SG-strict`) and any temporal protocol, and it serves visitor profiles, which the demo needs (D-003), through the missing-feature path (§7); their quality is not measured. The demo runs it in ~15 lines of numpy over the 13,681 × 128 item embeddings (~7 MB fp32) plus the MLP weights, so no 136,674 × 128 user table (~70 MB) ships; a parity test requires the numpy and torch forward passes on 1,000 users to agree to a max abs diff < 1e-5, checked with `torch.testing.assert_close(..., equal_nan=False)` plus an explicit no-NaN assertion. In training, the target item is removed from its own pooled history.
- **The brief's `user_id` embedding is kept,** as an optional input under ID-dropout (p = 0.3: the row is swapped for a shared OOV row, so the tower cannot rely on it alone) and as an ablation arm, so its value is measured rather than assumed.
- **No torchrec.** All parameters total ~78 MB (§13); plain `nn.Embedding` on one device is enough.

### 4.2 Loss: in-batch sampled softmax with temperature and logQ correction

For a batch of B (user, positive item) pairs with L2-normalised vectors `u_i`, `v_j`:

```text
s_ij = <u_i, v_j> / tau  -  log(B * p_batch(j))      p_batch(j) = count of item j in train.parquet / n_train
L    = -(1/B) * sum_i log( exp(s_ii) / sum_{j unmasked} exp(s_ij) )
```

- **Softmax over in-batch negatives.** B − 1 negatives per positive for one B × B matrix product, closer to retrieval than pointwise BCE or pairwise BPR. The loss matters more than the negatives: BCE → sampled softmax moved SASRec's HR@50 on ML-20M from 0.447 to 0.549 in Zhai et al. (2023).
- **Temperature.** Cosine scores lie in [−1, 1], so at τ = 1 a softmax over thousands of candidates is nearly flat; a smaller τ concentrates gradient on high-scoring negatives. τ is tuned per arm on V_A (§6) (`pending`).
- **logQ correction** (Yi et al. 2019). In-batch negatives arrive in proportion to item frequency, which over-penalises popular items. Subtracting the log expected count from every logit, the positive's included, makes the sampled softmax approximate the full softmax. Offline, `p_batch` is exact from the train split.
- **Accidental hits.** At B = 4,096 a row has ~4 columns that duplicate its own positive, because the top-100 movies are 23.5% of positives. In every arm, each off-diagonal column that is a train positive of the row's user is masked (§4.3); the row's own positive column is never masked. That is ~660 columns per row at B = 4,096 (the row-weighted expected value on the Day-1 train split; median 571, p90 1,298).

### 4.3 Round 2: mixed negative sampling (D-005, DECIDED)

The brief calls this round "hard negatives". In-batch negatives are only the items that occur as positives in the batch, so tail items rarely act as negatives and their scores are weakly constrained. Uniformly sampled negatives reach the tail: at B = 4,096 and M = 2,048 the bottom popularity tercile (4,561 items, ≤ 23 train positives each) gets 11.5% of all negative columns, against 0.5% with in-batch negatives only (expected values on the Day-1 train split). They are not hard in the model-score sense, so this document calls the round mixed negative sampling (MNS, Yang et al. 2020): in-batch negatives plus M = 2,048 uniformly sampled negatives per batch, logQ over the mixed proposal. "Hard negatives" is kept for the optional model-mined arm.

- **Mechanics.** Each batch adds M = 2,048 negatives shared by all rows, drawn uniformly over the catalogue: q(j) = 1/N_items = 1/13,681 (D-005 as revised on 2026-10-18; config `sampling: popularity_power`, `alpha: 0.0`). The exponent stays a knob; `alpha: 0.75`, the smoothed unigram of Mikolov et al. 2013, would lift the tail tercile's share of negative columns only to 1.0%. Logits become B × (B + M), and every logit of item j subtracts `log(B·p_batch(j) + M·q(j))`, its expected count under the mixed proposal. M = 0 is round 1.
- **Accidental-hit masking** covers every column, in-batch or sampled, that is a train positive of the row's user, except the row's own positive column (the diagonal), which is never masked: ~660 in-batch columns per row at B = 4,096 (§4.2). The lookup is a 136,674 × 13,681 bitset (234 MB) or sorted int64 `(user_idx << 32) | item_idx` keys searched with `searchsorted`.
- **No label leakage.** Popularity counts and masks come from the train split only, and val/test items are never excluded from the negatives: excluding them would leak the labels.
- **Optional mined hard negatives** come from ranks 101–500 of the round-1 model and are mixed with in-batch ones, not used alone (Huang et al. 2020).
- **Ablation (a) arms** (§11): A0 in-batch; A1 in-batch + logQ; A2 MNS: in-batch + M = 2,048 uniform, with the correction `log(B·p_batch(j) + M/N_items)`; A3 mined (optional); A4 full softmax, which is cheap at 13,681 items (4,096 × 13,681 logits: 224 MB fp32, ~0.73 GB at the backward peak) and is the reference the samplers approximate. Same 30-epoch budget per arm, best epoch selected on V_A HR@50; τ tuned on V_A; 3 seeds each.

### 4.4 Training plan (Day 3)

| Setting | Plan |
|---|---|
| Batch size | 4,096 (laptop MPS smoke run, cloud GPU); 8,192, 16,384 and every sampled-negative run on the cloud GPU (§13) |
| Optimiser | `AdamW(fused=True)`, no weight decay on embedding tables; lr and τ tuned on V_A; the epoch budget is fixed (below) |
| Precision | bf16 autocast if `torch.cuda.is_bf16_supported(including_emulation=False)` (compute capability ≥ 8.0), else fp16 + `GradScaler` (the T4 is sm_75 and has no native bf16) |
| Model selection | a fixed 30-epoch budget per run, the same for every arm, with no early stopping; the checkpoint with the best V_A HR@50 (exact search) is kept (`checkpoint_metric: va_hr@50`) |
| Seeds | ≥ 3 per arm, reported as mean ± sd (D-007) |
| Tracking | Weights & Biases: config, losses, V_A metrics, peak memory, throughput |

One retriever per protocol, trained on its train split only, serves val (model selection, ranker labels) and test (final numbers).

## 5. Retrieval: exact search, with an ANN study (D-008, DECIDED)

Exact scoring is the default: batched matmul → `masked_fill(M_u, −inf)` → `torch.topk` / `np.argpartition`, in chunks of ≤ 8,192 users (§13), where M_u is user u's masked item set. The 13,681 × 128 fp32 item vectors take 7.0 MB, and scoring every user is ~2.4 × 10¹¹ multiply-adds: seconds on a CPU (estimate). `IndexFlatIP` is an equivalent exact path when it runs one `search()` per user with `IDSelectorNot(IDSelectorBatch(M_u))`, or k = 500 + max_u |M_u| per chunk followed by dropping the masked ids, which is equivalent to masking first.

- **Seen items are set to −inf before top-k.** 1,293 users have more than 500 train positives (max 3,148), so "take the top 500, then drop seen items" would leave them with fewer than 500 candidates, or none.
- **Embeddings are L2-normalised at export,** guarded by a unit-norm test. The metric is inner product (cosine on unit vectors), matching training; FAISS ids are `item_idx`.
- **ANN study (Day 4; values `pending`).** FAISS IVF (`IndexIVFFlat`, or `IndexIVFPQ` with 16 B PQ16 codes; nlist ≤ 350 at 13,681 items, since FAISS wants ≥ 39 training points per centroid) and HNSW (`IndexHNSWFlat`, ~256 B of links per vector at M = 32) against exact search: overlap@500 with the exact top-500, HR@500 delta, p50/p95 latency at batch 1 and 1,024, build time and index size. FAISS is studied because a 10⁷-item catalogue would need it, not because this one does.

## 6. Ranker: LightGBM LambdaRank (D-006, DECIDED)

**Why trees.** Deep rankers are out of scope, and gradient-boosted trees are a strong default on ~25 heterogeneous tabular features. LightGBM trains on CPU, the model is a small text file, scoring 500 rows needs no torch (so it runs in the demo), and per-prediction SHAP values (`pred_contrib=True`) give the demo a "why this item" panel.

| Aspect | Plan (Day 6) |
|---|---|
| Val users | a disjoint split seeded by `val_partition.seed` = 42, fixed across training seeds, into V_A (retriever selection: best epoch, τ, hybrid α) and V_B (ranker labels); 10% of V_B is held out for LightGBM early stopping |
| Candidates and labels | the retriever's Top-500 for each V_B user's val event, as-of seen items masked; label 1 for the val item, 0 otherwise; low ratings feed features only, never labels |
| Missed positives | never injected: the group is dropped, and the drop rate (1 − HR@500 on V_B) is logged |
| Negatives kept | all 500 candidates per group (the test-time distribution); fallback: ranks 1–50 plus 50 random from 51–500 with row weight 9 on the random ones (approximate) |
| Objective | `lambdarank`, one query group per user, NDCG@10 |
| Test | same retriever; train + val masked; features as of the test cutoff; all 500 re-ranked; NDCG@10 and HR@10 over all test users (a miss scores 0), next to NDCG@10 in retriever order on the same candidates. Test runs once per final config and every test run is logged; any number conditioned on a retrieval hit is labelled as such |
| Size | training table, the default: 20K users (the cap) × 500 ≈ 10M rows ≈ 2 GB RAM (estimate); all 136,674 × 500 = 68M test rows need ~9.6–16 GB (estimate), so test scoring runs in user chunks |

- **One label set per decision.** V_A selects the retriever and V_B trains the ranker, so the ranker never learns from candidate lists whose retriever was tuned on the same labels.
- **No injected positives.** At test time a missed positive is simply absent; injecting it in training teaches the ranker a positive it never meets (no retrieval score, an out-of-distribution rank), and a group without a positive gives LambdaRank no gradient anyway.
- **All 500 candidates in training.** LambdaRank learns the positive-to-negative odds at each rank, and `retrieval_rank` is a feature. Keeping 11% of the rank-51–500 negatives in training and 100% at test would inflate those odds about 9× for ranks 51–500 and over-promote deep candidates at test; training on the test-time candidate distribution removes the skew exactly. If RAM is tight, the fallback is the 50 + 50 subsample with a LightGBM row weight of 450/50 = 9 on the random negatives, an approximate inverse-sampling correction because lambdarank applies row weights after the pairwise lambdas are accumulated.
- **One retriever run.** Candidates come from the seed-42 run of B1, the best arm of ablation (a) chosen by mean V_A HR@50 over seeds (the seed is pre-declared, not the best seed), referenced by its run id (`artifacts/checkpoints/<run_id>/`, api_spec.md §7). The ranker's 3 seeds are LightGBM seeds on that one run; retriever-seed variance is measured in ablation (a).

## 7. Feature plan and the as-of rule (Day 2)

**As-of rule.** `features.asof.history(stage='val'|'test')` (api_spec.md §5.6) is the only allowed way to build features, because the raw tables leak (§8.2):

- history positives come from split membership, not timestamps: train for the val stage, train + val for the test stage (a train positive that shares the cutoff second is history);
- raw ratings and tags are filtered to `timestamp < cutoff`, the user's val (val stage) or test (test stage) timestamp, and item statistics come from the train split only;
- candidates mask the history positives and every item the user rated before the cutoff (ML-20M holds one rating per user and movie, so this never hides the held-out item);
- a leakage-canary test plants events at and after each user's cutoff and asserts that no feature changes.

| Group | Feature | Source, as of the cutoff | Encoding |
|---|---|---|---|
| User | history | last ≤ 50 positives | mean-pooled item-tower embeddings (D-004); `user_idx` embedding optional (§4.1) |
| User | activity, mean rating, genre affinity | ratings and positives before the cutoff | log1p and standardised; normalised genre histogram |
| User | recency | time since the previous event | log-bucketed; deprioritised (D-011) |
| Item | `item_idx`, genres, year | `item_map`, `movies.parquet` | embedding; multi-hot; year bucket with a null bucket |
| Item | genome | `genome-scores.csv` (covers 74.4% of kept items) | PCA-64 plus a missing-genome flag |
| Item | text (optional; first to cut) | BGE-M3 over "title (year). genres. top genome tags" | frozen 1024-d, projected in the tower |
| Ranker | retrieval and item | dense score and rank, BM25 score; `item_popularity_train` and `item_mean_rating_train` (log count and mean rating on the train split, both stages), year bucket, genres | numeric / categorical |
| Ranker | user × item | genre affinity · item genres; similarity of the candidate to the as-of history; mean-rating and popularity gaps; the user's low-rating rate on the item's genres | numeric |

- **Missing behavioural features.** A user with no raw rating strictly before the cutoff (61 users at the val stage and 51 at the test stage, whose history rows all share the cutoff second; and every visitor profile) gets `mean_rating` = the mean of the history positives' ratings if any is known, else the training-set median, with a `mean_rating_missing` flag, and `recency_days` in an "unknown / same-session" bucket. The user tower trains with feature dropout p = 0.3 on the behavioural block (mirroring ID-dropout, §4.1), so this path is learned. The 61 + 51 users join the leakage-canary fixtures.
- **Priority (D-011).** Popularity, genre, genome and history-similarity features come before recency features, because "recent" mostly means "same sign-up session".
- **Accepted mild global leaks,** documented rather than removed: the tag genome (computed by GroupLens over the full history) and the global item filter (≥ 5 positives counted over all time).

## 8. Data decisions

Source: MovieLens-20M, archive md5 `cd245b17a1ae2cc31bb14903e1204af3`. Raw contents: 20,000,263 ratings (0.5–5.0 in half-star steps), 27,278 movies, 465,564 tag applications, 138,493 users, 26,744 distinct rated movies, 1995-01-09 to 2015-03-31.

**Day-1 run (current config, `splits/meta.json`).** 136,674 users, 13,681 items, 9,977,455 positives; 9,704,107 train, 136,674 val and 136,674 test rows; density 0.5336%; 20,580 users (15.06%) with the test and val events in the same second; 0 cold test items. `prepare` runs in 3.3 s with a warm file cache (9.1 s cold) at 2.3 GB peak RSS on the M5 (measured with `/usr/bin/time -l`, Polars 2.0.0).

| Decision | Value | Rationale |
|---|---|---|
| Positive | rating ≥ 4.0 | the Liang et al. (2018) binarisation of ML-20M |
| Item filter (D-002, DECIDED) | drop items with < 5 positives | a single-pass ≥ 5 item filter on positives (not a k-core: 1 item ends below 5), chosen so every item ID embedding has training signal; it departs from both Liang et al. (no item filter) and the ML-20M leave-one-out papers in §9.2 (all 26,744 items); note below |
| User filter | then drop users with < 5 positives | ≥ 3 train positives per user under leave-one-out |
| Passes | one pass, items then users | Liang's order; not iterated to a k-core, so some items can end below 5 |
| Raw tables | all 20,000,263 ratings in `ratings.parquet`; `tags.parquet` | low ratings and tags are features, read only through the as-of rule (§7, §8.2) |
| IDs | contiguous `user_idx` / `item_idx`, ascending by raw id | embedding row k is index k; stable across runs |
| Movie text | `title_clean` (year stripped), `year` (nullable), `genres` as a list (empty for "(no genres listed)") | the year regex also reads "Title (2007-)", "Title (2009– )", "Title (1975-1979)" (first year) and a stray trailing ")" as in "Mona and the Time of Burning Love (...) (1983))" |
| Demographics | none | ML-20M has none |

> **Comparability (D-002).** Liang et al.'s released preprocessing applies no item filter (`min_sc = 0`): 136,677 users, 20,720 items and 9,990,682 positives, and 20,108 items after their user split. Our item threshold of 5 therefore departs from Liang, and from the ML-20M leave-one-out papers of §9.2, which keep all 26,744 items; it is a single pass, not an iterated 5-core (Passes, above). D-002 keeps 5 for the leave-one-out split; the Liang variant runs with `min_item_interactions: 1`, which reproduces their filter (on positives, 1 is a no-op; `DataConfig` rejects 0).

### 8.1 Split and tie-breaking

- **Leave-one-out by time over positives.** Per user, test is the most recent positive, val the second most recent and train the rest, so `n_val = n_test = n_users`.
- **Order key** = (`timestamp`, `tie_key`); "most recent" is the largest key. `tie_key = splitmix64(((userId << 32) | movieId) XOR splitmix64(seed))`, computed in numpy with explicit 64-bit wrap-around over the raw ids before id remapping and stored as a UInt64 column. `remap_ids` passes it through after `user_idx` / `item_idx`; `leave_one_out_split` uses it when present and otherwise computes the same hash over (`user_idx`, `item_idx`). The split outputs drop it, so `splits/*.parquet` stay `user_idx` Int32, `item_idx` Int32, `rating` Float32, `timestamp` Int64. The code (`data/split.py`) is canonical.
- **Why this hash.** Ties at the boundary are common (§8.3). Ordering by movieId would push higher ids, which grow with the catalogue's addition date, into test, and file order is not reproducible across readers. The earlier affine-mod-prime hash favoured the higher of two adjacent item ids ~81% of the time; SplitMix64 is non-linear and favours neither (`test_tie_break_is_not_biased_toward_any_item_id`). Polars' `Expr.hash` is not used because it is not stable across Polars versions. Hashing raw ids keeps the tie order independent of the filter thresholds. A golden test pins the exact val/test pairs of a fixed 20-row table.
- **`meta.json`** (the code is canonical; schema in api_spec.md §3): `strategy`, `seed`, `tie_break` (a description of the hash), `n_users`, `n_items`, `n_interactions`, `n_train`, `n_val`, `n_test`, `density`; `users_with_boundary_tie` (int: test and val share a timestamp) and `boundary_tie_fraction`; `users_with_val_train_tie` (int: the val timestamp equals the newest train timestamp); `users_with_any_boundary_tie` and `any_boundary_tie_fraction`; `cold_test_items`, `cold_val_items`; `split_sha256` (sha256 over the sorted (`user_idx`, `item_idx`) pairs of val and test, so two machines can confirm they hold the same split); plus `dataset`, `config`, `n_ratings_raw`, `n_movies`, `n_tags`, `created_at`, `versions`.
- **Sensitivity and limits.** Only users tied at a boundary (24.2%) can change under another seed, and metrics are also reported split by tie / no-tie (D-011). Leave-one-out is not a global temporal split: other users' train events post-date a user's test event, so the model and global statistics see the future (§9.5).

### 8.2 Raw tables are unsafe for features

`ratings.parquet` and `tags.parquet` are raw. They contain every val and test positive; for 36.7% of users they contain ratings after the test event; and 711 users tagged their own test movie. A feature read straight from them leaks the label, so the as-of rule (§7) is the only way in: positives from split membership, raw rows only with `timestamp < cutoff`, item statistics from train, candidates masked as of the cutoff, and a leakage-canary test.

### 8.3 Timestamp semantics (D-011, DECIDED)

For 51.6% of users the whole positive history spans ≤ 1 hour (63.5% within a day), and 24.2% of users have a tie at a split boundary. "Most recent" therefore mostly means "last rated in a sign-up session", not "watched last". So the task is never described as next-movie prediction; metrics are also reported split by tie / no-tie and by session span; and recency features rank behind popularity, genre, genome and history-similarity features (§7).

## 9. Evaluation protocol — OPEN DECISION (D-001, due Day 4)

Fixed whatever the outcome: full-corpus ranking (no sampled-negative metrics; Krichene & Rendle 2020); each user's known items (the as-of history under A, the fold-in positives under B) set to −inf before top-k (§7); user-averaged metrics; paired bootstrap CIs and ≥ 3 seeds (§10); MostPop and EASE run in-repo under the same protocol (D-009). Protocol labels follow experiment_log.md: `LOO-full`, `LOO-asof`, `Liang-SG-strict` (held-out users never trained on; Liang et al. 2018 §4.3) and `Liang-SG-fit` (their fold-in rows are also in training; Milogradskii et al. 2024).

### 9.1 Candidate protocols

| | A. Leave-one-out by time (Day-1 split): `LOO-full`, `LOO-asof` | B. Liang et al. (2018) strong generalisation: `Liang-SG-strict`, `Liang-SG-fit` |
|---|---|---|
| Held out | each user's most recent positive (val: second most recent) | 10K val and 10K test users; a random 20% of each held-out user's positives |
| Training data | every user's train rows | the remaining users (`Liang-SG-strict`); in `Liang-SG-fit` also the held-out users' fold-in 80% |
| Reported | HR@{10,50,100,500}, NDCG@10, MRR@10 (Recall@K = HR@K) | truncated Recall@20 and Recall@50, NDCG@100, NDCG@10 |
| Catalogue | 13,681 items (item filter 5) | `min_item_interactions: 1`, Liang's filter: 20,720 items, 20,108 after the user split |
| Temporal realism | per user only; global look-ahead (§9.5) | none (random holdout) |
| Cost | built on Day 1 | new split artifact and fold-in path |

### 9.2 Verified literature anchors on ML-20M

Leave-one-out, full corpus, one held-out item per user. These papers count every rating as an interaction (26,744 items); "—" means not quoted here, and no verified NDCG@10 under this protocol exceeds 0.211 (HSTU-large):

| Model | HR@10 | HR@50 | NDCG@10 | Source |
|---|---|---|---|---|
| BPR-MF, the only non-sequential anchor | 0.081 | not reported; between HR@10 and HR@100 (0.337) | 0.039–0.041 | Klenitskiy & Vasilev 2023; Petrov & Macdonald 2022 |
| BERT4Rec | 0.282 | — | 0.170 | Klenitskiy & Vasilev 2023 |
| SASRec+ (full softmax) | 0.298 | — | 0.183 | Klenitskiy & Vasilev 2023 |
| SASRec+3000 (3,000 sampled negatives) | 0.309 | — | 0.187 | Klenitskiy & Vasilev 2023 |
| SASRec, 2023 recipe | — | 0.550 | — | Zhai et al. 2024 |
| HSTU | — | 0.589 | 0.188 | Zhai et al. 2024 |
| HSTU-large | — | 0.615 | 0.211 | Zhai et al. 2024 |

Liang strong generalisation. **Recall@50**, strict variant (`Liang-SG-strict`: held-out users never trained on; truncated: hits / min(K, n_u)): Popularity 0.235, WARP 0.466, SLIM 0.495, WMF 0.498, EASE 0.521 (Steck 2019, Table 1), CDAE 0.523, Mult-DAE 0.524, iALS 0.532, Mult-VAE 0.537, RaCT 0.543, H+Vamp 0.551, VASP 0.552, RecVAE 0.553 (highest verified). **NDCG@10**, fold-in-in-training variant (`Liang-SG-fit`; Milogradskii et al. 2024): ItemPop 0.128, BPR-MF 0.302, Mult-VAE 0.308, EASE 0.329.

### 9.3 Context, not comparable

No figure in §9.2 is like-for-like with this repo: preprocessing, catalogue and protocol all differ.

| Source | Preprocessing | Protocol | Figures used |
|---|---|---|---|
| Liang et al. 2018, WWW, [arXiv:1802.05814](https://arxiv.org/abs/1802.05814) | rating ≥ 4; users ≥ 5; no item filter: 136,677 users, 20,720 items (20,108 after the split) | strong generalisation: 10K + 10K held-out users, random 80/20 fold-in/target, held-out users never trained on (`Liang-SG-strict`) | SLIM, WMF, CDAE, Mult-DAE, Mult-VAE |
| Steck 2019, WWW, [arXiv:1905.03375](https://arxiv.org/abs/1905.03375); Rendle et al. 2022 (iALS); Lobel et al. 2020 (RaCT); Kim & Suh 2019 (H+Vamp); Vančura & Kordík 2021 (VASP); Shenbin et al. 2020 (RecVAE) | Liang's | Liang's (`Liang-SG-strict`) | Popularity, EASE; iALS, RaCT, H+Vamp, VASP, RecVAE; WARP (RecVAE Table 1, from RaCT / Lobel et al. 2020) |
| Milogradskii et al. 2024, RecSys | Liang's | Liang's users; the fold-in rows are also in training (`Liang-SG-fit`) | NDCG@10 |
| Klenitskiy & Vasilev 2023, RecSys, [arXiv:2309.07602](https://arxiv.org/abs/2309.07602); Petrov & Macdonald 2022, RecSys, [arXiv:2207.07483](https://arxiv.org/abs/2207.07483) | every rating is an interaction; 26,744 items | leave-one-out by time, full corpus | BPR-MF, BERT4Rec, SASRec+ |
| Zhai et al. 2024, ICML, [arXiv:2402.17152](https://arxiv.org/abs/2402.17152); Zhai et al. 2023, KDD | as above | as above | SASRec, HSTU; BCE vs sampled softmax |
| This repo, A | rating ≥ 4; items ≥ 5, then users ≥ 5: 136,674 users, 13,681 items | leave-one-out by time over positives, full corpus, `LOO-full` and `LOO-asof` (§9.5) | — |
| This repo, B | rating ≥ 4; users ≥ 5; `min_item_interactions: 1` | Liang's: `Liang-SG-strict` (recommended headline); MostPop and EASE also under `Liang-SG-fit` | — |

### 9.4 The targets against the anchors

- **Under A, no published model reaches the targets as a set.** NDCG@10 0.32 is above every verified figure (≤ 0.211). Recall@50 0.58 is reached only by HSTU-class generative sequential models (0.589–0.615), whose NDCG@10 is 0.188–0.211. Recall@50 0.42 is above the only non-sequential anchor quoted (BPR-MF, HR@100 0.337) in a non-comparable setup; in-repo EASE under `LOO-full` (P2-anchors in experiment_log.md) shows whether a non-sequential model reaches it.
- **Under B,** against the `Liang-SG-strict` Recall@50 anchors, Recall@50 0.42 for round 1 is plausible (Popularity 0.235, WMF 0.498); 0.50–0.52 is a realistic stretch (WMF to EASE); 0.58 exceeds every published figure (≤ 0.553). NDCG@10 0.32 is about EASE's 0.329 under `Liang-SG-fit`, a stretch; a `Liang-SG-strict` NDCG@10 is compared with that fit-variant figure, with the caveat stated.
- **Re-labelled targets (B, `Liang-SG-strict`):** Recall@50 ≥ 0.42 for round 1, stretch 0.50–0.52; NDCG@10 0.32, stretch (against the fit-variant anchor). Nothing is a result until measured.

### 9.5 Global look-ahead and catalogue-as-of masking

Under A, averaged over test users, 53.7% of all train rows are newer than the evaluated user's own test event (none of that user's own train rows are), and for the median test event 56.8% of the kept catalogue had no rating yet in `ratings.parquet` (its first rating is later than the event). The model, the logQ counts and popularity statistics see the future, and the unmasked candidate set includes items no one in the dataset had rated at that moment; this is the global look-ahead described by Ji et al. (TOIS 2023). Every leave-one-out result is reported twice: `LOO-full` (unmasked, for literature comparability) and `LOO-asof` (catalogue-as-of: items first rated after the cutoff are also masked). MostPop uses as-of popularity (§10) in both.

### 9.6 Recommendation (owner's call on Day 4)

- **Headline:** `Liang-SG-strict`, with Recall@20, Recall@50, NDCG@100 and NDCG@10 and paired bootstrap CIs over users. That is where the Recall@50 anchors live (§9.2), it gives the re-labelled targets a meaning, and the inductive user tower (D-004) scores users it never trained on. NDCG@10 is compared with the `Liang-SG-fit` anchors, with that caveat stated.
- **Secondary table:** the Day-1 split under `LOO-full` and `LOO-asof` (§9.5), with HR@{10,50,100,500}, NDCG@10 and MRR@10.
- **Calibration anchors:** MostPop and EASE run in-repo under both protocols, and under B in both variants, each compared with its own variant's published figure (EASE Recall@50 0.521 under `Liang-SG-strict`, NDCG@10 0.329 under `Liang-SG-fit`), which also measures the variant gap for two cheap models; a large gap points to a split or metric bug and is resolved before any model figure is reported. EASE's Gram matrix is 749 MB (fp32) at 13,681 items, with a peak of ~2.2 GB (G, its inverse and LAPACK's working copy), and takes minutes on the laptop (estimate); at Liang's 20,108–20,720 items the Gram is ~1.6–1.7 GB and the peak ~5 GB at 20,720 items (estimate). XᵀX is built and inverted in float32.
- **Criteria:** comparability with the literature, temporal realism, implementation cost before Day 5, fit with D-004, and honest reporting against the targets. The outcome is logged in experiment_log.md, and every results table names its protocol.

## 10. Metrics, confidence intervals and baselines (D-007, D-009, DECIDED)

- **Definitions under A.** r is the 1-based rank of the held-out item among unmasked items. HR@K = 1[r ≤ K], which equals Recall@K with one relevant item; NDCG@10 = 1 / log2(r + 1) if r ≤ 10, else 0; MRR@10 = 1/r if r ≤ 10, else 0. K ∈ {10, 50, 100, 500}; HR@500 caps the ranker. **Under B:** truncated Recall@K = hits / min(K, n_u), and NDCG@K takes the ideal DCG over min(K, n_u) items. api_spec.md §8 specifies the formulas and the bootstrap procedure.
- **Paired bootstrap.** Per-user metric vectors are stored, so CIs are recomputed without re-scoring. Resample users and take the mean per-user difference between two arms, with the same resample indices for both: 2,000 resamples, percentile 95% CI, `numpy.random.default_rng(42)`, indices drawn in fixed chunks of 100 replicates to bound memory. Ablation (c) uses a cluster bootstrap: users are resampled together with all their (user, tag) query events, an event being a (user, tag) pair with its relevant movie set (~2,000 events, §11). A difference is claimed only if the paired user-bootstrap CI excludes 0 and the per-seed paired Δs (seed k of arm A vs seed k of arm B, same users) all have the CI's sign; tables also report the per-seed Δ range. Overlapping marginal CIs are not a test.
- **Seeds.** ≥ 3 training seeds per arm, reported as mean ± sd. At n = 136,674 the 95% CI half-width on HR@50 is about ±0.27 percentage points (binomial, p ≈ 0.5), so under leave-one-out seed-to-seed variance, not user sampling, is the binding uncertainty; under the Liang protocol (10,000 test users) the half-width is ~±1.0 point, so user sampling also matters.
- **Diagnostics (reported, not optimised).** HR@50 by the train-popularity tercile of the target item, catalogue coverage of the Top-10 lists, the HR@500 ceiling, and subgroups by tie / no-tie and by the span of the user's positives (≤ 1 hour, ≤ 1 day, longer; D-011).
- **Baselines (D-009).** MostPop and EASE (Steck 2019, λ tuned on V_A) join every results table, plus ItemKNN if time permits. MostPop ranks by **as-of popularity**: an item's positives with timestamp strictly before the user's cutoff. Only MostPop uses it; the ranker's `item_popularity_train` counts the train split instead (§7). BM25 is a content baseline, not a collaborative-filtering one.

## 11. Ablations

| | Question | Arms | Held fixed | Readout | Day |
|---|---|---|---|---|---|
| (a) | Which negatives? (D-005) | A0 in-batch; A1 in-batch + logQ (ref); A2 MNS: in-batch + M = 2,048 uniform, logQ `log(B·p_batch(j) + M/N_items)`; A3 mined (optional); A4 full softmax | towers and features, B = 4,096, same 30-epoch budget per arm (best epoch selected on V_A HR@50), masking of the user's train positives except the row's own positive column (~660 per row at B = 4,096), users, protocol; τ tuned per arm on V_A; 3 seeds per arm | HR@50, also by train-popularity tercile; paired Δ | 5 |
| (b) | Dense vs lexical vs hybrid (D-010) | B0 BM25; B1 dense, best arm of (a) (ref); B2 hybrid, both by RRF (k = 60) and by a per-user z-scored convex mix with α tuned on V_A | candidate depth 500, masking before top-k, users, protocol | HR@50 and HR@500; NDCG@10 after the ranker | 7 |
| (c) | Does LLM expansion help a text query? (D-010) | C0 BM25 over the raw tag (ref); C1 BM25 over the Haiku expansion; C2 dense, query-conditioned (§12); C3 hybrid of C1 and C2 | query events, history cutoff, item documents | HR@10 and HR@50, paired over query events, bootstrapped by user (§10); grounding rate; p50/p95 latency; cost per 1K queries | 8 |

- **BM25 documents (D-010).** Item document = `title_clean` + genres + top-k genome tags + train-period user tags, excluding the evaluated user's own tags (711 users tagged their own test movie). The genome covers 74.4% of kept items and 99.2% of test events. A user's query is the same fields over their last ≤ 20 as-of positives. Scores come from one scipy.sparse BM25 matrix product.
- **(c) benchmark.** The brief's "offline CTR delta" cannot be measured offline. Instead: tag-as-query known-item retrieval over ~2,000 (user, tag) query events; the query is the tag text, the relevant set is the movies that user tagged with it, the history is taken as of the event, and the user's own tags are excluded from the item documents. A user-written tag is a real query with a known target set, unlike queries built from the target's own genome tags (a rejected alternative), which are optimistic by construction. The tagged movie is usually already rated, so the leave-one-out seen-item mask cannot apply unchanged; the masking is fixed when the benchmark is built.
- **Sequencing (D-012).** BM25 documents are built on Day 2; Day 7 runs BM25 retrieval and hybrid (b); Day 8 adds LLM expansion (c) on top.

## 12. LLM query expansion (Day 8)

The demo host has no GPU and 690 MB of guaranteed RAM (§16) and the demo never imports torch, so BGE-M3 (568M parameters: ~1.1 GB of weights in fp16, 2.3 GB fp32 checkpoint) cannot embed queries there, and the two-tower has no text encoder. Queries are grounded lexically instead:

1. The query goes to Claude Haiku (`claude-haiku-5-5`, configurable) through `anthropic` 1.x with JSON-schema structured output (`output_config.format`): `titles[]`, `keywords[]`, `genres[]`, the genres constrained to the catalogue vocabulary.
2. BM25 over the item documents (§11) returns the top-k grounded items.
3. g = normalise(mean of their item embeddings); the personalised query is q = normalise(λ·u + (1 − λ)·g), with u the as-of user vector and λ tuned on a held-out query set; exact top-k for q (§5), then the ranker. This path is the candidate for C2, the dense query-conditioned arm of ablation (c); its exact definition is fixed on Day 8.

- **Grounding and fallbacks.** Expanded titles that match no catalogue item are dropped, so the LLM never emits a recommendation directly. No API key, an error or a timeout sends the raw query straight to BM25; if nothing grounds, the user vector is used alone. The UI shows which path ran.
- **Reproducibility.** `claude-haiku-5-5` rejects non-default sampling parameters, so determinism comes from a response cache keyed by sha256(model_id + prompt + params) under `artifacts/llm_cache/`; for ablation (c) the cache is the experiment input. The demo serves precomputed expansions for example queries and caps live calls per session.
- **Security.** `ANTHROPIC_API_KEY` comes from the environment or Streamlit secrets and is never committed. The query is untrusted input; the output is schema-constrained and used only as search terms.
- **Cost.** Effort `low` (the model's default is `medium`). Ablation (c) should cost about $1 for ~2,000 queries (estimate: ~300 input and ≤ 1,000 output tokens per query, thinking included, at $0.10 / $0.50 per million tokens, so ~$0.53 per 1K queries); actual cost comes from `response.usage`.
- **Alternative.** `Qwen/Qwen2.5-0.5B-Instruct` served by vLLM on the GPU box (OpenAI-compatible endpoint), reserving 0.2 × VRAM with the flags in §13 (~3 GB on a T4, ~4.5 GB on an A10G), in its own venv or container because vLLM pins exact torch versions. It is not used in CI or the hosted demo, and its weaker knowledge of film titles means more expansions fail to ground.

## 13. Compute plan

| Workload | Where | Memory and time |
|---|---|---|
| `prepare`, EDA, tests, EASE | laptop: Apple M5, 16 GB, no NVIDIA GPU | `prepare` 3-9 s, 2.3 GB peak RSS (measured); EASE minutes; Gram 749 MB fp32; peak ~2.2 GB (G, its inverse and LAPACK's working copy), ~5 GB at 20,720 items (estimate); build XᵀX and invert in float32 |
| Two-tower smoke run, in-batch, B = 4,096 | laptop via MPS | 2.3 GB (K = 4,096; 1.2 GB in-batch) / 88 s (in-batch) per epoch (measured, E002); a 30-epoch smoke run takes ~44 min (E002) |
| B ≥ 8,192 or sampled negatives | cloud GPU: Modal A10G (best value) or a T4 (free on Kaggle/Colab) | table below |
| BGE-M3 item text (optional) | laptop or GPU | ~1.1 GB of weights in fp16 (2.3 GB fp32 checkpoint); embeds the 13,681 kept items in ~1 min, mostly checkpoint load (estimate) |
| LightGBM ranker | CPU | 20K users × 500 ≈ 10M rows ≈ 2 GB RAM (estimate); all 136,674 × 500 = 68M rows need ~9.6–16 GB (estimate), more than the laptop has, so scoring runs in user chunks |
| `Qwen/Qwen2.5-0.5B-Instruct` via vLLM (optional) | GPU box | 0.2 × VRAM with `--gpu-memory-utilization 0.2 --max-model-len 4096 --enforce-eager` (plus `--dtype half` on a T4): ~3 GB on a T4, ~4.5 GB on an A10G (estimate); never at vLLM defaults next to training |

**Two-tower peak VRAM / time per epoch — roofline estimate, ±30%, to be measured on Day 3** (`torch.cuda.max_memory_allocated()`, or the peak of `torch.mps.driver_allocated_memory()` on MPS, logged to W&B; the CUDA context adds a few hundred MB that the allocator does not count). In-batch softmax, plus sampled negatives where a header or cell says so; AMP on the CUDA rows; the M5 row ran bf16 autocast (E002), on synthetic tensors and towers of the planned shapes, with K the number of sampled negatives; Day 3 re-measures it with the real training loop; 136,674 × 128 user and 13,681 × 128 item tables; allocation coefficients measured on the real model shapes.

| GPU | B = 4,096 | B = 8,192 | B = 16,384 + 4,096 sampled negatives (worst case) |
|---|---|---|---|
| T4 16 GB | 1.4 GB / ~26 s | 2.5 GB / ~30 s | 6.1 GB / ~49 s |
| A10G 24 GB | 1.4 GB / ~12 s | 2.5 GB / ~16 s | 6.1 GB / ~26 s |
| RTX 3060 12 GB | 1.4 GB / ~20 s | 2.5 GB / ~27 s | 6.1 GB / ~46 s |
| RTX 4060 8 GB | 1.4 GB / ~25 s | 2.5 GB / ~34 s | 6.1 GB / ~57 s |
| Apple M5 via MPS (**measured** on the owner's laptop, E002) | 2.3 GB (K = 4,096; 1.2 GB in-batch) / 88 s (in-batch) | — | — |

- **Requirement.** Minimum 8 GB of VRAM; 16 GB or more recommended, and the A10G (24 GB) is the best value.
- **Why memory grows.** Autocast keeps softmax and cross-entropy in fp32, so the B × (B + M) logits cost ~13 B per element at the backward peak: 0.22 / 0.87 / 4.36 GB of the three columns. The static part is ~0.40 GB: parameters 78 MB, AdamW state 155 MB, and the index tensors kept on the GPU. The remaining 0.78 / 1.23 / 1.34 GB is the non-logit term (gradients, tower activations including the ≤ 50-item history pooling, workspace), to be measured on Day 3.
- **Evaluation is the easier way to OOM.** A one-shot 136,674 × 13,681 fp32 score matrix is 7.5 GB; evaluation scores chunks of ≤ 8,192 users (~0.45 GB each).
- **Budget (estimate).** By the table a 30-epoch B = 4,096 run is ~6 min on an A10G and ~13 min on a T4. The 3-seed policy (D-005, D-007) makes ablation (a) 15 runs (12 without A3); with smoke/tuning runs, data preparation and BGE-M3 the plan is ~3 GPU-hours on an A10G (~$3.5 at Modal's $1.10/h), budgeted at 4–5 h (~$5) with contingency; a T4 needs roughly twice the time (~$5 at $0.59/h), free on Kaggle/Colab. Every 10 extra B = 4,096 runs add ~1 A10G-hour (~$1.10). The plan covers one protocol; if D-001 adopts the Liang protocol, retraining ablation (a) on that split adds 15 runs ≈ 1.5 A10G-hours (~$1.65) at the rate above.
- **Training tips.** `AdamW(fused=True)`; no weight decay on embedding tables; bf16 autocast if `torch.cuda.is_bf16_supported(including_emulation=False)` (compute capability ≥ 8.0), else fp16 + `GradScaler` (the T4 is sm_75 and has no native bf16); accidental-hit masking of every off-diagonal column that is a train positive of the row's user, never the row's own positive column (~660 per row at B = 4,096, the expected value on the Day-1 train split; ~4 of them duplicate the row's own positive, because the top-100 movies are 23.5% of positives); for MNS, logQ over the mixed proposal, `log(B·p_batch(j) + M·q(j))` with uniform q(j) = 1/N_items.
- **Fresh GPU box.** `scripts/bootstrap_cloud.sh` installs uv, syncs the train, gpu and llm extras (§16), then downloads and prepares the data on the box, since the licence rules out shipping it and `prepare` is deterministic. `split_sha256` in `meta.json` must match the laptop run before training. W&B authenticates with `WANDB_API_KEY`; without a login, use `WANDB_MODE=offline` and `wandb sync` later.

## 14. Limitations

- **Cold start.** Items with fewer than 5 positives are removed and never recommended (D-002). The current split has no cold test items (`cold_test_items` = 0), and under leave-one-out every test user has training history, so offline metrics say nothing about new items or users; new users and visitor profiles rely on the inductive tower (D-004), visitors through its missing-feature path (§7). Neither new items (they would need an item-side OOV row, which the tower does not have) nor visitor profiles are measured.
- **Popularity bias and diversity.** Implicit positives are head-heavy (the top-100 movies are 23.5% of positives). logQ removes the in-batch sampling bias, which raises popular items' scores relative to uncorrected in-batch training (A0); it does not counter popularity bias. A0 vs A1 by train-popularity tercile measures the shift. MNS's uniform negatives constrain tail-item scores, but MNS still targets the full softmax, not a debiased objective. No debiasing, fairness, exposure or diversification objective (MMR, DPP) is optimised, so Top-10 lists can be redundant; MostPop and popularity-stratified HR make the head-heaviness visible, and coverage is a diagnostic only.
- **Offline only.** No exposure logs, position-bias correction, latency SLOs or online validation; offline gains need not transfer online.
- **Timestamp semantics (D-011).** 51.6% of users rated their whole positive history within an hour, so "most recent" is mostly an artefact of a sign-up session, and for users tied at a boundary (24.2%) the hash picks the held-out item. The task is not next-movie prediction (§8.3).
- **Global look-ahead.** Per-user leave-one-out lets the model and global statistics see the future: averaged over test users, 53.7% of train rows post-date the evaluated user's test event (§9.5). Results are reported under `LOO-full` and `LOO-asof`; the tag genome and the global item filter are further accepted mild leaks (§7).
- **Comparability.** No published figure is like-for-like (§9.3); in-repo MostPop and EASE are the only exact references.
- **Dataset age and nature.** Ratings end on 2015-03-31. Timestamps are rating times, not consumption times, and explicit ratings behave differently from modern implicit signals (impressions, dwell time, purchases).
- **Licence (D-003, OPEN).** The GroupLens licence permits research use and forbids redistribution without separate permission, so the repo ships no data and the public demo ships no per-user rating rows (§16). Whether derived artifacts may be published is part of D-003.

## 15. Repository layout vs the code standard

| Code standard | This repo | Note |
|---|---|---|
| `setup.py` | `pyproject.toml` | PEP 621, hatchling build, uv (`uv.lock`); extras in §16 |
| `requirements.txt` | root `requirements.txt` | Dockerfile only; Streamlit Cloud reads `src/shopsage_rec/serving/requirements.txt` (from the Day-2 hello-world deploy) and the root `packages.txt` (§16) |
| `src/data` | `src/shopsage_rec/data/` | `loader`, `preprocess`, `split`, `pipeline` |
| `src/features` | `src/shopsage_rec/features/` | Day 2: the `asof` accessor, feature builders, BM25 documents |
| `src/models` | `src/shopsage_rec/models/` | two-tower (Day 3) |
| `src/evaluation` | `src/shopsage_rec/evaluation/` | metrics and paired bootstrap (Day 2) |
| `src/serving` | `src/shopsage_rec/serving/` | `app.py`, the Streamlit entry point (hello-world deploy on Day 2, demo on Day 9); no Flask/FastAPI, by scope |
| (addition) | `src/shopsage_rec/{retrieval,ranking,llm}/` | the two stages plus the query path: exact search, FAISS study, BM25, hybrid; LightGBM; query expansion |
| `configs/` | `configs/{data,train,deploy}/*.yaml` | |
| `data/` | `data/{raw,processed,samples}/` | raw and processed are git-ignored; `samples/` holds only a README (licence) |
| `docs/`, `notebooks/` | `docs/{design.md,experiment_log.md,api_spec.md,figures/}`, `notebooks/01_eda.ipynb` | |
| `tests/` | `tests/` | pytest + hypothesis, including the golden split test |
| `scripts/` | `download_data.sh`, `prepare_data.sh`, `bootstrap_cloud.sh`; `train.sh` (Day 3), `eval.sh` (Day 4), `serve.sh` (Day 9) | |
| (addition) | `artifacts/` | git-ignored: checkpoints, embeddings, indexes, LightGBM model, LLM cache |
| (addition) | `Dockerfile`, `.streamlit/config.toml`, `.github/workflows/ci.yml`, `Makefile`, `LICENSE` (MIT), `README.md` | CPU demo image, demo config, CI, task runner |

## 16. Packaging and environment

Core: `requires-python = ">=3.11,<3.13"`; `polars>=2.0,<3` (2.0.0 shipped 2026-10-06; the eager API is unchanged), `pyarrow`, `duckdb`, `numpy>=2.0`, `typer` (vendors Click since 0.26, so code never imports `click`), `pydantic`, `pyyaml`. `pyproject.toml` and `uv.lock` are the source of truth (api_spec.md §9).

| Extra | Contents | Used on |
|---|---|---|
| `train` | torch>=2.6, wandb, lightgbm, scikit-learn, scipy, sentence-transformers>=5.0, rank-bm25, tqdm; no FAISS | laptop (MPS), GPU box |
| `cpu` | faiss-cpu>=1.15,<1.16 | laptop or a CPU host |
| `gpu` | faiss-gpu>=1.15,<1.16 on Linux x86_64 only: Meta's official CUDA 12.6 wheel | GPU box |
| `llm` | anthropic>=1.0,<2 (1.x is a breaking major); the Haiku model id is `claude-haiku-5-5` | GPU box, laptop |
| `demo` | streamlit>=1.41, faiss-cpu, lightgbm, anthropic>=1.0,<2, rank-bm25 | Streamlit Cloud, Docker |
| `eda`, `dev` | notebooks and plots; test, lint and type tools (`dev` includes `eda`) | laptop, CI |

- **Dropped:** the unofficial `faiss-gpu-cu12` wheel, and `bitsandbytes`, because the small model (~78 MB of parameters, §13) never needs quantisation.
- **Conflicts.** `[tool.uv]` declares cpu vs gpu and demo vs gpu as conflicting extras: both FAISS wheels install the same top-level `faiss` package.
- **torch on Linux** comes from the explicit index `https://download.pytorch.org/whl/cu126` (torch 2.14.1+cu126), so its CUDA runtime matches faiss-gpu and it runs on drivers ≥ 525 (CUDA 12 minor-version compatibility; ≥ 560 natively), which covers Colab, Kaggle and RunPod T4 hosts; PyPI's default Linux wheel links CUDA 13 and needs driver ≥ 580. Blackwell GPUs (RTX 50xx, B200) need the cu130 index instead, a documented switch. macOS keeps the PyPI wheel (MPS).
- **OpenMP for LightGBM.** macOS: `brew install libomp` before importing lightgbm (the LightGBM 4.7 wheel needs `libomp.dylib`); Debian: `libgomp1` (Dockerfile); Streamlit Cloud: the root `packages.txt` lists `libgomp1`.
- **Commands.** Laptop, Day 1: `uv sync --extra dev`. Laptop, training on MPS: `uv sync --extra dev --extra train --extra cpu`. Cloud GPU box: `scripts/bootstrap_cloud.sh`, which runs `uv sync --python 3.12 --extra train --extra gpu --extra llm`, then download and prepare.

**Streamlit Community Cloud.** 690 MB of RAM guaranteed, up to 2.7 GB; at most 2 CPU cores; Python 3.12 by default; Debian 11; the app sleeps after 12 h idle. It installs from the first dependency file it finds, searching the entrypoint's directory before the repo root, in the order uv.lock > Pipfile > environment.yml > requirements.txt > pyproject.toml, so the committed root `uv.lock` would win over a root `requirements.txt`. **Decided plan (in place from the Day-2 hello-world deploy):** `src/shopsage_rec/serving/requirements.txt`, generated by `uv export --frozen --no-hashes --no-emit-project --extra demo`, plus a two-line `sys.path` shim in `app.py` because the package itself is not installed; the root `requirements.txt` stays for the Dockerfile only. Git-ignored artifacts are fetched at app start from a versioned, sha256-pinned GitHub Release asset. **Licence (D-003, OPEN):** until GroupLens grants permission, the public demo ships no per-user rating rows (visitors build a "pick 5 movies" profile instead) and downloads titles from GroupLens at startup instead of bundling them.

## 17. Schedule (D-012, DECIDED)

- **Day 1:** data pipeline, EDA, leave-one-out split, docs. **Day 2:** as-of accessor, features and leakage canary; metrics and paired bootstrap; MostPop and EASE; BM25 item documents; hello-world Streamlit deploy.
- **Day 3:** two-tower round 1 (MPS smoke run, then cloud GPU), with VRAM measured against §13; D-004. **Day 4:** exact retrieval and the ANN study; D-001, plus the Liang split if adopted (`Liang-SG-strict`; MostPop and EASE also under `Liang-SG-fit`).
- **Day 5:** MNS round, ablation (a). **Day 6:** LightGBM ranker.
- **Day 7:** BM25 retrieval and hybrid, ablation (b). **Day 8:** LLM query expansion, ablation (c). The two days are swapped against the original plan, so BM25 exists before the LLM path is grounded on it.
- **Day 9:** Streamlit demo and deployment (§16). **Day 10:** write-up, README, publish.
- **Cut order if late:** BGE-M3 (genome PCA-64 instead), mined hard negatives, hybrid tuning, the live LLM path, the cloud GPU (train on MPS).

## 18. References

- Burges. *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research MSR-TR-2010-82.
- Chen et al. *BGE M3-Embedding.* arXiv:2402.03216, 2024.
- Cormack, Clarke, Büttcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009.
- Covington, Adams, Sargin. *Deep Neural Networks for YouTube Recommendations.* RecSys 2016.
- Harper, Konstan. *The MovieLens Datasets: History and Context.* ACM TiiS 2015.
- Huang et al. *Embedding-based Retrieval in Facebook Search.* KDD 2020.
- Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 2011.
- Ji, Sun, Zhang, Li. *A Critical Study on Data Leakage in Recommender System Offline Evaluation.* ACM TOIS 2023.
- Johnson, Douze, Jégou. *Billion-scale Similarity Search with GPUs.* arXiv:1702.08734 (IEEE Transactions on Big Data).
- Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NIPS 2017.
- Kim, Suh. *Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms.* RecSys 2019.
- Klenitskiy, Vasilev. *Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?* RecSys 2023. https://arxiv.org/abs/2309.07602
- Krichene, Rendle. *On Sampled Metrics for Item Recommendation.* KDD 2020.
- Liang, Krishnan, Hoffman, Jebara. *Variational Autoencoders for Collaborative Filtering.* WWW 2018. https://arxiv.org/abs/1802.05814
- Lobel, Li, Gao, Carin. *RaCT: Toward Amortized Ranking-Critical Training for Collaborative Filtering.* ICLR 2020.
- Malkov, Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI 2020.
- Mikolov et al. *Distributed Representations of Words and Phrases and their Compositionality.* NIPS 2013.
- Milogradskii et al. *Revisiting BPR: A Replicability Study of a Common Recommender System Baseline.* RecSys 2024.
- Petrov, Macdonald. *A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation.* RecSys 2022. https://arxiv.org/abs/2207.07483
- Rendle, Krichene, Zhang, Koren. *Revisiting the Performance of iALS on Item Recommendation Benchmarks.* RecSys 2022.
- Robertson, Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR, 2009.
- Shenbin, Alekseev, Tutubalina, Malykh, Nikolenko. *RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback.* WSDM 2020.
- Steck. *Embarrassingly Shallow Autoencoders for Sparse Data.* WWW 2019. https://arxiv.org/abs/1905.03375
- Steele, Lea, Flood. *Fast Splittable Pseudorandom Number Generators.* OOPSLA 2014 (SplitMix64).
- Vančura, Kordík. *Deep Variational Autoencoder with Shallow Parallel Path for Top-N Recommendation (VASP).* ICANN 2021.
- Yang et al. *Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations.* WWW 2020 Companion.
- Yi et al. *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations.* RecSys 2019.
- Zhai et al. *Revisiting Neural Retrieval on Accelerators.* KDD 2023.
- Zhai et al. *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.* ICML 2024. https://arxiv.org/abs/2402.17152
