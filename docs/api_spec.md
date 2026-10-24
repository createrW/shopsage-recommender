# Artifact and CLI Specification

> **Re-planned on 2026-10-23 (efficiency first).** The headline protocol is now the Liang split (§2.7), evaluated by `shopsage evaluate` (§6.1). §5.6 (the as-of accessor), §6.2 and §7 (planned commands and artifacts) still describe the Day-1 design and are superseded by [plan.md](plan.md) and [memory.md](memory.md) §6 until the S7 rewrite; where they disagree, those win.

**Status:** v0.4, S1 (2026-10-23) · **Scope:** on-disk artifacts, configuration, Python API, CLI, metric definitions and environment of `shopsage_rec`

There is no HTTP API. Online serving (FastAPI/gRPC) is out of scope and the Streamlit demo reads artifacts directly, so these contracts are the interfaces between pipeline stages. Names below are the Day 1 contract. Where the contract does not fix a detail (exact type annotations, error behaviour), the code in `src/shopsage_rec/` is authoritative and this document is updated to match. Design rationale is in [design.md](design.md); decisions D-001 to D-012 are in [experiment_log.md](experiment_log.md).

## 1. Conventions

- Paths are relative to the repository root. `data/raw/`, `data/processed/` and `artifacts/` are git-ignored, and `data/samples/` holds only a README, because MovieLens may not be redistributed.
- Tables are Parquet. Dtypes use Polars names (`String` is also spelled `Utf8`).
- Timestamps are `Int64` Unix seconds, kept as integers with no timezone conversion.
- There are two ID spaces: raw (`userId`, `movieId`, as in MovieLens) and contiguous (`user_idx`, `item_idx`, for modelling). They are never mixed; conversion goes only through `mappings/`.
- Only `movies.year` is nullable.
- Row order is not part of the contract; consumers sort explicitly.
- `ratings.parquet` and `tags.parquet` are raw. Feature code never reads them directly; it goes through the as-of accessor (§5.6).
- **PLANNED** marks interfaces that do not exist yet. Their names may change on the day they are built, and this file changes with them.

## 2. Processed data: `data/processed/`

Written by `shopsage prepare` (`run_prepare`).

```text
data/processed/
├── ratings.parquet
├── movies.parquet
├── tags.parquet
├── mappings/
│   ├── user_map.parquet
│   └── item_map.parquet
└── splits/
    ├── train.parquet
    ├── val.parquet
    ├── test.parquet
    └── meta.json
```

### 2.1 `ratings.parquet`: all 20,000,263 raw ratings

| Column | Dtype | Meaning |
|---|---|---|
| `userId` | Int32 | raw MovieLens user id |
| `movieId` | Int32 | raw MovieLens movie id |
| `rating` | Float32 | 0.5–5.0 in half-star steps |
| `timestamp` | Int64 | Unix seconds |

Unfiltered: ratings below `min_rating` stay here as explicit negatives for ranker features. **Raw, and unsafe for features as is:** the table contains every val and test positive and, for 36.7% of users, ratings made after their own test event. Features read it only through §5.6.

### 2.2 `movies.parquet`

| Column | Dtype | Meaning |
|---|---|---|
| `movieId` | Int32 | raw id |
| `title` | String | raw title as in `movies.csv` |
| `title_clean` | String | title without surrounding whitespace and without its trailing year group (rules below) |
| `year` | Int32, nullable | first year of the trailing year group; null when there is none |
| `genres` | List[String] | genre names; empty list for "(no genres listed)" |

Year rules, matched at the end of the whitespace-stripped title:

| Trailing group | Example `title` | `year` | `title_clean` |
|---|---|---|---|
| `(YYYY)` | `Toy Story (1995)` | 1995 | `Toy Story` |
| `(YYYY)` after other parentheses, which are kept | `Blade Runner (Director's Cut) (1982)` | 1982 | `Blade Runner (Director's Cut)` |
| open range, hyphen or en dash, optional spaces | `Big Bang Theory, The (2007-)`; `Diplomatic Immunity (2009– )` | 2007; 2009 | `Big Bang Theory, The`; `Diplomatic Immunity` |
| closed range: the first year | `Fawlty Towers (1975-1979)` | 1975 | `Fawlty Towers` |
| stray extra `)` | `Mona and the Time of Burning Love (Mona ja palavan rakkauden aika) (1983))` | 1983 | `Mona and the Time of Burning Love (Mona ja palavan rakkauden aika)` |
| none | `Babylon 5` | null | `Babylon 5` |

Covers the 27,278 movies of `movies.csv`. The modelled catalogue is the subset listed in `mappings/item_map.parquet`.

### 2.3 `tags.parquet`

| Column | Dtype | Meaning |
|---|---|---|
| `userId` | Int32 | raw id of the tagging user |
| `movieId` | Int32 | raw movie id |
| `tag` | String | free-text tag |
| `timestamp` | Int64 | Unix seconds |

All 465,564 tag applications. **Raw, and unsafe for features as is:** tags may post-date a user's cutoff, and 711 users tagged their own test movie. Features read it only through §5.6; BM25 item documents use train-period tags and never the evaluated user's own (D-010).

### 2.4 `mappings/`

| File | Column | Dtype | Meaning |
|---|---|---|---|
| `user_map.parquet` | `user_idx` | Int32 | 0..U−1, assigned in ascending `userId` order |
| `user_map.parquet` | `userId` | Int32 | raw id |
| `item_map.parquet` | `item_idx` | Int32 | 0..I−1, assigned in ascending `movieId` order |
| `item_map.parquet` | `movieId` | Int32 | raw id |

The maps cover exactly the users and items that survive filtering (the union of the three splits); U = `n_users` and I = `n_items` in `meta.json`. **Alignment rule** for every downstream artifact: row k of a per-user (per-item) array belongs to `user_idx` (`item_idx`) k.

### 2.5 `splits/train.parquet`, `splits/val.parquet`, `splits/test.parquet`

| Column | Dtype | Meaning |
|---|---|---|
| `user_idx` | Int32 | contiguous user index |
| `item_idx` | Int32 | contiguous item index |
| `rating` | Float32 | original rating, ≥ `min_rating` |
| `timestamp` | Int64 | Unix seconds |

| Split | Rows per user | Content |
|---|---|---|
| `test` | exactly 1 | the positive with the largest (`timestamp`, `tie_key`) |
| `val` | exactly 1 | the second largest |
| `train` | ≥ 3 | all remaining positives |

`tie_key` is a UInt64 splitmix64 hash of the raw (`userId`, `movieId`) and `split.seed` (§5.4). It only decides same-second ties. `prepare` computes it in memory and does not persist it, so the split files hold exactly the four columns above.

### 2.6 Invariants

Consumers may rely on these (Day-1 leave-one-out split):

1. `ratings.parquet`, `movies.parquet` and `tags.parquet` have 20,000,263, 27,278 and 465,564 rows.
2. Every split row has `rating ≥ min_rating`.
3. `n_val = n_test = n_users`, and every user has at least `min_user_interactions − 2` train rows.
4. Per user: max train `timestamp` ≤ val `timestamp` ≤ test `timestamp`. Equality occurs only in same-second ties, which `tie_key` decides; `users_with_val_train_tie` and `users_with_boundary_tie` count them (§3).
5. No (`user_idx`, `item_idx`) pair appears in more than one split, and the three splits together are exactly the filtered positives: `n_train + n_val + n_test = n_interactions`.
6. `user_idx` ∈ [0, U) and `item_idx` ∈ [0, I), both contiguous.
7. Every val/test `item_idx` is in `item_map` but may have no train rows; `cold_val_items` and `cold_test_items` count such items.
8. The split files carry exactly the four §2.5 columns and dtypes; `tie_key` is never written.
9. Re-running with the same config reproduces identical split contents and the same `split_sha256`, independently of the Polars version: the hash is computed in numpy, never with `Expr.hash`. A different `split.seed` can change split membership only for users counted in `users_with_any_boundary_tie`, and every count except `cold_val_items` and `cold_test_items` stays the same.

### 2.7 The Liang split: `data/processed/liang/`

Written by `shopsage prepare` with `configs/data/movielens_20m_liang.yaml` (`split.strategy: liang`; `data/liang.py`), the default config since S1. Same layout as §2, with five split files:

| File | Rows (ML-20M) | Content |
|---|---|---|
| `splits/train.parquet` | 8,538,846 | every positive of the 116,677 training users |
| `splits/val_tr.parquet`, `splits/val_te.parquet` | 589,887 / 142,514 | validation users' fold-in items (model input) and targets |
| `splits/test_tr.parquet`, `splits/test_te.parquet` | 579,008 / 139,775 | test users' fold-in items and targets |
| `mappings/user_map.parquet` | 136,677 | `user_idx` (Int32, ascending `userId`), `userId`, `role` (`train`, `val` or `test`) |
| `mappings/item_map.parquet` | 20,108 | `item_idx` (Int32, ascending `movieId`) over the training catalogue |

Split files carry `user_idx`, `item_idx`, `rating` and `timestamp`, sorted by `user_idx` then `item_idx`. `meta.json` holds `strategy` (`liang`), `seed`, `n_heldout_users`, `test_prop`, the user, item and row counts, `val_users_with_targets` / `test_users_with_targets` (and `..._without_targets`), and `split_sha256`, a sha256 over the five files' sorted (`userId`, `movieId`) pairs, plus the provenance keys of §3. Invariants: the three user groups are disjoint; held-out rows only use training-catalogue items; for a held-out user with n ≥ 5 catalogue items, exactly `int(test_prop × n)` are targets; the same seed reproduces `split_sha256`. `load_liang(processed_dir)` returns the split as binary CSR matrices over `item_idx` (rows: training users; validation and test users with at least one target).

## 3. `splits/meta.json`

Written by `write_splits`: the split statistics of `leave_one_out_split` (`SplitResult.stats`, "split" below) merged with the run metadata that `run_prepare` passes as `extra=` ("prepare" below). The key set is the code's (`src/shopsage_rec/data/split.py`, `src/shopsage_rec/data/pipeline.py`); if the two ever disagree, the code is canonical and this table is corrected.

| Key | Type | From | Meaning |
|---|---|---|---|
| `strategy` | string | split | `"leave_one_out"` |
| `seed` | int | split | `split.seed`, the tie-break seed |
| `tie_break` | string | split | human-readable description of the tie-break hash, its seed and the key source (example below) |
| `n_users` | int | split | users after filtering (U) |
| `n_items` | int | split | items after filtering (I) |
| `n_interactions` | int | split | filtered positives, = `n_train + n_val + n_test` |
| `n_train` | int | split | rows in `train.parquet` |
| `n_val` | int | split | rows in `val.parquet` (= `n_users`) |
| `n_test` | int | split | rows in `test.parquet` (= `n_users`) |
| `density` | float | split | `n_interactions / (n_users × n_items)` |
| `users_with_boundary_tie` | int | split | users whose test and val events share a timestamp, so `tie_key` decided which is test |
| `boundary_tie_fraction` | float | split | `users_with_boundary_tie / n_users` |
| `users_with_val_train_tie` | int | split | users whose val timestamp equals their newest train timestamp, so `tie_key` decided which positive is val |
| `users_with_any_boundary_tie` | int | split | users with either tie: the only users whose split another seed can change |
| `any_boundary_tie_fraction` | float | split | `users_with_any_boundary_tie / n_users` |
| `cold_test_items` | int | split | distinct test `item_idx` values with no row in `train.parquet` |
| `cold_val_items` | int | split | distinct val `item_idx` values with no row in `train.parquet` |
| `split_sha256` | string | split | 64 hex characters: SHA-256 fingerprint of the held-out sets (below) |
| `dataset` | string | prepare | dataset identifier from the config |
| `config` | object | prepare | snapshot of the resolved `DataConfig`, including `split`; `raw_dir` and `processed_dir` are absolute |
| `n_ratings_raw` | int | prepare | rows in `ratings.parquet` |
| `n_movies` | int | prepare | rows in `movies.parquet` |
| `n_tags` | int | prepare | rows in `tags.parquet` |
| `created_at` | string | prepare | ISO 8601 UTC timestamp of the run, to the second |
| `versions` | object | prepare | versions of `shopsage_rec`, `polars` and `python` |

**`tie_break`** as `prepare` writes it: `splitmix64(((user << 32) | item) xor splitmix64(seed=42)); key from precomputed column 'tie_key'`. Called without a precomputed key column, `leave_one_out_split` ends the string with `key from (user_idx, item_idx)` instead.

**`split_sha256`** is the SHA-256 over, in this order: the ASCII bytes `val`; the val (`user_idx`, `item_idx`) pairs sorted by (`user_idx`, `item_idx`), as a row-major int64 array in native byte order (little-endian on every supported platform); the ASCII bytes `test`; the test pairs in the same layout. Train is the complement, so for a given set of positives the digest identifies the whole split. Downstream manifests record it (§7) so that consumers can refuse artifacts built on another split. experiment_log.md (E001) has a snippet that recomputes it from the parquet files.

**Relations** consumers may assert: `n_val = n_test = n_users`; `users_with_boundary_tie ≤ users_with_any_boundary_tie ≤ n_users`; `users_with_val_train_tie ≤ users_with_any_boundary_tie ≤ users_with_boundary_tie + users_with_val_train_tie`.

Shape only; the placeholders are not values:

```json
{
  "strategy": "leave_one_out",
  "seed": "<int>",
  "tie_break": "<string>",
  "n_users": "<int>",
  "n_items": "<int>",
  "n_interactions": "<int>",
  "n_train": "<int>",
  "n_val": "<int>",
  "n_test": "<int>",
  "density": "<float>",
  "users_with_boundary_tie": "<int>",
  "boundary_tie_fraction": "<float in [0, 1]>",
  "users_with_val_train_tie": "<int>",
  "users_with_any_boundary_tie": "<int>",
  "any_boundary_tie_fraction": "<float in [0, 1]>",
  "cold_test_items": "<int>",
  "cold_val_items": "<int>",
  "split_sha256": "<64 hex characters>",
  "dataset": "<string>",
  "config": {"<DataConfig field>": "<value>", "split": {"strategy": "leave_one_out", "seed": "<int>"}},
  "n_ratings_raw": "<int>",
  "n_movies": "<int>",
  "n_tags": "<int>",
  "created_at": "<ISO 8601, UTC>",
  "versions": {"shopsage_rec": "<version>", "polars": "<version>", "python": "<version>"}
}
```

Consumers ignore keys they do not use. A key the writer adds is recorded here and in the change log (§10).

## 4. Configuration: `configs/data/*.yaml`

Loaded by `shopsage_rec.config.load_data_config(path) -> DataConfig`. `configs/data/movielens_20m.yaml` is the source of truth for values; the table records the Day 1 decisions.

| Key | Type | `movielens_20m` | Meaning |
|---|---|---|---|
| `dataset` | str | `movielens-20m` | identifier copied into `meta.json` |
| `raw_dir` | path | `data/raw/ml-20m` | directory holding the unzipped CSVs |
| `processed_dir` | path | `data/processed` | output root for §2 |
| `min_rating` | float in [0.5, 5.0] | 4.0 | positive threshold, inclusive |
| `min_item_interactions` | int ≥ 1 | 5 | first pass: drop items with fewer positives (D-002) |
| `min_user_interactions` | int ≥ 3 | 5 | second pass: drop users with fewer positives |
| `split.strategy` | `"leave_one_out"` | `leave_one_out` | the only implemented value |
| `split.seed` | int in [0, 2^64) | 42 | seed mixed into `tie_key` (§5.4) |

Bounds are enforced at load time and unknown keys are rejected. Structure (values must track the real file):

```yaml
dataset: movielens-20m
raw_dir: data/raw/ml-20m
processed_dir: data/processed
min_rating: 4.0
min_item_interactions: 5
min_user_interactions: 5
split:
  strategy: leave_one_out
  seed: 42
```

The Liang protocol (`Liang-SG-strict` and `Liang-SG-fit`, §8; D-001, D-002) will add a strong-generalisation `split.strategy` value and a config with `min_item_interactions: 1`, which reproduces Liang et al.'s preprocessing: no item filter, 136,677 users, 20,720 items and 9,990,682 positives (their user split then leaves 20,108 items). On positives, 1 is a no-op, equivalent to their threshold 0; the validator rejects 0. Neither exists today. Because `tie_key` hashes raw ids, a different filter does not change how the surviving ties are broken.

**Training configuration (S2).** The Two-Tower has no YAML file: its settings are `TrainConfig` (`models/trainer.py`) and the arms A0–A4 (`ARMS`), set from `shopsage train` options and snapshotted into each run's `metrics.json`. The ranker has none either: `DEFAULT_PARAMS` and `train_rankers` arguments (`ranking/train.py`), snapshotted in its `summary.json`. **PLANNED config families.** A draft exists (`configs/deploy/streamlit.yaml`), but nothing loads or validates it yet. Schemas are fixed on the day each lands; contents are indicative.

| Directory | Day | Expected contents |
|---|---|---|
| `configs/train/` | 3, 5 (two-tower); 6 (ranker) | two-tower: user-tower inputs (D-004: history length, behavioural features, `user_id` embedding on/off, ID-dropout rate), item-tower inputs (genres, year bucket, genome PCA-64 with a missing-genome flag, optional BGE-M3), embedding dim, batch size, τ, negative mode (in-batch, in-batch + logQ, MNS + logQ, mined, full softmax; D-005), M, α, optimiser, a fixed 30-epoch budget with `checkpoint_metric: va_hr@50` (the checkpoint with the best V_A HR@50 is kept; no early stopping), AMP dtype, seeds, W&B project; ranker: candidate source and depth, `candidates_per_group: all_500`, feature list, LightGBM parameters (D-006); every train config: `val_partition: {seed, frac_va}`, the V_A / V_B user split, fixed across training seeds |
| `configs/deploy/` | 9 | `bundle: {release_tag, sha256}` (the demo bundle's GitHub Release tag and pinned sha256); the bundle's file list (§7), with `bm25_matrix.npz` (subject to D-003) and `llm_cache/` and without `user_map`; `llm_expansion` with the model id `claude-haiku-5-5`, `prompt_version`, `cache_dir`, `timeout_s` and `max_calls_per_session`; memory caps |

## 5. Python API (Day 1)

Names and parameters are the Day 1 contract; annotations the contract does not fix are shown as expected.

### 5.1 `shopsage_rec.config`

```python
SplitConfig(strategy: Literal["leave_one_out"], seed: int)
DataConfig(dataset, raw_dir, processed_dir, min_rating, min_item_interactions,
           min_user_interactions, split: SplitConfig)
load_data_config(path, root=None) -> DataConfig
```

Relative `raw_dir` and `processed_dir` are anchored at `root=` when it is given, resolved to an absolute path; otherwise at the repository root, the nearest ancestor of the config file that holds `pyproject.toml`, falling back to the current working directory. Both paths are therefore absolute, as `meta.json` records them (§3).

### 5.2 `shopsage_rec.data.loader`

| Function | Returns |
|---|---|
| `load_ratings(raw_dir)` | `pl.DataFrame`: `userId` Int32, `movieId` Int32, `rating` Float32, `timestamp` Int64 |
| `load_movies(raw_dir)` | `pl.DataFrame`: `movieId` Int32, `title` String, `genres` String (raw, pipe-separated); `parse_movies(load_movies(raw_dir))` has the §2.2 schema |
| `load_tags(raw_dir)` | `pl.DataFrame`: `userId` Int32, `movieId` Int32, `tag` String, `timestamp` Int64 |

### 5.3 `shopsage_rec.data.preprocess`

| Function | Contract |
|---|---|
| `parse_movies(movies)` | keeps `movieId` and the raw `title`; derives `title_clean` and `year` by the §2.2 rules; turns `genres` into List[String], empty for the sentinel |
| `filter_interactions(ratings, *, min_rating, min_item_interactions, min_user_interactions)` | keeps `rating ≥ min_rating`, then drops items below `min_item_interactions`, then users below `min_user_interactions`; one pass, in that order; returns the four ratings columns; `ValueError` if a threshold is below 1 |
| `build_id_maps(interactions) -> (user_map, item_map)` | contiguous 0-based Int32 indices in ascending raw-id order (§2.4) |
| `remap_ids(interactions, user_map, item_map)` | replaces `userId` / `movieId` with `user_idx` / `item_idx`; the output columns are `user_idx`, `item_idx`, then every other input column unchanged and in input order (`rating`, `timestamp` and extras such as `tie_key`); `ValueError` if an id map has duplicate keys or any row fails to map |

### 5.4 `shopsage_rec.data.split`

| Name | Contract |
|---|---|
| `SplitResult(train, val, test, stats)` | three `pl.DataFrame`s with the input's columns minus the tie-key column (the §2.5 schema in the pipeline); `stats` holds the "split" keys of §3 |
| `leave_one_out_split(interactions, *, seed=42, user_col="user_idx", item_col="item_idx", time_col="timestamp", tie_key_col="tie_key") -> SplitResult` | semantics below |
| `compute_tie_key(frame, user_col, item_col, seed) -> pl.Series` | UInt64 Series named `tie_key`: `splitmix64(((user << 32) \| item) xor splitmix64(seed))` per row; `ValueError` if a user or item id is null or lies outside [0, 2³²), or if `seed < 0` |
| `splitmix64(x) -> np.ndarray` | vectorised SplitMix64 output function (Steele, Lea & Flood 2014) over uint64, with wrapping 64-bit numpy arithmetic |
| `write_splits(result, out_dir, extra=None) -> Path` | writes `train.parquet`, `val.parquet`, `test.parquet` (zstd) and `meta.json` = `result.stats` merged with `extra` |

`leave_one_out_split`:

1. **Validation.** `ValueError` if a required column is missing, `seed < 0`, a user id, item id or timestamp is null, a (user, item) pair repeats, or a user has fewer than 3 rows.
2. **Tie key.** When `tie_key_col` is a column of `interactions`, it is used verbatim (cast to UInt64); `run_prepare` fills it from the raw (`userId`, `movieId`) before remapping. Otherwise `compute_tie_key` runs over (`user_col`, `item_col`) with `seed`, which gives a different but equally valid order. With a precomputed column, `seed` only labels `stats` (`seed`, `tie_break`), so pass the seed the key was built with.
3. **Order.** Each user's rows are sorted by (`time_col`, tie key), both descending: position 0 is test, 1 is val, the rest train. The key is injective in (user, item), since splitmix64 is a bijection on 64-bit integers, so no tie survives.
4. **Output.** Every input column except `tie_key_col`, in input order.
5. **Why this hash.** Computed in numpy, it does not depend on the Polars version (`Expr.hash` is not stable across releases and is never used). It is non-linear, so it favours neither the higher nor the lower item id; the affine-mod-prime hash it replaces favoured the higher of two adjacent item ids ~81% of the time. Keyed on raw ids, its tie decisions survive changes to the filters and therefore to the contiguous maps. A golden test in `tests/test_split.py` pins the `split_sha256` of a fixed 20-row table, that is, its exact val/test pairs.

### 5.5 `shopsage_rec.data.pipeline`

| Name | Contract |
|---|---|
| `run_prepare(cfg) -> PrepareReport` | load → parse → write the raw tables → filter → `compute_tie_key` over the raw ids with `split.seed` → build maps → `remap_ids` (carries `tie_key`) → `leave_one_out_split` → `write_splits`; `PrepareReport` summarises the run (counts, `users_with_boundary_tie`, `users_with_any_boundary_tie`, `cold_test_items`, `split_sha256`, elapsed seconds), with values mirrored in `meta.json` |

```python
from shopsage_rec.config import load_data_config
from shopsage_rec.data.pipeline import run_prepare

report = run_prepare(load_data_config("configs/data/movielens_20m.yaml"))  # from the repo root: same as `shopsage prepare`
```

### 5.6 `shopsage_rec.features.asof` (PLANNED, Day 2)

The single as-of accessor, and the only allowed way to build features (D-012). Nothing in `features/` exists yet; parameters other than `stage` are fixed on Day 2.

```python
history(stage: Literal["val", "test"], ...)
```

| Rule | `stage="val"` | `stage="test"` |
|---|---|---|
| Cutoff, per user | timestamp of the val event | timestamp of the test event |
| History positives | split membership: train | split membership: train + val |
| Raw ratings and tags | rows with `timestamp` < cutoff | rows with `timestamp` < cutoff |
| Item statistics (`item_popularity_train`, `item_mean_rating_train`, ...) | train split only | train split only |
| Candidate mask | the history positives and every item the user rated before the cutoff | the same, at the test cutoff |

- History comes from split membership, not timestamps: a train positive that shares the cutoff second (a val/train tie) is history, while raw ratings and tags must be strictly earlier.
- `item_popularity_train` (log count) and `item_mean_rating_train` (mean rating) are computed over the train split at both stages, so the ranker gets the same statistic in training (val stage) and at test. The `_train` suffix says what they are: train-split statistics, not time-filtered ones, since the train split holds other users' later events. "As-of popularity" (positives with timestamp strictly before the user's cutoff; design.md §10) is a different statistic, used by MostPop only.
- **Missing values.** When a user has no raw rating strictly before the cutoff (61 users at the val stage and 51 at the test stage, whose history rows all share the cutoff second; and every visitor profile), `mean_rating` is the mean over the history positives' ratings if any is known, else the training-set median, with a `mean_rating_missing` flag, and `recency_days` goes to an "unknown / same-session" bucket. The user tower trains with feature dropout p = 0.3 on the behavioural block (mirroring ID-dropout), so this path is learned (design.md §7).
- A leakage-canary test plants events at and after each user's cutoff and asserts that no feature changes. Its fixtures include the 61 val-stage and 51 test-stage users of the missing-value rule.
- The tag genome and the global item filter are accepted mild global leaks, documented rather than removed (experiment_log.md, Protocols).

## 6. CLI: `shopsage`

Typer application, installed as the `shopsage` entry point by `uv sync`. Invoke it as `uv run shopsage …`, or as plain `shopsage` with `.venv` activated. Global options: `--verbose` / `-v` (debug logging) and `--version`, an eager option: `shopsage --version` prints the version and exits 0.

### 6.1 Available (S1)

| Command | Reads | Writes |
|---|---|---|
| `shopsage prepare --config configs/data/movielens_20m_liang.yaml` | CSVs in `raw_dir` | the Liang split of §2.7, including `splits/meta.json` (wraps `run_prepare`); prints the `LiangReport` summary. With `--config configs/data/movielens_20m.yaml` it writes the Day-1 split of §2 and prints a `PrepareReport` |
| `shopsage stats --processed-dir data/processed/liang` | `<processed-dir>/splits/meta.json` | that file, pretty-printed to stdout |
| `shopsage train --arm A1 --seed 42 [--epochs 30 --tau 0.05 --lr 0.002 --batch-size 4096 --user-frac 1 --device auto --mined-from RUN --run-id ID --no-wandb]` | the Liang split, `item_features.npz` (built from `genome-scores.csv` on first use) | `artifacts/runs/<run_id>/`: `item_embeddings.npy` (float32 [20,108, 128], unit norm), `user_tower.npz` (`hist_emb`, `w1`, `b1`, `w2`, `b2`: the numpy user tower), `model.pt` (best V_A epoch), `per_user.parquet` and `metrics.json` (config, learning curve, V_A Recall@50, test metrics with CIs, peak memory) |
| `shopsage mine --run-id A1-s42` | that run's exports | `artifacts/runs/<run_id>/mined_101_500.npy` (int32 [116,677, 400]: ranks 101–500 per training user, training positives excluded) |
| `shopsage sweep [--epochs 30 --tune-epochs 10 --no-wandb]` | the Liang split | every run of the S2 sweep, one after another, and `artifacts/runs/sweep_summary.json`; `scripts/modal_train.py` runs the same plan on Modal in parallel |
| `shopsage recall [--dense-runs A1-s42,A1-s43,A1-s44 --depth 200 --parity-users 1000 --run-id ID]` | the Liang split, the dense runs' exports, `genome-tags.csv`, `genome-scores.csv`, training users' tags | `artifacts/eval/<run_id>/`: `per_user.parquet` (`bm25/…`, `dense:<run>/…`, `hybrid:<run>/…`) and `summary.json` (BM25 grid on V_A, arms with CIs, paired deltas, candidate coverage, Milvus Lite parity and latency) |
| `shopsage rank [--dense-run A1-s42 --seeds 42,43,44 --latency-users 500 --run-id ID]` | the Liang split, the dense run's exports, the BM25 documents, a dense EASE fit (λ = 1,000) pruned to 100 neighbours per item | `artifacts/rank/<run_id>/`: `models/ranker-s<seed>.txt` and `models/ranker-no-ease-s<seed>.txt` (LightGBM text models), `per_user.parquet` (`ease/…`, `dense/…`, `ranker:s<seed>/…`, seed means), `summary.json` (settings, training stats, arms with seed sd and CIs, paired deltas, feature importance, serving latency) |
| `shopsage tags [--n-events 1000 --workers 8 --seed 0 --exclude-run RUN --run-id ID]` (needs the `llm` extra and `ANTHROPIC_API_KEY`; without a key every query falls back to the raw text) | held-out users' tags, the BM25 documents (E008 settings), `claude-haiku-5-5` | `artifacts/llm/<run_id>/`: `per_event.parquet` (C0 / C1 / C1r metrics, expansion source, grounding), `expansions.json`, `summary.json` (arms, paired deltas, subgroups, calls, tokens, cost, latency); `artifacts/llm/cache/<sha256>.json` (one response per request) |
| `shopsage evaluate --processed-dir data/processed/liang --out-dir artifacts/eval` | the Liang split (`load_liang`) | `<out-dir>/<run-id>/per_user.parquet` (one row per test user with targets, one `model/metric` column each) and `summary.json` (test metrics with bootstrap CIs, EASE's λ selection on V_A, the paired EASE − MostPop differences, timings, peak RSS); options `--models mostpop,ease`, `--ease-lambdas 250,500,1000`, `--run-id` (default `local-YYYYMMDD-HHMM`) |

The flag values shown are also the defaults (since S1 they follow the Liang split), resolved against the repository root found from the current working directory (the nearest directory at or above it that holds `pyproject.toml`, else the current working directory), so both commands work from any subdirectory of the repository. `prepare` also takes `--root DIR`, which must be an existing directory; it is resolved to an absolute path and anchors the config's relative paths (§5.1). When a raw CSV is missing, `prepare` exits 1 with a one-line `Error: ...` message (the loader's hint to run `scripts/download_data.sh`) instead of a traceback, and it creates no output directory before the raw files are loaded.

### 6.2 PLANNED

Flags are fixed on the day each command lands.

| Command | Day | Purpose | Reads | Writes |
|---|---|---|---|---|
| `shopsage build-index` | 4 | FAISS IVF and HNSW indexes for the ANN study; exact search needs none (D-008) | `item_embeddings.npy` | `artifacts/index/` |
| `shopsage rank` | 6 | train and apply the LightGBM ranker (D-006) | candidates, as-of features | `artifacts/ranker/` |
| `shopsage expand` | 8 (swapped with BM25, D-012) | LLM query expansion through the response cache (D-010) | queries, `ANTHROPIC_API_KEY` | `artifacts/llm_cache/` |
| `shopsage demo` | 9 | run the Streamlit app (`src/shopsage_rec/serving/app.py`) locally | the demo bundle (§7) | — |

### 6.3 Scripts

| Script | Day | Purpose |
|---|---|---|
| `scripts/download_data.sh` | 1 | fetch, md5-verify and unpack ML-20M into `data/raw/` (archive md5 `cd245b17a1ae2cc31bb14903e1204af3`) |
| `scripts/prepare_data.sh` | 1 | `uv run shopsage prepare --config CONFIG` (default `configs/data/movielens_20m_liang.yaml` since S1); downloads first if the raw CSVs are missing |
| `scripts/bootstrap_cloud.sh` | 1 (S1) | fresh Linux GPU box: install uv → `uv sync --python 3.12 --extra train --extra llm` → torch and LightGBM sanity check → download → the Liang split; prints the VRAM guidance and the next command (`train.sh` lands in S2). `--skip-data` stops after the environment |
| `scripts/bench_two_tower_step.py` | 1 | one Two-Tower training step on synthetic tensors of the ML-20M shapes: ms/step, s/epoch, peak memory (E002 in experiment_log.md) |
| `scripts/train.sh` | 3 | training entry point on the GPU box |
| `scripts/eval.sh` | 4 | evaluation entry point |
| `scripts/serve.sh` | 9 | start the Streamlit demo (`src/shopsage_rec/serving/app.py`) |

## 7. Artifacts: `artifacts/` (PLANNED)

Nothing here exists on Day 1. Everything is keyed by run id, so indexes, rankers and the demo can check which model they belong to. Configs also reference a run by its run id, never by a run name: one run name covers all seeds of an arm.

```text
artifacts/
├── checkpoints/<run_id>/            Day 3     model and optimiser state, resolved config
├── embeddings/<run_id>/             Day 3
│   ├── item_embeddings.npy          float32 [n_items, 128], row k = item_idx k, L2-normalised
│   ├── user_embeddings_val.npy      float32 [n_users, 128], history as of each val cutoff
│   ├── user_embeddings_test.npy     float32 [n_users, 128], history as of each test cutoff
│   ├── user_tower.npz               user-tower weights for the numpy forward pass (demo)
│   └── manifest.json
├── text/item_text_embeddings.npy    Day 2-3   optional BGE-M3 dense vectors, float16 [n_items, 1024]
├── index/<run_id>/                  Day 4     FAISS IVF and HNSW indexes (ANN study) + manifest.json
├── ranker/<run_id>/                 Day 6     LightGBM model (text format), feature list, manifest.json
├── bm25/                            Day 2, 7  item documents (Day 2); scipy.sparse BM25 matrix (Day 7)
├── llm_cache/                       Day 8     cached expansions (JSONL), keyed by sha256(model_id + prompt + params)
├── eval/<run_id>/                   Day 2+    per-user metrics (parquet), summary with CIs (JSON)
└── demo/                            Day 9     the demo bundle, published as a versioned GitHub Release asset
```

| Artifact | Producer | Consumers | Contract |
|---|---|---|---|
| `item_embeddings.npy` | `shopsage train` | `evaluate`, `build-index`, ranker features, demo | C-contiguous float32 `[n_items, 128]` (13,681 × 128 ≈ 7 MB with the Day-1 config); row = `item_idx`; unit L2 norm, asserted by a test at export (D-008) |
| `user_embeddings_{val,test}.npy` | `shopsage train` | `evaluate`, ranker features | float32 `[n_users, 128]`; row = `user_idx`; one file per stage, because each vector is computed from history as of that stage's cutoff (§5.6); never shipped to the demo (D-003, D-004) |
| `user_tower.npz` | `shopsage train` | demo | MLP weights for the numpy forward pass over pooled item embeddings plus behavioural features (missing values per §5.6); numpy and torch agree on 1,000 users to a max abs diff < 1e-5 (D-004), tested with `torch.testing.assert_close(..., equal_nan=False)` and an explicit no-NaN assertion |
| `item_text_embeddings.npy` | BGE-M3 job | item tower | float16 `[n_items, 1024]`, row = `item_idx`, BGE-M3 over "title (year). genres. top genome tags"; first on the cut list, with genome PCA-64 as the fallback (D-012) |
| FAISS index | `shopsage build-index` | ANN study | ids = `item_idx`; inner-product metric; built from the same run's `item_embeddings.npy`. Exact search is the default retrieval and needs no stored index (D-008): batched matmul → `masked_fill(M_u, −inf)` → `torch.topk` / `np.argpartition`, in chunks of ≤ 8,192 users. Where IndexFlatIP is used, it runs one `search()` per user with `IDSelectorNot(IDSelectorBatch(M_u))`, or k = 500 + max_u \|M_u\| per chunk followed by dropping the masked ids, which is equivalent to masking first (§8.1) |
| BM25 matrix | Day 7 | retrieval, ranker features, demo | scipy.sparse, rows = `item_idx`, over the D-010 item documents |
| LightGBM model | `shopsage rank` | `evaluate`, demo | text model file plus an ordered feature list; CPU inference only |
| `manifest.json` | each producer | each consumer | run id, config hash, `created_at` and `split_sha256` of the data's `meta.json`, array shapes and dtypes, normalisation flag, package versions; consumers refuse a mismatch |

**Demo** (Day 9, PLANNED; entry point `src/shopsage_rec/serving/app.py`). It reads only the demo bundle: item embeddings, user-tower weights, the ranker, the BM25 matrix (`bm25_matrix.npz`; its vocabulary comes from MovieLens titles and tags, so whether it may ship is part of D-003) and precomputed LLM expansions for its example queries (`llm_cache/`). It never imports torch. A visitor's "pick 5 movies" profile goes through the numpy user tower and the missing-value rule of §5.6 (its quality is not measured), so no user table (`user_map` is not in the bundle) and no per-user rating rows are shipped, and movie titles are downloaded from GroupLens at startup rather than bundled (D-003). The bundle is fetched at startup from a versioned GitHub Release asset and verified against a pinned sha256 (`bundle: {release_tag, sha256}` in `configs/deploy/streamlit.yaml`, §4); live LLM calls are capped per session (`llm_expansion.max_calls_per_session`). Memory target: the 690 MB that Streamlit Community Cloud guarantees (§9.5).

## 8. Metric definitions (PLANNED)

Implemented on Day 2 with the MostPop and EASE anchors (D-012) and frozen with the protocol decision D-001 on Day 4. The evaluation code becomes authoritative when it lands, and this section is updated to match. Protocol labels (`LOO-full`, `LOO-asof`, `Liang-SG-strict`, `Liang-SG-fit`) are defined in experiment_log.md; `Liang-SG-*` below means both Liang variants. In `Liang-SG-strict` the held-out users are never trained on (Liang et al. 2018); in `Liang-SG-fit` their fold-in 80% is part of the training data (Milogradskii et al., RecSys 2024).

### 8.1 Candidates and ranks

For user u at a stage (val or test), R_u is the set of held-out relevant items and n_u = #R_u: under `LOO-*` the single val or test item, under `Liang-SG-*` the held-out 20% of the user's positives. M_u is the masked set. Under `LOO-*` it holds the history positives and every item the user rated before the cutoff (§5.6), and under `LOO-asof` also the items first rated after the cutoff; under `Liang-SG-*` it holds the user's fold-in positives. Every catalogue item outside M_u is scored (full-corpus ranking, no sampled candidates; Krichene & Rendle 2020). Masked items get score −∞ before the top-k (D-008). Dropping masked ids from a plain top-500 afterwards is not allowed, because it leaves fewer than 500 items; the one allowed variant over-fetches k = 500 + max_u |M_u| per chunk of users and then drops the masked ids, which returns the same top-500 as masking first (§7, FAISS index). rank_u(i) is the 1-based position of item i in descending score order, ties broken by ascending `item_idx`.

### 8.2 Per-user metrics

```text
hits_u(K)       = #{ i in R_u : rank_u(i) <= K }
HR@K            = 1 if hits_u(K) >= 1, else 0
Recall@K        = hits_u(K) / n_u
Recall@K, trunc = hits_u(K) / min(K, n_u)                          (Liang et al. 2018)
DCG@K           = sum over i in R_u with rank_u(i) <= K of 1 / log2(rank_u(i) + 1)
IDCG@K          = sum over j = 1 .. min(K, n_u) of 1 / log2(j + 1)
NDCG@K          = DCG@K / IDCG@K                                   (binary relevance)
MRR@K           = 1 / r if r <= K, else 0, where r = min over i in R_u of rank_u(i)
```

- **Leave-one-out** (`LOO-*`): n_u = 1. With r the rank of the held-out item, HR@K = Recall@K = truncated Recall@K = 1[r ≤ K], NDCG@K = 1 / log2(r + 1) for r ≤ K (IDCG = 1), and MRR@K = 1/r for r ≤ K. Recall and HitRate are the same number, so tables report HR@{10, 50, 100, 500}, NDCG@10 and MRR@10 (D-007). HR@500 is the ceiling of the ranker, which only re-orders the Top-500.
- **Strong generalisation** (`Liang-SG-*`): n_u is about 20% of the user's positives, so plain Recall@K cannot reach 1 when n_u > K; the truncated form divides by min(K, n_u) and can. Headline: truncated Recall@20 and Recall@50, NDCG@100 and NDCG@10. The published Recall@20/50 and NDCG@100 anchors are `Liang-SG-strict`; the NDCG@10 anchors are `Liang-SG-fit` (D-001).

### 8.3 Aggregation and reporting

- A table value is the mean of the per-user values over the evaluated users. Misses count as 0, including end-to-end ranker metrics, where a user whose item is not retrieved scores 0 (D-006). Any number conditioned on a retrieval hit is labelled as such.
- Per-user vectors are stored under `artifacts/eval/<run_id>/`, so CIs and comparisons are recomputed without re-scoring.
- Subgroups (D-011): with or without a boundary tie, recomputed from the split files, and by the time span of the user's positives (≤ 1 hour, ≤ 1 day, longer).
- ANN study (D-008): overlap@500 is the mean over users of |ANN top-500 ∩ exact top-500| / 500, reported with the HR@500 Δ against exact search and p50 / p95 latency.

### 8.4 Bootstrap

1. **Inputs.** Per-user vectors a and b of two arms over the same users in `user_idx` order. For an arm with several training seeds, each user's value is the mean over seeds.
2. **Resampling.** `rng = numpy.random.default_rng(42)`. For each of 2,000 replicates, draw n indices with replacement (`rng.integers(0, n, size=n)`) and record Δ_r = mean(a[idx]) − mean(b[idx]). Both arms use the same indices in every replicate: the bootstrap is paired.
3. **Interval.** The reported Δ is mean(a) − mean(b) on the full sample; its 95% CI is the 2.5th to 97.5th percentile of the Δ_r (percentile method). A difference is claimed only if the paired user-bootstrap CI excludes 0 and the per-seed paired Δs (seed k of arm A vs seed k of arm B, same users) all have the CI's sign; tables also report the per-seed Δ range. Overlapping marginal CIs are never used as a test.
4. **Single arm.** The same procedure on a alone gives its CI. Seeding with 42 for every computation over the same users gives every arm in a table identical resamples.
5. **Memory.** Indices are drawn in chunks of 100 replicates (100 × 136,674 int64 indices ≈ 109 MB) to bound memory; the chunk size is fixed so that results reproduce exactly.
6. **Seeds.** Each trained arm runs at least 3 training seeds; tables give mean ± sd (ddof = 1) of the per-seed means next to the bootstrap CI. The bootstrap captures user-sampling variance only: at n = 136,674 its half-width on HR@50 is at most ~0.27 points (binomial, p = 0.5), so under `LOO-*` seed variance usually dominates; under the Liang protocol (10,000 test users) the half-width is ~±1.0 point, so user sampling also matters.
7. **Unit.** Users. Ablation (c) scores (user, tag) query events, each a (user, tag) pair with its relevant movie set (~2,000 events), and uses a cluster bootstrap: users are resampled together with all their query events, because one user contributes many events.

## 9. Environment

`pyproject.toml` and `uv.lock` are the source of truth; this section summarises the packaging contract they implement.

### 9.1 Python and dependencies

- Python `>=3.11,<3.13`.
- Core, installed by every `uv sync`: `polars>=2.0,<3` (2.0.0 shipped on 2026-10-06; the eager API used here is unchanged), `pyarrow`, `duckdb`, `numpy>=2.0`, `scipy>=1.12` (sparse matrices: evaluation, EASE, BM25), `typer`, `pydantic`, `pyyaml`. Typer has vendored Click since 0.26, so code never imports `click`.

| Extra | Contents | Use |
|---|---|---|
| `train` | `torch>=2.6`, `wandb`, `lightgbm`, `scikit-learn`, `sentence-transformers>=5.0`, `tqdm`; no vector index | training and the ranker |
| `llm` | `anthropic>=1.0,<2` (1.x is a breaking major); Haiku's model id is `claude-haiku-5-5` | query expansion |
| `demo` | `streamlit>=1.41`, `pymilvus[milvus-lite]>=3.0`, `milvus-lite>=3.2`, `lightgbm`, `anthropic>=1.0,<2` | the Streamlit app |
| `eda` | notebook and plotting tools | `notebooks/01_eda.ipynb` |
| `dev` | `eda`, `pymilvus[milvus-lite]`, and the test, lint and type-check tools | tests and CI |

- Milvus Lite 3.x is pure Python over `faiss-cpu`, which it installs; pymilvus 3.x imports pandas. The project itself no longer depends on FAISS, and the `cpu` / `gpu` FAISS extras and their conflicts are gone (2026-10-23).
- Dropped: `bitsandbytes`, because the ~78 MB model never needs quantisation, and `rank-bm25`, since BM25 is a scipy.sparse matrix (D-010).

### 9.2 torch builds

- **Linux:** torch comes from the explicit index `https://download.pytorch.org/whl/cu126` (torch 2.14.1+cu126), which runs on NVIDIA drivers ≥ 525 (CUDA 12 minor-version compatibility; ≥ 560 natively) and covers Colab, Kaggle, RunPod and Modal hosts. PyPI's default Linux wheel links CUDA 13 and needs driver ≥ 580.
- **Blackwell GPUs** (RTX 50xx, B200) need the cu130 index instead: switch the index URL in `pyproject.toml`, run `uv lock`, and sync again.
- **macOS:** the PyPI wheel, with MPS on Apple Silicon.

### 9.3 Install commands

| Where | Command |
|---|---|
| Laptop (data, evaluation, tests) | `uv sync --extra dev` |
| Laptop, training on MPS | `uv sync --extra dev --extra train` |
| Cloud GPU box | `bash scripts/bootstrap_cloud.sh`, which runs `uv sync --python 3.12 --extra train --extra llm` and then download and the Liang split (§6.3) |

`uv sync` is exact: it removes packages of extras not named on the command line, so name every extra each time (`uv run --extra X` adds missing packages instead).

### 9.4 System libraries

| Platform | Install | Why |
|---|---|---|
| macOS | `brew install libomp` before importing `lightgbm` | the LightGBM 4.7 wheel needs `libomp.dylib` |
| Debian, Docker image | `libgomp1` (in the `Dockerfile`) | OpenMP runtime that LightGBM links |
| Streamlit Community Cloud | `libgomp1` in the root `packages.txt` | the same, installed by apt on the host |

### 9.5 Streamlit Community Cloud (decided; in place from the Day-2 hello-world deploy)

- **Host:** 690 MB of RAM guaranteed, up to 2.7 GB; at most 2 CPU cores; Python 3.12 by default; Debian 11; the app sleeps after 12 hours idle.
- **Dependency file.** Community Cloud installs from the first dependency file it finds, searching the entrypoint's directory before the repository root, in the order `uv.lock` > `Pipfile` > `environment.yml` > `requirements.txt` > `pyproject.toml`. The committed root `uv.lock` would therefore win, so the demo gets its own file next to the entry point:

  ```bash
  uv export --frozen --no-hashes --no-emit-project --extra demo -o src/shopsage_rec/serving/requirements.txt
  ```

  `--no-emit-project` leaves the local package out; a two-line `sys.path` shim in `app.py` makes `shopsage_rec` importable from `src/`. The root `requirements.txt` serves the `Dockerfile` only.
- **Artifacts** are git-ignored and fetched at startup from a versioned, sha256-pinned GitHub Release asset (§7). What that asset and the public demo may contain is open under D-003: no per-user rating rows, and titles downloaded from GroupLens at startup.

## 10. Change log

| Date | Version | Change |
|---|---|---|
| 2026-10-18 | v0 | Day 1 contract: processed data, `meta.json`, data config, Python API, CLI; planned artifacts listed |
| 2026-10-18 | v0.1 | Review against the code: config bounds and `dataset` value, `write_splits` also writes `meta.json`, extra `meta.json` keys allowed, CLI defaults and `--root`, demo entry point, `min_item_interactions: 1` for the Liang filter |
| 2026-10-18 | v0.2 | splitmix64 `tie_key` over raw ids replaces the affine hash (`compute_tie_key`, `splitmix64`, `tie_key_col`); `meta.json` gains `tie_break`, `users_with_val_train_tie`, `users_with_any_boundary_tie`, `any_boundary_tie_fraction`, `cold_val_items`, `split_sha256`, and `users_with_boundary_tie` becomes an int count (test/val tie only) beside `boundary_tie_fraction`; invariants updated; `remap_ids` passes extra columns through; `parse_movies` year rules for ranges and stray parentheses; raw-table warning and the planned as-of accessor; metric definitions and bootstrap (§8); environment (§9); planned artifacts follow D-003, D-004, D-008 and D-010 |
| 2026-10-23 | v0.4 | S1, efficiency-first re-plan: the Liang split (§2.7, `split.strategy: liang`, `load_liang`), the default config and processed dir follow it; `shopsage evaluate` for MostPop and EASE (§6.1); the `cpu` and `gpu` FAISS extras removed, scipy in the core dependencies, `pymilvus[milvus-lite]` in the demo and dev extras (§9); banner on the sections the re-plan supersedes |
| 2026-10-18 | v0.3 | Final review: `Liang-SG-strict` / `Liang-SG-fit` labels (§4, §8); `split.seed` in [0, 2^64); planned train-config keys (`checkpoint_metric`, `candidates_per_group`, `val_partition`) and deploy-config keys (`bundle`, `llm_expansion`); `load_data_config(root=)` and `--root` resolve to absolute paths and `--root` must exist; eager `--version`; CLI defaults resolve against the repo root; a missing CSV exits 1 with a one-line error; `ValueError` on null ids (`compute_tie_key`, `leave_one_out_split`) and duplicate map keys (`remap_ids`); §5.6 `item_popularity_train` / `item_mean_rating_train` and the missing-value rule; exact-search path and the equivalent over-fetch (§7, §8.1); per-seed paired Δ rule, Liang half-width and cluster bootstrap for ablation (c) (§8.4); GPU-arch fallback and `bench_two_tower_step.py` (§6.3); demo bundle files (§7) |
