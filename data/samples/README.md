# data/samples

Locally generated subsamples of MovieLens-20M for fast iteration (pipeline smoke runs,
notebook prototyping). Everything in this folder except this README is git-ignored
(`.gitignore`: `data/samples/*`, `!data/samples/README.md`).

## Why the repo ships no sample data

The GroupLens usage licence (full text in `data/raw/ml-20m/README.txt` after download) does
not allow redistributing the data without separate permission. A subsample is still
MovieLens data, so it is never committed, attached to a GitHub release, baked into the
Docker image, deployed with the Streamlit demo or logged as a W&B artifact. Each machine
regenerates samples from its own md5-verified download:

```bash
scripts/download_data.sh    # -> data/raw/ml-20m/
```

Unit tests do not use this folder: `tests/conftest.py` writes tiny synthetic
MovieLens-shaped CSVs to a temporary directory, so CI never needs the real data.

## Generating a sample

No `shopsage` command does this, and none is in the 10-day plan. Run this from the repo root
after the download:

```bash
uv run python - <<'PY'
import shutil
from pathlib import Path

import polars as pl

raw, out = Path("data/raw/ml-20m"), Path("data/samples/ml-20m-u5000")
out.mkdir(parents=True, exist_ok=True)
# infer_schema=False reads every field as a string, so values are copied verbatim.
ids = pl.scan_csv(raw / "ratings.csv", infer_schema=False).select("userId").unique()
users = ids.collect().sort("userId").sample(n=5000, seed=42)
for name in ("ratings.csv", "tags.csv"):
    rows = pl.scan_csv(raw / name, infer_schema=False)
    rows.join(users.lazy(), on="userId", how="semi").sink_csv(out / name)
shutil.copy(raw / "movies.csv", out / "movies.csv")
PY
```

| Aspect | Choice |
|---|---|
| Sampling unit | Users, with their complete rating and tag histories, so per-user leave-one-out splits stay valid |
| Determinism | `sample(n=5000, seed=42)` over the sorted user ids: the same users on every run with a given Polars version |
| Output | `ratings.csv` and `tags.csv` filtered to the sampled users; `movies.csv` copied unchanged (the raw layout the loaders read) |
| Use | Copy `configs/data/movielens_20m.yaml` into the sample folder (git-ignored), point `raw_dir` at the sample and `processed_dir` at e.g. `data/samples/ml-20m-u5000/processed`, then run `scripts/prepare_data.sh <copy>` |

## Citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
https://doi.org/10.1145/2827872
