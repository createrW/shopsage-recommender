"""ShopSage Recommender: Two-Tower retrieval + LightGBM ranking on MovieLens-20M.

Sub-packages (mirrors the two-stage recsys architecture):

- ``data``        load / preprocess / split MovieLens-20M with Polars
- ``features``    user & item feature engineering (genre multi-hot, year, activity priors)
- ``models``      Two-Tower PyTorch model + training loop (in-batch softmax, mixed negatives)
- ``retrieval``   exact top-k (default), FAISS ANN study, BM25 baseline, hybrid fusion
- ``ranking``     LightGBM re-ranker over Top-500 candidates
- ``evaluation``  HR@K / Recall@K / NDCG@K / MRR@10 with paired bootstrap confidence intervals
- ``llm``         optional LLM query expansion (Claude Haiku)
- ``serving``     Streamlit demo entry point (offline artifacts, CPU-only)
"""

__version__ = "0.1.0"
