"""ShopSage Recommender: Two-Tower retrieval + LightGBM ranking on MovieLens-20M.

Sub-packages (mirrors the two-stage recsys architecture):

- ``data``        load / preprocess / split MovieLens-20M with Polars (leave-one-out, Liang)
- ``features``    user & item feature engineering (genre multi-hot, year, activity priors)
- ``models``      Two-Tower PyTorch model + training loop (in-batch softmax, mixed negatives)
- ``retrieval``   exact top-k offline, BM25 path, RRF hybrid fusion, Milvus Lite serving index
- ``ranking``     LightGBM re-ranker over the dense and BM25 candidates
- ``evaluation``  truncated Recall@K / NDCG@K / HR@K with paired bootstrap confidence intervals
- ``llm``         optional LLM query expansion (Claude Haiku)
- ``serving``     Streamlit entry point: deployment probe, then the CPU-only demo
"""

__version__ = "0.1.0"
