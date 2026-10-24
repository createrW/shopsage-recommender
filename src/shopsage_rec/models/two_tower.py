"""The Two-Tower retriever (D-004): an item tower over item features, an inductive user tower.

- **Item tower:** ``item_idx`` embedding ⊕ genre multi-hot ⊕ year-bucket embedding ⊕ genome
  PCA ⊕ missing-genome flag → MLP → ``dim``, L2-normalised.
- **User tower:** the mean of history-item embeddings (a table of its own) over the user's items
  ⊕ ``log1p(history length)`` → MLP → ``dim``, L2-normalised. There is no user-ID table, so it
  scores users never seen in training (``Liang-SG-strict``) and demo visitors alike.

In training, a row's history is ``max_history`` items drawn with replacement from the user's
other training positives (``sample_history``); at evaluation it is all of the user's input items.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from shopsage_rec.features.items import ItemFeatures


def sample_history(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    rows: torch.Tensor,
    users: torch.Tensor,
    size: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``size`` history items per row, drawn with replacement from the row's user, never the row's
    own item. Returns ``(items, weights, n_history)``; weight 0 marks a draw of the target."""
    start = indptr[users]
    n = indptr[users + 1] - start
    target = rows - start
    draws = torch.rand(rows.shape[0], size, generator=generator, device=rows.device)
    pos = (draws * n[:, None]).long()
    weights = (pos != target[:, None]).float()
    return indices[start[:, None] + pos], weights, (n - 1).float()


class TwoTower(nn.Module):
    genres: torch.Tensor
    year: torch.Tensor
    genome: torch.Tensor
    missing: torch.Tensor

    def __init__(
        self,
        feats: ItemFeatures,
        *,
        dim: int = 128,
        id_dim: int = 64,
        hist_dim: int = 64,
        year_dim: int = 16,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        n_items = feats.n_items
        self.register_buffer("genres", torch.from_numpy(feats.genres))
        self.register_buffer("year", torch.from_numpy(feats.year_bucket.astype(np.int64)))
        self.register_buffer("genome", torch.from_numpy(feats.genome))
        self.register_buffer("missing", torch.from_numpy(feats.genome_missing)[:, None])
        self.item_id = nn.Embedding(n_items, id_dim)
        self.year_emb = nn.Embedding(feats.n_year_buckets, year_dim)
        item_in = id_dim + feats.genres.shape[1] + year_dim + feats.genome.shape[1] + 1
        self.item_mlp = nn.Sequential(nn.Linear(item_in, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.hist_emb = nn.Embedding(n_items, hist_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(hist_dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, dim)
        )
        nn.init.normal_(self.item_id.weight, std=0.05)
        nn.init.normal_(self.hist_emb.weight, std=0.05)

    def item_vectors(self, items: torch.Tensor | None = None) -> torch.Tensor:
        if items is None:
            items = torch.arange(self.item_id.num_embeddings, device=self.genres.device)
        x = torch.cat(
            [
                self.item_id(items),
                self.genres[items],
                self.year_emb(self.year[items]),
                self.genome[items],
                self.missing[items],
            ],
            dim=-1,
        )
        return F.normalize(self.item_mlp(x), dim=-1)

    def _user_head(self, mean: torch.Tensor, n_history: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mean, torch.log1p(n_history)[:, None]], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def user_vectors(
        self, history: torch.Tensor, weights: torch.Tensor, n_history: torch.Tensor
    ) -> torch.Tensor:
        """Users from sampled history items (``weights`` 0 drops a draw)."""
        emb = self.hist_emb(history) * weights[..., None]
        mean = emb.sum(1) / weights.sum(1, keepdim=True).clamp_min(1.0)
        return self._user_head(mean, n_history)

    def user_vectors_csr(self, indptr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Users from all their items, given as CSR row pointers and item indices."""
        counts = (indptr[1:] - indptr[:-1]).float()
        mean = F.embedding_bag(indices, self.hist_emb.weight, indptr[:-1], mode="mean")
        return self._user_head(mean, counts)

    def export_user_tower(self) -> dict[str, np.ndarray]:
        """Weights for ``numpy_tower.user_vectors`` (float32)."""

        def arr(t: torch.Tensor) -> np.ndarray:
            return t.detach().float().cpu().numpy()

        first, last = self.user_mlp[0], self.user_mlp[2]
        assert isinstance(first, nn.Linear)
        assert isinstance(last, nn.Linear)
        return {
            "hist_emb": arr(self.hist_emb.weight),
            "w1": arr(first.weight),
            "b1": arr(first.bias),
            "w2": arr(last.weight),
            "b2": arr(last.bias),
        }
