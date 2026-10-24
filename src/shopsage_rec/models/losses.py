"""Sampled softmax for the Two-Tower, with logQ correction and accidental-hit masking (D-005).

Columns of the logit matrix, left to right:

- the batch's own positives (``inbatch=True``): row i's target is column i;
- ``shared`` negatives used by every row (uniform samples, or all items for the full softmax,
  where ``inbatch=False`` and row i's target is the column of its positive item);
- ``rowwise`` negatives, k per row (negatives mined for that row's user).

Logits are cosine similarities divided by ``tau``. ``logq`` (one value per in-batch or shared
column) is subtracted so the sampled softmax approximates the full one (Yi et al. 2019); mined
columns get no correction. Any column holding a training positive of the row's user is set to
-inf, except the row's target column.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


class PositiveSet:
    """Membership test for (user, item) training positives on the training device."""

    def __init__(self, users: torch.Tensor, items: torch.Tensor, n_items: int) -> None:
        self.n_items = n_items
        self.keys = torch.unique(users.long() * n_items + items.long())  # sorted

    def to(self, device: torch.device | str) -> PositiveSet:
        self.keys = self.keys.to(device)
        return self

    def contains(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Broadcast ``users`` against ``items``; True where the pair is a training positive."""
        query = users.long() * self.n_items + items.long()
        idx = torch.searchsorted(self.keys, query).clamp_(max=self.keys.numel() - 1)
        return self.keys[idx] == query


def sampled_softmax(
    user: torch.Tensor,
    pos_vecs: torch.Tensor,
    pos_items: torch.Tensor,
    row_users: torch.Tensor,
    *,
    tau: float,
    positives: PositiveSet | None,
    logq: torch.Tensor | None,
    shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
    rowwise: tuple[torch.Tensor, torch.Tensor] | None = None,
    inbatch: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(loss, logits, target)``; logits and loss are float32."""
    user = user.float()
    blocks: list[torch.Tensor] = []
    items: list[torch.Tensor] = []
    rows = torch.arange(user.shape[0], device=user.device)
    if inbatch:
        logits = user @ pos_vecs.float().T / tau
        if logq is not None:
            logits = logits - logq[None, :]
        blocks.append(logits)
        items.append(pos_items[None, :].expand(user.shape[0], -1))
        target = rows
    else:
        if shared is None:
            raise ValueError("inbatch=False needs shared columns that contain the positives")
        target = pos_items
    if shared is not None:
        vecs, shared_items, shared_logq = shared
        logits = user @ vecs.float().T / tau
        if shared_logq is not None:
            logits = logits - shared_logq[None, :]
        blocks.append(logits)
        items.append(shared_items[None, :].expand(user.shape[0], -1))
    if rowwise is not None:
        vecs, row_items = rowwise
        blocks.append((user[:, None, :] * vecs.float()).sum(-1) / tau)
        items.append(row_items)
    logits = torch.cat(blocks, dim=1)
    if positives is not None:
        hits = positives.contains(row_users[:, None], torch.cat(items, dim=1))
        hits[rows, target] = False
        logits = logits.masked_fill(hits, -torch.inf)
    return F.cross_entropy(logits, target), logits, target
