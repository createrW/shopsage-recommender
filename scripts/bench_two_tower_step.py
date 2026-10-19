"""Wall-clock and peak memory of one Two-Tower training step on synthetic tensors (E002).

Behind the "measured" M5 / MPS row of the VRAM table (README "Cloud GPU", design.md §13): the
same towers, sampled-softmax loss with logQ and accidental-hit masking, and fused AdamW as the
Day-3 trainer, over random tensors of the exact MovieLens-20M shapes (136,674 users, 13,681
items, 9,704,107 train positives). It measures kernel time and allocator peak, not model
quality; P3-smoke re-measures with the real training loop.

    uv run --extra train python scripts/bench_two_tower_step.py --B 4096 --K 0 --dtype bf16
    uv run --extra train python scripts/bench_two_tower_step.py --B 4096 --K 4096 --device cuda

Prints one JSON line: ms per step, steps and seconds per epoch, and the peak of
``torch.mps.driver_allocated_memory()`` (MPS) or ``torch.cuda.max_memory_allocated()`` (CUDA).
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F  # noqa: N812  (PyTorch convention)
from torch import nn

N_USERS, N_ITEMS, N_TRAIN = 136_674, 13_681, 9_704_107


class Tower(nn.Module):
    """ID embedding + dense features (+ optional year bucket) -> MLP -> L2-normalised vector."""

    def __init__(self, n_ids: int, d: int, n_dense: int, n_year: int = 0, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(n_ids, d)
        self.year = nn.Embedding(n_year, 16) if n_year else None
        in_dim = d + n_dense + (16 if n_year else 0)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(
        self, ids: torch.Tensor, dense: torch.Tensor, year: torch.Tensor | None = None
    ) -> torch.Tensor:
        parts = [self.emb(ids), dense]
        if self.year is not None and year is not None:
            parts.append(self.year(year))
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


def sampled_softmax_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    item_idx: torch.Tensor,
    cand: torch.Tensor,
    logq: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """In-batch (+ sampled) softmax with logQ correction; duplicates of the row's item masked."""
    batch = u.shape[0]
    logits = (u / tau) @ v.T
    logits.sub_(logq[cand].to(logits.dtype))
    mask = item_idx[:, None] == cand[None, :]
    mask[:, :batch].fill_diagonal_(False)  # never mask the row's own positive column
    logits.masked_fill_(mask, float("-inf"))
    return F.cross_entropy(logits, torch.arange(batch, device=u.device))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--B", type=int, default=4096, help="batch size (rows = in-batch columns)")
    ap.add_argument("--K", type=int, default=0, help="extra sampled negative columns per batch")
    ap.add_argument("--steps", type=int, default=30, help="timed steps after 5 warm-up steps")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--opt", default="fused", choices=["fused", "foreach"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(0)
    user = Tower(N_USERS, 128, 25).to(dev)
    item = Tower(N_ITEMS, 128, 20, n_year=12).to(dev)
    params = list(user.parameters()) + list(item.parameters())
    kw = {"fused": True} if args.opt == "fused" else {"foreach": True}
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4, **kw)

    train_u = torch.randint(0, N_USERS, (N_TRAIN,), dtype=torch.int32, device=dev)
    train_i = torch.randint(0, N_ITEMS, (N_TRAIN,), dtype=torch.int32, device=dev)
    user_feat = torch.randn(N_USERS, 25, device=dev)
    item_genre = (torch.rand(N_ITEMS, 20, device=dev) < 0.1).float()
    item_year = torch.randint(0, 12, (N_ITEMS,), device=dev)
    logq = torch.log(torch.rand(N_ITEMS, device=dev) + 1e-3)
    sampler_weights = torch.rand(N_ITEMS, device=dev)
    perm = torch.randperm(N_TRAIN, device=dev)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]

    def sync() -> None:
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    def step(s: int) -> torch.Tensor:
        sel = perm[s * args.B : (s + 1) * args.B]
        uidx = train_u[sel].long()
        iidx = train_i[sel].long()
        negs = torch.multinomial(sampler_weights, args.K, replacement=True) if args.K else iidx[:0]
        cand = torch.cat([iidx, negs])
        with torch.autocast(
            dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
        ):
            u = user(uidx, user_feat[uidx])
            v = item(cand, item_genre[cand], item_year[cand])
            loss = sampled_softmax_loss(u, v, iidx, cand, logq, 0.05)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for s in range(5):
        step(s)
    sync()
    t0 = time.perf_counter()
    for s in range(args.steps):
        loss = step(5 + s)
    sync()
    dt = (time.perf_counter() - t0) / args.steps
    steps_per_epoch = -(-N_TRAIN // args.B)
    peak_gb = None
    if dev.type == "mps":
        peak_gb = round(torch.mps.driver_allocated_memory() / 1e9, 3)
    elif dev.type == "cuda":
        peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    print(
        json.dumps(
            {
                "device": args.device,
                "torch": torch.__version__,
                "B": args.B,
                "K": args.K,
                "dtype": args.dtype,
                "opt": args.opt,
                "ms_per_step": round(dt * 1e3, 2),
                "steps_per_epoch": steps_per_epoch,
                "sec_per_epoch": round(dt * steps_per_epoch, 1),
                "peak_memory_gb": peak_gb,
                "loss": float(loss.detach()),
            }
        )
    )


if __name__ == "__main__":
    main()
