"""Two-Tower model, history sampling, sampled-softmax variants and the numpy user tower."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from shopsage_rec.features.items import ItemFeatures  # noqa: E402
from shopsage_rec.models.losses import PositiveSet, sampled_softmax  # noqa: E402
from shopsage_rec.models.numpy_tower import user_vectors as np_user_vectors  # noqa: E402
from shopsage_rec.models.two_tower import TwoTower, sample_history  # noqa: E402


def _feats(n_items: int = 30, seed: int = 0) -> ItemFeatures:
    rng = np.random.default_rng(seed)
    return ItemFeatures(
        genres=(rng.random((n_items, 19)) < 0.2).astype(np.float32),
        year_bucket=rng.integers(0, 6, n_items),
        n_year_buckets=6,
        genome=rng.standard_normal((n_items, 8)).astype(np.float32),
        genome_missing=(rng.random(n_items) < 0.3).astype(np.float32),
    )


def test_sample_history_never_contains_the_target_and_stays_within_the_user() -> None:
    # user 0 owns rows 0-4 (items 10-14), user 1 rows 5-10 (items 20-25)
    indptr = torch.tensor([0, 5, 11])
    indices = torch.tensor([10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25])
    rows = torch.tensor([2, 7, 0])
    users = torch.tensor([0, 1, 0])
    gen = torch.Generator().manual_seed(0)

    hist, weight, n_hist = sample_history(indptr, indices, rows, users, 40, generator=gen)

    assert hist.shape == weight.shape == (3, 40)
    assert n_hist.tolist() == [4.0, 5.0, 4.0]
    kept = [set(hist[r][weight[r] > 0].tolist()) for r in range(3)]
    assert kept[0] <= {10, 11, 13, 14}
    assert kept[1] <= {20, 21, 23, 24, 25}
    assert kept[2] <= {11, 12, 13, 14}
    assert all(k for k in kept)


def test_towers_return_unit_vectors() -> None:
    model = TwoTower(_feats(), dim=16, id_dim=8, hist_dim=8, hidden=32)
    hist = torch.tensor([[1, 2, 3], [4, 4, 0]])
    weight = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    items = model.item_vectors()
    users = model.user_vectors(hist, weight, torch.tensor([3.0, 2.0]))

    assert items.shape == (30, 16)
    assert users.shape == (2, 16)
    assert torch.allclose(items.norm(dim=1), torch.ones(30), atol=1e-5)
    assert torch.allclose(users.norm(dim=1), torch.ones(2), atol=1e-5)


def test_numpy_user_tower_matches_torch_on_1000_users() -> None:
    torch.manual_seed(0)
    model = TwoTower(_feats(n_items=200), dim=32, id_dim=16, hist_dim=24, hidden=64).eval()
    rng = np.random.default_rng(1)
    fold_in = sp.random(1000, 200, density=0.05, format="csr", random_state=rng, dtype=np.float32)
    fold_in.data[:] = 1.0
    fold_in[np.flatnonzero(np.diff(fold_in.indptr) == 0), 0] = 1.0  # every user has history
    fold_in = sp.csr_matrix(fold_in)

    with torch.no_grad():
        expected = model.user_vectors_csr(
            torch.from_numpy(fold_in.indptr.astype(np.int64)),
            torch.from_numpy(fold_in.indices.astype(np.int64)),
        ).numpy()
    got = np_user_vectors(fold_in, model.export_user_tower())

    assert np.isfinite(got).all()
    assert np.abs(got - expected).max() < 1e-5


def test_positive_set_membership() -> None:
    pos = PositiveSet(torch.tensor([0, 0, 2]), torch.tensor([3, 5, 1]), n_items=10)

    got = pos.contains(torch.tensor([[0], [2], [1]]), torch.tensor([[3, 5, 1]]))

    assert got.tolist() == [[True, True, False], [False, False, True], [False, False, False]]


def _unit(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def test_in_batch_logits_mask_accidental_hits_but_never_the_target() -> None:
    u = _unit(torch.randn(3, 4))
    v = _unit(torch.randn(3, 4))
    pos_items = torch.tensor([7, 8, 7])  # rows 0 and 2 share item 7
    users = torch.tensor([0, 1, 2])
    positives = PositiveSet(torch.tensor([0, 1, 1, 2]), torch.tensor([7, 8, 7, 7]), n_items=10)

    _, logits, target = sampled_softmax(
        u, v, pos_items, users, tau=0.1, positives=positives, logq=None
    )

    assert target.tolist() == [0, 1, 2]
    assert torch.isfinite(logits.diagonal()).all()
    assert logits[0, 2] == -torch.inf  # item 7 again, a positive of user 0
    assert logits[2, 0] == -torch.inf
    assert logits[1, 0] == -torch.inf  # user 1 also has item 7
    assert torch.isfinite(logits[0, 1])


def test_logq_is_subtracted_from_every_shared_column() -> None:
    u = _unit(torch.randn(2, 4))
    v = _unit(torch.randn(2, 4))
    logq = torch.tensor([0.5, -1.0])

    _, plain, _ = sampled_softmax(
        u, v, torch.tensor([1, 2]), torch.tensor([0, 1]), tau=0.2, positives=None, logq=None
    )
    _, corrected, _ = sampled_softmax(
        u, v, torch.tensor([1, 2]), torch.tensor([0, 1]), tau=0.2, positives=None, logq=logq
    )

    assert torch.allclose(plain - corrected, logq.expand(2, 2))


def test_full_softmax_targets_the_positive_among_all_unmasked_items() -> None:
    torch.manual_seed(0)
    n = 6
    u = _unit(torch.randn(2, 4))
    all_items = _unit(torch.randn(n, 4))
    pos_items = torch.tensor([1, 4])
    positives = PositiveSet(torch.tensor([0, 0, 1]), torch.tensor([1, 3, 4]), n_items=n)

    loss, logits, target = sampled_softmax(
        u,
        all_items[pos_items],
        pos_items,
        torch.tensor([0, 1]),
        tau=0.5,
        positives=positives,
        logq=None,
        shared=(all_items, torch.arange(n), None),
        inbatch=False,
    )

    assert target.tolist() == [1, 4]
    assert logits[0, 3] == -torch.inf  # user 0's other positive
    expected = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(loss, expected)


def test_mined_columns_are_appended_per_row_after_the_shared_ones() -> None:
    u = _unit(torch.randn(2, 4))
    v = _unit(torch.randn(2, 4))
    mined = _unit(torch.randn(2, 3, 4))

    _, logits, _ = sampled_softmax(
        u,
        v,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        tau=1.0,
        positives=None,
        logq=None,
        rowwise=(mined, torch.tensor([[5, 6, 7], [5, 6, 8]])),
    )

    assert logits.shape == (2, 5)
    assert torch.allclose(logits[:, 2:], (u[:, None, :] * mined).sum(-1))
