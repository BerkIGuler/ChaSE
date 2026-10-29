import numpy as np
import torch

import chase.probes as P
from chase.bootstrap import r2_pooled
from chase.probes import mlp_probe, r2, ridge_probe, shuffled


def _split(x, y):
    return x[:600], y[:600], x[600:800], y[600:800], x[800:], y[800:]


def test_ridge_matches_the_closed_form():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 20, generator=g, dtype=torch.float64) * torch.linspace(0.1, 5, 20) + 3
    y = torch.stack([x[:, 0] - 2 * x[:, 3], x[:, 5]], 1) + 0.5 * torch.randn(1000, 2, generator=g)
    res = ridge_probe(*_split(x, y), alphas=[10.0], return_pred=True)
    xs = (x - x[:600].mean(0)) / x[:600].std(0)
    ym = y[:600].mean(0)
    w = torch.linalg.solve(xs[:600].T @ xs[:600] + 10.0 * torch.eye(20, dtype=torch.float64), xs[:600].T @ (y[:600] - ym))
    pred = xs[800:] @ w + ym
    np.testing.assert_allclose(res["pred_test"], pred.numpy(), atol=1e-4)
    assert np.isclose(res["test"], r2(y[800:], pred), atol=1e-6)
    assert ridge_probe(*_split(x, y))["test"] > 0.9


def test_shuffled_control_reads_nothing():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1000, 8, generator=g)
    y = x[:, :2] * 3 + 0.1 * torch.randn(1000, 2, generator=g)
    xtr, ytr, xva, yva, xte, yte = _split(x, y)
    assert ridge_probe(xtr, ytr, xva, yva, xte, yte)["test"] > 0.95
    assert abs(ridge_probe(xtr, shuffled(ytr, 0), xva, yva, xte, yte)["test"]) < 0.05


def test_wide_inputs_take_the_chunked_path(monkeypatch):
    g = torch.Generator().manual_seed(0)
    x = [torch.randn(n, 16, generator=g) for n in (400, 100, 100)]
    w = torch.randn(16, 1, generator=g)
    data = sum(([xi, xi @ w + 0.1 * torch.randn(len(xi), 1, generator=g)] for xi in x), [])
    ref = ridge_probe(*[d.clone() for d in data])
    monkeypatch.setattr(P, "LARGE", 10)                            # force the float32, chunked path
    monkeypatch.setattr(P, "CHUNK_ROWS", 64)
    big = ridge_probe(*[d.clone() for d in data])
    assert abs(big["test"] - ref["test"]) < 1e-4 and big["alpha"] == ref["alpha"]


def test_wide_standardization_is_idempotent(monkeypatch):
    monkeypatch.setattr(P, "LARGE", 10)
    g = torch.Generator().manual_seed(0)
    x = [torch.randn(n, 8, generator=g) for n in (300, 100, 100)]
    for xi in x:
        xi[:, 0] = 5.0                                              # a constant column (zero train variance)
    x[1][:, 0] = 6.0
    a = P.standardize(*[xi.clone() for xi in x], inplace=True)
    b = P.standardize(*[ai.clone() for ai in a], inplace=True)
    for ai, bi in zip(a, b):
        torch.testing.assert_close(ai, bi, rtol=1e-4, atol=1e-4)   # a second pass changes nothing
    assert a[1][:, 0].abs().max() < 2                               # not scaled by 1e8
    data = sum(([xi, xi[:, 1:2]] for xi in x), [])
    assert mlp_probe(tuple(data), [16])["test"] > 0.5               # the wide path of the MLP runs


def test_mlp_pooled_scores_in_target_units():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 8)).astype(np.float32)
    y = np.stack([x[:, 0], 0.2 * x[:, 1] + 0.1 * rng.normal(size=600)], 1).astype(np.float32)
    t = [torch.from_numpy(a) for a in (x[:400], y[:400], x[400:500], y[400:500], x[500:], y[500:])]
    r = mlp_probe(t, [16], 0, return_pred=True, pooled=True)
    assert np.isclose(r["test"], r2_pooled(y[500:], r["pred_test"]), atol=1e-4)


def test_mlp_reads_a_nonlinear_quantity():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(12000, 4, generator=g)
    y = (x[:, :1] ** 2) + 0.05 * torch.randn(12000, 1, generator=g)
    data = (x[:10000], y[:10000], x[10000:11000], y[10000:11000], x[11000:], y[11000:])
    assert ridge_probe(*data)["test"] < 0.1 and mlp_probe(data, [64])["test"] > 0.8


def test_constant_train_column_is_not_amplified():
    g = torch.Generator().manual_seed(0)
    x = [torch.randn(n, 4, generator=g) for n in (10000, 1000, 1000)]
    x[0][:, 3] = 1.0                                                # constant on train, varies on val and test
    x[1][:, 3] += 1.0
    x[2][:, 3] += 1.0
    y = [xi[:, :1] for xi in x]
    data = (x[0], y[0], x[1], y[1], x[2], y[2])
    assert ridge_probe(*data)["test"] > 0.99 and mlp_probe(data, [16])["test"] > 0.9
    assert P.standardize(x[0], x[2])[1][:, 3].abs().max() < 10


def test_features_of_a_small_scale_are_standardized():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 8, generator=g, dtype=torch.float64)
    y = x[:, :1] + 0.1 * torch.randn(1000, 1, generator=g, dtype=torch.float64)
    assert ridge_probe(*_split(x * 1e-7, y))["test"] > 0.95
