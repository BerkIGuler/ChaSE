"""Linear (ridge) and MLP probes of frozen features.

Each probe takes features [n, D] and targets [n, k] of three sets of links. The first set fits the probe. The second
set sets its hyperparameters. The third set gets the predictions and the score. The features are standardized with
the mean and the standard deviation of the first set.

- Ridge. Closed form for every penalty of ALPHAS from one eigendecomposition of X^T X. The penalty with the best R^2
  on the second set is kept.
- MLP. ReLU hidden layers of the given widths, trained with AdamW on mini-batches. The weight decay (WEIGHT_DECAYS)
  and the stopping epoch are chosen on the second set. A regression target is standardized with the statistics of
  the first set.

With pooled, the R^2 is one value from the squared errors pooled over the outputs (the (sin, cos) of an angle).
Otherwise it is the mean of the R^2 of each output.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ALPHAS = np.logspace(-2, 6, 17)          # ridge penalties, on standardized features (sum of squares, not mean)
# Features with more values than this (rows x dimensions of the first set) stay in float32 and are multiplied in
# float64 chunks, so that wide inputs (the 14-symbol raw baselines) fit in memory
LARGE = 2 * 10 ** 8
CHUNK_ROWS = 8192
CONSTANT_SD = 1e-6                       # a feature column with a smaller standard deviation on train is constant
WEIGHT_DECAYS = (1e-4, 1e-2)
MAX_EPOCHS = 100
PATIENCE = 8
BATCH = 2048
LR = 1e-3


def _mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w in float64, x read in row chunks (x may be float32)."""
    if len(x) == 0:
        return torch.zeros(0, w.shape[1], dtype=torch.float64, device=w.device)
    return torch.cat([x[i:i + CHUNK_ROWS].double() @ w for i in range(0, len(x), CHUNK_ROWS)])


def _gram(x: torch.Tensor) -> torch.Tensor:
    """x^T x in float64, accumulated over row chunks."""
    g = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
    for i in range(0, len(x), CHUNK_ROWS):
        c = x[i:i + CHUNK_ROWS].double()
        g += c.T @ c
    return g


def _tmv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x^T y in float64 over row chunks."""
    out = torch.zeros(x.shape[1], y.shape[1], dtype=torch.float64, device=x.device)
    for i in range(0, len(x), CHUNK_ROWS):
        out += x[i:i + CHUNK_ROWS].double().T @ y[i:i + CHUNK_ROWS]
    return out


def _scale(xtr: torch.Tensor):
    """Mean and standard deviation of xtr per column. A column that is constant on xtr is only centered, so that its
    values on the other sets are not multiplied by a huge factor, and a second pass leaves the tensors as they are."""
    mu, sd = xtr.mean(0), xtr.std(0)
    return mu, torch.where(sd < CONSTANT_SD, torch.ones_like(sd), sd)


def standardize(xtr: torch.Tensor, *others: torch.Tensor, inplace: bool = False):
    """Standardize every tensor with the mean and standard deviation of xtr."""
    mu, sd = _scale(xtr)
    if inplace:                             # wide inputs are changed in place, without copies
        return [x.sub_(mu).div_(sd) for x in (xtr, *others)]
    return [(x - mu) / sd for x in (xtr, *others)]


def r2(y: torch.Tensor, p: torch.Tensor, pooled: bool = False) -> float:
    """R^2 averaged over the outputs, or with pooled one R^2 from the squared errors summed over the outputs."""
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    if pooled:
        return float(1 - ss_res.sum() / ss_tot.sum().clamp_min(1e-12))
    return float((1 - ss_res / ss_tot.clamp_min(1e-12)).mean())


def shuffled(y: torch.Tensor, seed: int) -> torch.Tensor:
    """The rows of y in a random order (the shuffled-target control)."""
    g = torch.Generator().manual_seed(seed)
    return y[torch.randperm(len(y), generator=g).to(y.device)]


def ridge_probe(xtr, ytr, xva, yva, xte, yte, alphas=ALPHAS, return_pred=False, pooled=False) -> Dict:
    """Ridge with an intercept. The penalty is chosen by the R^2 on (xva, yva). Inputs are float tensors on one
    device, y [n, k]. Returns the R^2 on test and on val and the penalty. With return_pred, also the test
    predictions [n_test, k] (numpy) as "pred_test"."""
    if xtr.numel() > LARGE:                 # float32 features, float64 products in chunks
        xtr, xva, xte = standardize(xtr.float(), xva.float(), xte.float(), inplace=True)
        mm, tmv = _mm, _tmv
        lam, V = torch.linalg.eigh(_gram(xtr))
    else:
        xtr, xva, xte = (x.to(torch.float64, copy=True) for x in (xtr, xva, xte))
        mu, sd = _scale(xtr)
        for x in (xtr, xva, xte):
            x.sub_(mu).div_(sd)
        mm, tmv = (lambda x, w: x @ w), (lambda x, y: x.T @ y)
        lam, V = torch.linalg.eigh(xtr.T @ xtr)
    y, yva, yte = ytr.double(), yva.double(), yte.double()
    ym = y.mean(0)
    b = V.T @ tmv(xtr, y - ym)
    best = (-np.inf, None, None)
    for a in alphas:
        w = V @ (b / (lam + a)[:, None])
        s = r2(yva, mm(xva, w) + ym, pooled)
        if s > best[0]:
            best = (s, float(a), w)
    s_va, a, w = best
    pred = mm(xte, w) + ym
    out = {"test": r2(yte, pred, pooled), "val": s_va, "alpha": a}
    if return_pred:
        out["pred_test"] = pred.float().cpu().numpy()
    return out


def _mlp(d_in: int, widths: Sequence[int], d_out: int) -> nn.Module:
    layers, d = [], d_in
    for w in widths:
        layers += [nn.Linear(d, w), nn.ReLU()]
        d = w
    return nn.Sequential(*layers, nn.Linear(d, d_out))


def _train(widths, wd, xtr, ytr, xva, yva, seed, pooled=False):
    torch.manual_seed(seed)
    net = _mlp(xtr.shape[1], widths, ytr.shape[1]).to(xtr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
    loss_fn = nn.MSELoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad = -np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        for idx in torch.randperm(len(xtr), generator=g).split(BATCH):
            idx = idx.to(xtr.device)
            opt.zero_grad()
            loss_fn(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            s = r2(yva, net(xva), pooled)
        if s > best:
            best, best_state, bad = s, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net.eval(), best


def mlp_probe(data: Tuple, widths: Sequence[int], seed: int = 0, return_pred: bool = False,
              pooled: bool = False) -> Dict:
    """data is (xtr, ytr, xva, yva, xte, yte), tensors on one device, y [n, k]. Returns the R^2 on test and on val and
    the chosen weight decay. With return_pred, also the test predictions in the units of the target as numpy under
    "pred_test". With pooled, the outputs share one scale, so that the MSE loss weighs them as the pooled R^2 does."""
    xtr, ytr, xva, yva, xte, yte = data
    big = xtr.numel() > LARGE                                      # wide inputs are standardized in place
    xtr, xva, xte = standardize(xtr.float(), xva.float(), xte.float(), inplace=big)
    ytr, yva, yte = ytr.float(), yva.float(), yte.float()
    mu, sd = ytr.mean(0), ytr.std(0).clamp_min(1e-8)               # the R^2 does not change
    if pooled:
        sd = sd.pow(2).mean().sqrt().expand_as(sd)
    ytr, yva, yte = (ytr - mu) / sd, (yva - mu) / sd, (yte - mu) / sd
    best = (-np.inf, None, None)
    for wd in WEIGHT_DECAYS:
        net, s = _train(widths, wd, xtr, ytr, xva, yva, seed, pooled)
        if s > best[0]:
            best = (s, wd, net)
    s_va, wd, net = best
    with torch.no_grad():
        out_te = net(xte)
        res = {"test": r2(yte, out_te, pooled), "val": s_va, "weight_decay": wd}
        if return_pred:
            res["pred_test"] = (out_te * sd + mu).float().cpu().numpy()
        return res
