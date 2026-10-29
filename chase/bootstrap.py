"""Scores, spatial tiles, the tile bootstrap and the levels of a quantity.

Links in the same area have similar channels, so they are not independent. The bootstrap thus draws whole tiles of
the test links (TILE_M squares of the user positions, per city) with replacement. All models and quantities share the
same draws, so differences between models and means over a group are paired.
"""

from typing import Dict

import numpy as np

TILE_M = 50.0


def r2_mean(y: np.ndarray, p: np.ndarray) -> float:
    """R^2 averaged over the outputs."""
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0).clip(1e-12)
    return float((1 - ss_res / ss_tot).mean())


def r2_pooled(y: np.ndarray, p: np.ndarray) -> float:
    """One R^2 from the squared errors pooled over the outputs, 1 - sum ||e||^2 / sum ||y - mean(y)||^2."""
    return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean(0)) ** 2).sum(), 1e-12))


def tiles(labels: Dict[str, np.ndarray], tile_m: float = TILE_M) -> np.ndarray:
    """A tile id per link, from its city and its user position in tile_m squares. The links of one user to several
    sites share a tile."""
    pos = labels["rx_position"][:, :2].astype(np.float64)
    key = np.stack([labels["city"].astype(np.int64),
                    np.floor(pos[:, 0] / tile_m).astype(np.int64), np.floor(pos[:, 1] / tile_m).astype(np.int64)], 1)
    _, tid = np.unique(key, axis=0, return_inverse=True)
    return tid.ravel()


def tile_draws(tile: np.ndarray, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """[n_boot, n_tiles] counts of each tile in a bootstrap of whole tiles (tile ids 0..n_tiles-1). A draw of n_tiles
    tiles with replacement gives multinomial counts."""
    g = int(tile.max()) + 1
    return np.random.default_rng(seed).multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(np.float64)


def r2_draws(y: np.ndarray, p: np.ndarray, tile: np.ndarray, draws: np.ndarray, pooled: bool = False) -> np.ndarray:
    """R^2 (mean over the outputs, or pooled over them) in every bootstrap draw, from per-tile sums. A tile drawn c
    times adds c times its squared errors, its sums of y and y^2 and its count."""
    y, p = y.reshape(len(y), -1).astype(np.float64), p.reshape(len(p), -1).astype(np.float64)
    g = draws.shape[1]
    sums = lambda v: np.stack([np.bincount(tile, v[:, j], g) for j in range(v.shape[1])], 1)   # noqa: E731
    res, sy, syy = draws @ sums((y - p) ** 2), draws @ sums(y), draws @ sums(y ** 2)
    n = draws @ np.bincount(tile, minlength=g).astype(np.float64)
    tot = syy - sy ** 2 / n[:, None]
    if pooled:
        return 1 - res.sum(1) / np.maximum(tot.sum(1), 1e-12)
    return (1 - res / np.maximum(tot, 1e-12)).mean(1)


def ci95(v: np.ndarray, alpha: float = 0.05):
    """Percentile interval over the finite draws. (None, None) when no draw is finite, so that the JSON stays
    valid."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None, None
    lo, hi = np.quantile(v, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def levels(r_lin: float, r_best: float) -> Dict[str, float]:
    """The available, entangled and unread shares of a quantity (scores clipped to [0, 1])."""
    a, b = min(max(r_lin, 0.0), 1.0), min(max(r_best, 0.0), 1.0)
    b = max(a, b)
    return {"available": a, "entangled": b - a, "unread": 1 - b}
