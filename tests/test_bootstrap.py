import numpy as np
import torch

from chase.bootstrap import ci95, levels, r2_draws, r2_mean, r2_pooled, tile_draws, tiles
from chase.probes import r2


def _data(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    _, tile = np.unique(rng.integers(0, 150, n), return_inverse=True)
    y = rng.normal(size=(n, 2))
    return y, y + rng.normal(scale=0.6, size=(n, 2)), tile


def test_r2_draws_full_sample():
    """A draw that takes every tile once is the full sample."""
    y, p, tile = _data()
    one = np.ones((1, tile.max() + 1))
    assert np.isclose(r2_draws(y, p, tile, one)[0], r2_mean(y, p))
    assert np.isclose(r2_draws(y, p, tile, one, pooled=True)[0], r2_pooled(y, p))


def test_r2_draws_matches_a_loop_bootstrap():
    y, p, tile = _data()
    lo, hi = ci95(r2_draws(y, p, tile, tile_draws(tile, 3000, 0)))
    rng = np.random.default_rng(0)
    members = [np.flatnonzero(tile == g) for g in range(tile.max() + 1)]
    vals = []
    for _ in range(3000):
        i = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        vals.append(r2_mean(y[i], p[i]))
    lo2, hi2 = ci95(np.array(vals))
    assert abs(lo - lo2) < 0.01 and abs(hi - hi2) < 0.01


def test_tile_draws_count_every_draw():
    _, _, tile = _data()
    assert np.all(tile_draws(tile, 50, 0).sum(1) == tile.max() + 1)


def test_ci95_ignores_nan_draws():
    assert ci95(np.array([np.nan, 0.1, 0.2, 0.3])) == ci95(np.array([0.1, 0.2, 0.3]))
    assert ci95(np.array([np.nan, np.nan])) == (None, None)


def test_torch_r2_matches_numpy():
    y, p, _ = _data()
    y[:, 1] *= 0.3                          # unequal output variances, so pooled and mean differ
    assert np.isclose(r2(torch.from_numpy(y), torch.from_numpy(p), pooled=True), r2_pooled(y, p))
    assert np.isclose(r2(torch.from_numpy(y), torch.from_numpy(p)), r2_mean(y, p))


def test_tiles_group_nearby_links_per_city():
    lab = {"city": np.array([0, 0, 0, 1]),
           "rx_position": np.array([[1, 1, 0], [49, 2, 0], [51, 2, 0], [1, 1, 0]], np.float32)}
    t = tiles(lab)
    assert t[0] == t[1] and len({t[0], t[2], t[3]}) == 3


def test_levels():
    assert levels(0.5, 0.8) == {"available": 0.5, "entangled": 0.30000000000000004, "unread": 0.19999999999999996}
    assert levels(-0.2, 0.4)["available"] == 0.0
    assert levels(0.6, 0.4) == {"available": 0.6, "entangled": 0.0, "unread": 0.4}
