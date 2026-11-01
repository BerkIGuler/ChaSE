"""The partition of CSIGen runs, and the mean power that the builder computes from it."""

import numpy as np
import pytest
import yaml

from chase.cfr import cfr_from_paths
from chase.data import build_cell
from chase.partition import EXCLUDED, GUARD, PARTS, TEST, TRAIN, VAL, make_partition, val_users

CFR = {"carrier_frequency": 3.5e9, "num_subcarriers": 8, "num_ofdm_symbols": 3, "subcarrier_spacing": 30e3,
       "normalize_delays": False, "normalize": False}
ARRAY = {"num_rows": 1, "num_cols": 4, "num_ant": 4}
SPLIT = {"name": "new_cell", "num_sectors": 1, "held_out_site": {"a_1": 1, "b_1": 0},
         "val": {"fraction": 0.2, "block_m": 50, "guard_m": 10, "seed": 0}}


def _run(run, rng, n_users=40, n_sites=2):
    """A CSIGen-like run of n_sites sites with one sector each. Each file holds the links of its site to all users."""
    run.mkdir(parents=True)
    pos = np.c_[rng.uniform(0, 300, (n_users, 2)), np.full(n_users, 1.5)]
    serving = rng.integers(0, n_sites, n_users)
    for tx in range(n_sites):
        counts = rng.integers(1, 4, n_users)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        p = int(offsets[-1])
        paths = {"paths_offsets": offsets,
                 "paths_a": ((rng.normal(size=(p, 1, 4)) + 1j * rng.normal(size=(p, 1, 4))) * 1e-5
                             ).astype(np.complex64),
                 "paths_tau": rng.uniform(1e-7, 2e-6, p).astype(np.float32),
                 "paths_doppler": rng.uniform(-500, 500, p).astype(np.float32)}
        meta = {"num_valid_channels": n_users, "num_sectors": 1, "tx_idx": tx,
                "rx_names": [f"UE_{u}" for u in range(n_users)],
                "rx_positions": pos, "rx_serving_tx": serving, "rx_velocities": rng.normal(size=(n_users, 3)),
                "rx_orientations": np.zeros((n_users, 3)), "tx_orientation": np.zeros(3),
                "los_binary": rng.integers(0, 2, n_users), "cfr_params": CFR, "tx_array": ARRAY,
                "rx_array": {"num_rows": 1, "num_cols": 1, "num_ant": 1},
                "path_stats_fields": ["path_gain_db"], "path_stats": {"path_gain_db": rng.normal(size=n_users)}}
        np.savez(run / f"channel_tx_{tx}.npz", h=cfr_from_paths(paths, **CFR), metadata=np.array(meta, dtype=object),
                 **paths)
    return pos[:, :2], serving


def test_val_users_take_whole_blocks_and_a_guard():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 500, (400, 2))
    users = np.arange(400)
    v, g = val_users(users, pos, 0.1, 50, 10, np.random.default_rng(1))
    assert len(v) >= 40 and not set(v) & set(g)
    block = np.floor((pos - pos.min(0)) / 50).astype(int)
    vb = {tuple(b) for b in block[v]}
    assert all(tuple(block[u]) not in vb for u in users if u not in set(v))     # whole blocks only
    d = np.sqrt(((pos[:, None] - pos[v][None]) ** 2).sum(-1)).min(1)
    assert set(g) == {u for u in users if u not in set(v) and d[u] < 10}
    with pytest.raises(ValueError, match="between 0 and 1"):
        val_users(users, pos, 1.0, 50, 10, rng)


def test_make_partition_and_the_mean_power(tmp_path):
    rng = np.random.default_rng(2)
    runs = tmp_path / "runs"
    info = {c: _run(runs / c, rng) for c in ("a_1", "b_1")}
    m = make_partition([runs / "a_1", runs / "b_1"], SPLIT, tmp_path / "part")
    p = dict(np.load(tmp_path / "part" / "partition.npz"))
    assert list(p["cities"]) == ["a_1", "b_1"] and list(p["parts"]) == list(PARTS)
    for ci, city in enumerate(("a_1", "b_1")):
        _, serving = info[city]
        held = SPLIT["held_out_site"][city]
        c = p["city"] == ci
        own = p["own_site"][c]
        assert np.array_equal(own, serving[p["user"][c]] == p["tx"][c])
        site_held = (p["tx"][c] == held) | (serving[p["user"][c]] == held)
        assert np.all(p["part"][c][site_held & own & (p["tx"][c] == held)] == TEST)
        assert np.all(np.isin(p["part"][c][site_held], [TEST, EXCLUDED]))
        assert np.all(np.isin(p["part"][c][~site_held], [TRAIN, VAL, GUARD]))
        assert (p["part"][c] == VAL).any() and m["counts"][city]["test_own_site"] > 0
    with pytest.raises(FileExistsError):
        make_partition([runs / "a_1", runs / "b_1"], SPLIT, tmp_path / "part")
    again = make_partition([runs / "b_1", runs / "a_1"], SPLIT, tmp_path / "part2")   # the seed follows the config
    q = dict(np.load(tmp_path / "part2" / "partition.npz"))
    assert np.array_equal(q["part"][q["city"] == 1], p["part"][p["city"] == 0]) and again["split"] == SPLIT

    built = build_cell(runs, tmp_path / "part", tmp_path / "data")
    power, n = 0.0, 0
    for ci, city in enumerate(("a_1", "b_1")):
        for tx in (0, 1):
            with np.load(runs / city / f"channel_tx_{tx}.npz", allow_pickle=True) as z:
                h = z["h"]
            rows = p["row"][(p["city"] == ci) & (p["tx"] == tx) & (p["part"] == TRAIN)]
            power += float((np.abs(h[rows].astype(np.complex128)) ** 2).sum())
            n += h[rows].size
    assert np.isclose(built["statistics"]["mean_power"], power / n, rtol=1e-12)
    assert built["statistics"]["mean_power_links"] == "all links of the train part"


def test_published_power_is_copied(tmp_path):
    rng = np.random.default_rng(3)
    runs = tmp_path / "runs"
    for c in ("a_1", "b_1"):
        _run(runs / c, rng)
    make_partition([runs / "a_1", runs / "b_1"], SPLIT, tmp_path / "part")
    m = yaml.safe_load((tmp_path / "part" / "manifest.yaml").read_text())
    m["statistics"] = {"mean_power": 7e-9}
    (tmp_path / "part" / "manifest.yaml").write_text(yaml.safe_dump(m))
    assert build_cell(runs, tmp_path / "part", tmp_path / "data")["statistics"] == {"mean_power": 7e-9}


def test_partition_input_checks(tmp_path):
    pos = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 5.0]])
    with pytest.raises(ValueError, match="one block"):
        val_users(np.arange(3), pos, 0.1, 50, 10, np.random.default_rng(0))
    rng = np.random.default_rng(4)
    _run(tmp_path / "x" / "a_1", rng)
    _run(tmp_path / "y" / "b_1", rng)
    with pytest.raises(ValueError, match="all in one folder"):
        make_partition([tmp_path / "x" / "a_1", tmp_path / "y" / "b_1"], SPLIT, tmp_path / "part")
    with pytest.raises(ValueError, match="all in one folder"):
        make_partition([tmp_path / "x" / "a_1", tmp_path / "x" / "a_1"], SPLIT, tmp_path / "part")


def test_other_statistics_are_kept(tmp_path):
    rng = np.random.default_rng(5)
    runs = tmp_path / "runs"
    for c in ("a_1", "b_1"):
        _run(runs / c, rng)
    make_partition([runs / "a_1", runs / "b_1"], SPLIT, tmp_path / "part")
    m = yaml.safe_load((tmp_path / "part" / "manifest.yaml").read_text())
    m["statistics"] = {"note": "kept"}
    (tmp_path / "part" / "manifest.yaml").write_text(yaml.safe_dump(m))
    stats = build_cell(runs, tmp_path / "part", tmp_path / "data")["statistics"]
    assert stats["note"] == "kept" and stats["mean_power"] > 0
