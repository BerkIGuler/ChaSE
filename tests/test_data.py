"""The channel rebuild and the builder of the sets, on small synthetic CSIGen files."""

import numpy as np
import pytest
import yaml

from chase.cfr import cfr_from_paths, load_cfr
from chase.data import PARTS, build_cell

CFR = {"carrier_frequency": 3.5e9, "num_subcarriers": 8, "num_ofdm_symbols": 3, "subcarrier_spacing": 30e3,
       "normalize_delays": False, "normalize": False}
STATS = ("path_gain_db", "delay_spread_s")


def _csigen_file(path, n, rng, tx_idx):
    """A CSIGen-like channel_tx file with n channels of 0 to 4 paths, 4 TX antennas and the saved h."""
    counts = rng.integers(0, 5, n)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    p = int(offsets[-1])
    paths = {"paths_offsets": offsets,
             "paths_a": ((rng.normal(size=(p, 1, 4)) + 1j * rng.normal(size=(p, 1, 4))) * 1e-5).astype(np.complex64),
             "paths_tau": rng.uniform(1e-7, 2e-6, p).astype(np.float32),
             "paths_doppler": rng.uniform(-500, 500, p).astype(np.float32)}
    h = cfr_from_paths(paths, **CFR)
    meta = {"num_valid_channels": n, "los_binary": rng.integers(0, 2, n), "rx_positions": rng.normal(size=(n, 3)),
            "rx_velocities": rng.normal(size=(n, 3)), "rx_orientations": rng.normal(size=(n, 3)),
            "tx_orientation": rng.normal(size=3), "tx_idx": tx_idx, "cfr_params": CFR,
            "path_stats_fields": list(STATS), "path_stats": {k: rng.normal(size=n) for k in STATS}}
    np.savez(path, h=h, metadata=np.array(meta, dtype=object), **paths)
    return h, meta


def test_load_cfr_gives_the_saved_channels(tmp_path):
    h, _ = _csigen_file(tmp_path / "channel_tx_0.npz", 20, np.random.default_rng(0), 0)
    assert np.array_equal(load_cfr(tmp_path / "channel_tx_0.npz"), h)
    no_paths = np.diff(np.load(tmp_path / "channel_tx_0.npz")["paths_offsets"]) == 0
    assert h.shape == (20, 1, 1, 4, 3, 8) and no_paths.any() and np.abs(h[no_paths]).max() == 0


def test_one_path_in_closed_form():
    """h(f, t) = a exp(-j 2 pi (f_c + f) tau) exp(j 2 pi nu t), with t = n / subcarrier_spacing."""
    a, tau, nu = 2e-5 * np.exp(0.3j), 1e-6, 120.0
    paths = {"paths_offsets": np.array([0, 1]), "paths_a": np.full((1, 1, 1), a, np.complex64),
             "paths_tau": np.array([tau], np.float32), "paths_doppler": np.array([nu], np.float32)}
    h = cfr_from_paths(paths, **CFR)[0, 0, 0, 0]                             # [T, K]
    f = (np.arange(8) - 4) * CFR["subcarrier_spacing"]
    t = np.arange(3) / CFR["subcarrier_spacing"]
    tau, nu = float(np.float32(tau)), float(np.float32(nu))
    ref = np.complex64(a) * np.exp(-2j * np.pi * (CFR["carrier_frequency"] + f[None]) * tau) \
        * np.exp(2j * np.pi * nu * t[:, None])
    np.testing.assert_allclose(h, ref, rtol=1e-5, atol=1e-11)


def test_build_cell(tmp_path):
    rng = np.random.default_rng(1)
    runs, part_dir, out = tmp_path / "runs", tmp_path / "partition", tmp_path / "data"
    files = {}
    for city in ("a_1", "b_1"):
        (runs / city).mkdir(parents=True)
        for tx in range(2):
            files[(city, tx)] = _csigen_file(runs / city / f"channel_tx_{tx}.npz", 10, rng, tx)
    city, tx, row = (np.repeat(np.arange(2), 20), np.tile(np.repeat(np.arange(2), 10), 2), np.tile(np.arange(10), 4))
    part = rng.integers(0, len(PARTS), 40).astype(np.int8)
    part[:3] = [0, 1, 2]                                                  # every set has a link
    own = rng.random(40) < 0.8
    own[:3] = True
    part_dir.mkdir()
    np.savez(part_dir / "partition.npz", cities=np.array(["a_1", "b_1"]), city=city, tx=tx, row=row,
             user=np.arange(40), part=part, own_site=own)
    (part_dir / "manifest.yaml").write_text(yaml.safe_dump({"statistics": {"mean_power": 1e-10}}))
    m = build_cell(runs, part_dir, out)
    assert m["statistics"] == {"mean_power": 1e-10} and list(m["runs"]) == ["a_1", "b_1"]
    for i, s in enumerate(("train", "val", "test")):
        sel = np.flatnonzero((part == i) & own)
        h, lab = np.load(out / f"{s}.npy"), dict(np.load(out / f"{s}_labels.npz"))
        assert m["counts"][s] == len(sel) == len(h) and h.shape[1:] == (3, 4, 8)
        assert np.array_equal(lab["user"], sel) and set(lab) >= {"los_binary", "rx_position", "tx_orientation", *STATS}
        for j, k in enumerate(sel):
            hf, meta = files[(("a_1", "b_1")[city[k]], tx[k])]
            assert np.array_equal(h[j], hf[row[k], 0, 0].transpose(1, 0, 2))
            assert lab["path_gain_db"][j] == np.float32(meta["path_stats"]["path_gain_db"][row[k]])
            assert np.array_equal(lab["rx_position"][j], meta["rx_positions"][row[k]].astype(np.float32))
    with pytest.raises(FileExistsError):
        build_cell(runs, part_dir, out)


def _cell(tmp_path, rng):
    runs, part_dir = tmp_path / "runs", tmp_path / "partition"
    (runs / "a_1").mkdir(parents=True)
    _csigen_file(runs / "a_1" / "channel_tx_0.npz", 6, rng, 0)
    part_dir.mkdir()
    np.savez(part_dir / "partition.npz", cities=np.array(["a_1"]), city=np.zeros(6, int), tx=np.zeros(6, int),
             row=np.arange(6), user=np.arange(6), part=np.array([0, 0, 1, 1, 2, 2], np.int8), own_site=np.ones(6, bool))
    return runs, part_dir


def test_build_checks_the_files(tmp_path):
    rng = np.random.default_rng(2)
    runs, part_dir = _cell(tmp_path, rng)
    _csigen_file(runs / "a_1" / "channel_tx_0.npz", 6, rng, 3)              # the file of another tx
    with pytest.raises(ValueError, match="the file is of tx 3"):
        build_cell(runs, part_dir, tmp_path / "out1")
    _csigen_file(runs / "a_1" / "channel_tx_0.npz", 6, rng, 0)
    z = dict(np.load(part_dir / "partition.npz"))
    z["row"][1] = -1
    np.savez(part_dir / "partition.npz", **z)
    with pytest.raises(ValueError, match="asks for rows -1"):
        build_cell(runs, part_dir, tmp_path / "out2")
    z["row"][1], z["city"][1] = 1, 5                                         # a city index of no run
    np.savez(part_dir / "partition.npz", **z)
    with pytest.raises(ValueError, match="no CSIGen file gave these links"):
        build_cell(runs, part_dir, tmp_path / "out3")
