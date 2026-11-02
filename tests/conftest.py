"""A small synthetic data set with labels, channels and the features of one model."""

import numpy as np
import pytest
import yaml

from chase.cfr import cfr_from_paths

SIZES = {"train": 300, "val": 160, "test": 200}


def _labels(n, rng):
    lab = {k: rng.uniform(0.1, 1.0, n).astype(np.float32) for k in (
        "distance_m", "k_factor_db", "delay_spread_s", "first_delay_s", "dominant_tau_s", "asd_rad", "zsd_rad",
        "asa_rad", "zsa_rad", "dominant_theta_t", "dominant_phi_t", "dominant_theta_r", "dominant_phi_r",
        "dominant_doppler_hz", "max_abs_doppler_hz", "power_frac_los", "power_frac_diffraction",
        "power_frac_specular", "power_frac_refraction", "dominant_power_fraction", "dominant_num_bounces")}
    lab["path_gain_db"] = rng.uniform(-160, -60, n).astype(np.float32)
    lab["city"] = rng.integers(0, 2, n)
    lab["rx_position"] = rng.uniform(0, 400, (n, 3)).astype(np.float32)
    for k in ("rx_velocity", "rx_orientation", "tx_orientation"):
        lab[k] = rng.normal(size=(n, 3)).astype(np.float32)
    return lab


@pytest.fixture
def sizes():
    return SIZES


@pytest.fixture
def config(tmp_path):
    rng = np.random.default_rng(0)
    data, feats = tmp_path / "data", tmp_path / "features" / "toy" / "mean"
    data.mkdir()
    feats.mkdir(parents=True)
    for s, n in SIZES.items():
        lab = _labels(n, rng)
        np.savez(data / f"{s}_labels.npz", **lab)
        gain = 10 ** (lab["path_gain_db"] / 20)
        h = (rng.normal(size=(n, 2, 1, 4, 8)) + 1j * rng.normal(size=(n, 2, 1, 4, 8))) * gain[:, None, None, None, None]
        np.save(data / f"{s}.npy", h.astype(np.complex64))
        x = np.stack([lab["path_gain_db"], lab["distance_m"] ** 2], 1)        # reads path gain and distance
        np.save(feats / f"{s}.npy", np.concatenate([x, rng.normal(size=(n, 6))], 1).astype(np.float32))
    cfg = {"data_dir": str(data), "features_dir": str(tmp_path / "features"), "out_dir": str(tmp_path / "out"),
           "models": {"raw": {"raw": "complex"}, "raw_logmag": {"raw": "logmag"}, "toy": {"readouts": ["mean"]}},
           "min_path_gain_db": -150, "levels": {"mlp8": [8]}, "seeds": [0], "report": {"n_boot": 50}}
    path = tmp_path / "chase.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


CFR = {"carrier_frequency": 3.5e9, "num_subcarriers": 8, "num_ofdm_symbols": 3, "subcarrier_spacing": 30e3,
       "normalize_delays": False, "normalize": False}
ARRAY = {"num_rows": 1, "num_cols": 4, "num_ant": 4}
PARTITION_SPLIT = {"name": "new_cell", "num_sectors": 1, "held_out_site": {"a_1": 1, "b_1": 0},
         "val": {"fraction": 0.2, "block_m": 50, "guard_m": 10, "seed": 0}}


def _csigen_run(run, rng, n_users=40, n_sites=2):
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


@pytest.fixture
def csigen_run():
    """A function that writes a small CSIGen-like run (see _csigen_run)."""
    return _csigen_run


@pytest.fixture
def split():
    """The split config of the synthetic runs a_1 and b_1."""
    return {**PARTITION_SPLIT, "val": dict(PARTITION_SPLIT["val"])}
