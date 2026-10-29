"""A small synthetic data set with labels, channels and the features of one model."""

import numpy as np
import pytest
import yaml

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
        h = (rng.normal(size=(n, 2, 4, 8)) + 1j * rng.normal(size=(n, 2, 4, 8))) * gain[:, None, None, None]
        np.save(data / f"{s}.npy", h.astype(np.complex64))
        x = np.stack([lab["path_gain_db"], lab["distance_m"] ** 2], 1)        # reads path gain and distance
        np.save(feats / f"{s}.npy", np.concatenate([x, rng.normal(size=(n, 6))], 1).astype(np.float32))
    cfg = {"data_dir": str(data), "features_dir": str(tmp_path / "features"), "out_dir": str(tmp_path / "out"),
           "models": {"raw": {"raw": "complex"}, "raw_logmag": {"raw": "logmag"}, "toy": {"readouts": ["mean"]}},
           "min_path_gain_db": -150, "levels": {"mlp8": [8]}, "seeds": [0], "report": {"n_boot": 50}}
    path = tmp_path / "chase.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path
