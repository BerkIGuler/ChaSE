"""The run and the report, end to end, on a small synthetic data set."""

import json

import numpy as np
import pytest
import yaml

import chase.run as R
from chase.quantities import GROUPS, QUANTITIES
from chase.report import write_report
from chase.run import load_config, run

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


def test_run_resume_and_report(config, monkeypatch):
    cfg = load_config(config)
    assert cfg["report"]["raw_baselines"] == ["raw", "raw_logmag"] and cfg["quantities"] == QUANTITIES
    run(cfg, str(config), device="cpu")
    out = R.output_dir(cfg)
    res = json.loads((out / "toy" / "mean.json").read_text())
    assert list(res["targets"]) == QUANTITIES and res["code"]["version"]
    assert res["targets"]["aod"]["r2"] == "pooled" and res["targets"]["distance"]["r2"] == "mean"
    n = res["targets"]["path_gain"]["n"]
    assert n["val_fit"] + n["val_select"] < SIZES["val"] and n["test"] < SIZES["test"]   # the coverage floor

    def fail(*args):
        raise AssertionError("a complete result was probed again")
    monkeypatch.setattr(R, "probe_quantity", fail)
    run(cfg, str(config), device="cpu")                         # resumes, nothing to probe

    rep = json.loads(write_report(cfg, str(config)).read_text())
    assert rep["complete"] and set(rep["models"]) == {"raw", "raw_logmag", "toy"}
    toy = rep["models"]["toy"]
    assert set(toy["groups"]) == set(GROUPS) and set(toy["lost"]) == set(QUANTITIES)
    pg = toy["quantities"]["path_gain"]
    assert pg["r_lin"] > 0.9 and pg["r_best"] >= pg["r_lin"] and pg["ci_best"][0] <= pg["r_best"] <= pg["ci_best"][1]
    assert abs(pg["available"] + pg["entangled"] + pg["unread"] - 1) < 1e-9
    assert rep["raw_channel"]["path_gain"]["model_best"] in ("raw", "raw_logmag")
    assert "lost" not in rep["models"]["raw"] and set(rep["controls"]) == {"linear", "mlp8"}
    assert (out / "report.md").read_text().startswith("# ChaSE report")


def test_report_leaves_out_other_settings(config):
    cfg = load_config(config)
    cfg["quantities"] = ["path_gain"]
    cfg["models"] = {"toy": cfg["models"]["toy"]}
    run(cfg, device="cpu")
    cfg["seeds"] = [1]                                           # other settings than the run
    rep = json.loads(write_report(cfg).read_text())
    assert not rep["complete"] and rep["models"] == {} and rep["incomplete"] == {"toy": ["path_gain"]}


def test_config_checks(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"data_dir": "d", "out_dir": "o", "models": {"m": {"readouts": ["mean"]}}}))
    with pytest.raises(ValueError, match="features_dir"):
        load_config(path)
    path.write_text(yaml.safe_dump({"data_dir": "d", "out_dir": "o", "models": {}, "quantities": ["heading"]}))
    with pytest.raises(ValueError, match="not ChaSE quantities"):
        load_config(path)


def test_errors_for_a_constant_quantity_and_bad_features(config, tmp_path):
    cfg = load_config(config)
    cfg["models"] = {"toy": cfg["models"]["toy"]}
    cfg["quantities"] = ["dominant_bounces"]
    data = tmp_path / "data"
    lab = dict(np.load(data / "test_labels.npz"))
    lab["dominant_num_bounces"][:] = 1.0                        # constant on the test links
    np.savez(data / "test_labels.npz", **lab)
    with pytest.raises(ValueError, match="does not vary on the test links"):
        run(cfg, device="cpu")
    f = tmp_path / "features" / "toy" / "mean" / "val.npy"
    x = np.load(f)
    x[0, 0] = np.nan
    np.save(f, x)
    with pytest.raises(ValueError, match="NaN or inf"):
        run(cfg, device="cpu")


def test_resume_records_another_config(config):
    cfg = load_config(config)
    cfg["models"] = {"toy": cfg["models"]["toy"]}
    cfg["quantities"] = ["path_gain"]
    run(cfg, device="cpu")
    cfg["quantities"] = ["path_gain", "distance"]
    run(cfg, device="cpu")
    res = json.loads((R.output_dir(cfg) / "toy" / "mean.json").read_text())
    assert list(res["targets"]) == ["path_gain", "distance"]
    assert res["config"]["quantities"] == ["path_gain"]
    assert [r["config"]["quantities"] for r in res["resumed_with"]] == [["path_gain", "distance"]]
