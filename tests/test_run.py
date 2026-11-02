"""The run and the report, end to end, on a small synthetic data set."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import chase.run as R
from chase.quantities import GROUPS, QUANTITIES
from chase.report import write_report
from chase.run import load_config, run


def test_run_resume_and_report(config, sizes, monkeypatch):
    cfg = load_config(config)
    assert cfg["report"]["raw_baselines"] == ["raw", "raw_logmag"] and cfg["quantities"] == QUANTITIES
    run(cfg, str(config), device="cpu")
    out = R.output_dir(cfg)
    res = json.loads((out / "toy" / "mean.json").read_text())
    assert list(res["targets"]) == QUANTITIES and res["code"]["version"]
    assert res["targets"]["aod"]["r2"] == "pooled" and res["targets"]["distance"]["r2"] == "mean"
    n = res["targets"]["path_gain"]["n"]
    assert n["val_fit"] + n["val_select"] < sizes["val"] and n["test"] < sizes["test"]   # the coverage floor

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
    for readouts in ([], ["mean", "mean"], "mean", [""], ["../x"]):
        path.write_text(yaml.safe_dump({"data_dir": "d", "out_dir": "o", "features_dir": "f",
                                        "models": {"m": {"readouts": readouts}}}))
        with pytest.raises(ValueError, match="a list of one or more readouts"):
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


def test_model_names_are_plain(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"data_dir": "d", "out_dir": "o", "models": {"../m": {"raw": "complex"}}}))
    with pytest.raises(ValueError, match="plain name"):
        load_config(path)


def test_mean_power_of_the_raw_baselines(config, monkeypatch):
    cfg = load_config(config)
    cfg["models"] = {"raw": cfg["models"]["raw"]}
    cfg["quantities"] = ["path_gain"]
    data = Path(cfg["data_dir"])
    res = lambda: json.loads((R.output_dir(cfg) / "raw" / "raw.json").read_text())  # noqa: E731
    fp = lambda: {**res()["identity"]["features"], "mean_power_source": res()["mean_power_source"]}  # noqa: E731
    run(cfg, device="cpu")                                          # no manifest, the train channels
    assert np.isclose(fp()["mean_power"], R.mean_power(np.load(data / "train.npy")))
    assert fp()["mean_power_source"] == "train channels"
    (data / "manifest.yaml").write_text(yaml.safe_dump({"statistics": None}))   # a partition without statistics
    run(cfg, device="cpu")
    assert fp()["mean_power_source"] == "train channels"
    (data / "manifest.yaml").write_text(yaml.safe_dump({"statistics": {"mean_power": 2.5e-9}}))
    run(cfg, device="cpu")                                          # the manifest of the builder
    assert fp()["mean_power"] == 2.5e-9 and fp()["mean_power_source"] == "data_dir/manifest.yaml"
    cfg["mean_power"] = 1e-9
    run(cfg, device="cpu")                                          # the config wins
    assert fp()["mean_power"] == 1e-9 and fp()["mean_power_source"] == "config"
    cfg["mean_power"] = None
    (data / "manifest.yaml").write_text(yaml.safe_dump({"statistics": {"mean_power": 1e-9}}))
    monkeypatch.setattr(R, "probe_quantity", lambda *a: (_ for _ in ()).throw(AssertionError("probed again")))
    run(cfg, device="cpu")                                          # the same value from the manifest resumes
    monkeypatch.undo()
    cfg["mean_power"] = 0.0
    with pytest.raises(ValueError, match="positive and finite"):
        run(cfg, device="cpu")


def test_wide_path_gives_the_same_scores(config, monkeypatch):
    import chase.probes as P
    cfg = load_config(config)
    cfg["models"] = {"raw": cfg["models"]["raw"]}
    cfg["quantities"] = ["path_gain", "aod"]
    run(cfg, device="cpu")
    ref = json.loads((R.output_dir(cfg) / "raw" / "raw.json").read_text())["targets"]
    cfg["out_dir"] = cfg["out_dir"] + "_wide"
    monkeypatch.setattr(R, "LARGE", 1000)                           # every subset takes the wide path
    monkeypatch.setattr(P, "LARGE", 1000)
    run(cfg, device="cpu")
    new = json.loads((R.output_dir(cfg) / "raw" / "raw.json").read_text())["targets"]
    for q in cfg["quantities"]:
        for k in ("val", "test", "control"):
            assert abs(new[q]["linear"][k] - ref[q]["linear"][k]) < 1e-4
        assert new[q]["linear"]["alpha"] == ref[q]["linear"]["alpha"]
        for lv, m in new[q]["mlp"].items():
            assert abs(m["test_mean"] - ref[q]["mlp"][lv]["test_mean"]) < 1e-3


def test_default_coverage_floor(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"data_dir": "d", "out_dir": "o", "models": {"raw": {"raw": "complex"}}}))
    assert load_config(path)["min_path_gain_db"] == -150           # the coverage floor of the paper


def test_the_minimal_labels_of_docs_data_format(config):
    """docs/data_format.md: city, rx_position and path_gain_db, plus the fields of the probed quantities, suffice."""
    cfg = load_config(config)
    data = Path(cfg["data_dir"])
    for s in ("train", "val", "test"):
        lab = dict(np.load(data / f"{s}_labels.npz"))
        np.savez(data / f"{s}_labels.npz", **{k: lab[k] for k in ("city", "rx_position", "path_gain_db",
                                                                   "distance_m", "delay_spread_s")})
    cfg["quantities"] = ["distance", "path_gain", "delay_spread"]
    run(cfg, device="cpu")
    assert json.loads(write_report(cfg).read_text())["complete"]


def test_label_checks_before_any_probe(config, monkeypatch):
    cfg = load_config(config)
    data = Path(cfg["data_dir"])
    monkeypatch.setattr(R, "probe_quantity", lambda *a: (_ for _ in ()).throw(AssertionError("probed")))
    lab = dict(np.load(data / "val_labels.npz"))
    np.savez(data / "val_labels.npz", **{k: v for k, v in lab.items() if k != "k_factor_db"})
    with pytest.raises(ValueError, match=r"val_labels.npz has no \['k_factor_db'\]"):
        run(cfg, device="cpu")
    lab["rx_position"][0, 0] = np.nan
    np.savez(data / "val_labels.npz", **lab)
    with pytest.raises(ValueError, match="rx_position has NaN"):
        run(cfg, device="cpu")
    lab["rx_position"][0, 0] = 1.0
    np.savez(data / "val_labels.npz", **{**lab, "city": lab["city"].astype(str)})
    with pytest.raises(ValueError, match="city must be an integer"):
        run(cfg, device="cpu")
    np.savez(data / "val_labels.npz", **{k: v for k, v in lab.items() if k != "path_gain_db"})
    with pytest.raises(ValueError, match="path_gain_db"):                # read first by the coverage floor
        run(cfg, device="cpu")
