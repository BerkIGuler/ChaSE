"""The run and the report, end to end, on a small synthetic data set."""

import json

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
