"""The commands of ChaSE (chase/cli.py), as pip installs them from pyproject.toml."""

import importlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml

from chase import cli

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_every_command_points_to_a_function():
    scripts = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]
    assert set(scripts) == {"chase-make-partition", "chase-build-data", "chase-run", "chase-report"}
    for target in scripts.values():
        module, name = target.split(":")
        assert callable(getattr(importlib.import_module(module), name))


def test_run_and_report_commands(config, capsys):
    cfg = yaml.safe_load(config.read_text())
    cfg["quantities"] = ["path_gain"]
    cfg["models"] = {"raw": cfg["models"]["raw"]}
    config.write_text(yaml.safe_dump(cfg))
    cli.run_main([str(config), "--device", "cpu"])
    cli.report_main([str(config)])
    report = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert json.loads(report.read_text())["complete"]


def test_partition_and_build_commands(tmp_path, csigen_run, split):
    rng = np.random.default_rng(0)
    for c in ("a_1", "b_1"):
        csigen_run(tmp_path / "runs" / c, rng)
    path = tmp_path / "split.yaml"
    path.write_text(yaml.safe_dump(split))
    cli.make_partition_main([str(path), str(tmp_path / "runs" / "a_1"), str(tmp_path / "runs" / "b_1"),
                             str(tmp_path / "part")])
    cli.build_data_main([str(tmp_path / "runs"), str(tmp_path / "part"), str(tmp_path / "data")])
    assert (tmp_path / "data" / "manifest.yaml").exists()


def test_help(capsys):
    with pytest.raises(SystemExit) as e:
        cli.run_main(["--help"])
    assert e.value.code == 0 and "chase-run" in capsys.readouterr().out


def test_models_need_a_name(config):
    with pytest.raises(SystemExit):
        cli.run_main([str(config), "--models"])
