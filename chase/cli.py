"""The command-line commands of ChaSE, which pip installs from [project.scripts] in pyproject.toml. Each command prints its arguments with --help."""

import argparse
from pathlib import Path

import yaml

from chase.data import build_cell
from chase.partition import make_partition
from chase.report import write_report
from chase.run import load_config, run


def make_partition_main(argv=None):
    ap = argparse.ArgumentParser(prog="chase-make-partition",
                                 description="Split the links of CSIGen runs into the train, val and test parts")
    ap.add_argument("split", type=Path, help="the split config (see configs/split_example.yaml)")
    ap.add_argument("runs", type=Path, nargs="+",
                    help="the CSIGen runs, one folder per city or town, all in one folder")
    ap.add_argument("out_dir", type=Path, help="the partition folder, for chase-build-data")
    a = ap.parse_args(argv)
    make_partition([r.expanduser() for r in a.runs], yaml.safe_load(a.split.expanduser().read_text()),
                   a.out_dir.expanduser())


def build_data_main(argv=None):
    ap = argparse.ArgumentParser(prog="chase-build-data",
                                 description="Write the sets of an evaluation cell from CSIGen runs and a partition")
    ap.add_argument("runs_dir", type=Path, help="the folder that holds the CSIGen runs of the partition")
    ap.add_argument("partition_dir", type=Path, help="the folder with partition.npz")
    ap.add_argument("out_dir", type=Path, help="the data_dir of the ChaSE config")
    a = ap.parse_args(argv)
    m = build_cell(a.runs_dir.expanduser(), a.partition_dir.expanduser(), a.out_dir.expanduser())
    print(m["counts"])


def run_main(argv=None):
    ap = argparse.ArgumentParser(prog="chase-run",
                                 description="Probe every model, readout and quantity of a ChaSE config")
    ap.add_argument("config", type=Path)
    ap.add_argument("--models", nargs="+", default=None, help="only these models of the config")
    ap.add_argument("--readouts", nargs="+", default=None, help="only these readouts")
    ap.add_argument("--snr-db", type=float, default=None,
                    help="probe the channels with AWGN at this SNR (features from <features_dir>/snr<X>)")
    ap.add_argument("--device", default=None, help="torch device (default cuda if available, else cpu)")
    a = ap.parse_args(argv)
    run(load_config(a.config.expanduser()), str(a.config), a.models, a.readouts, a.snr_db, a.device)


def report_main(argv=None):
    ap = argparse.ArgumentParser(prog="chase-report", description="Write the ChaSE report of a config")
    ap.add_argument("config", type=Path)
    ap.add_argument("--snr-db", type=float, default=None, help="the report of the runs at this SNR")
    a = ap.parse_args(argv)
    print(write_report(load_config(a.config.expanduser()), str(a.config), a.snr_db))
