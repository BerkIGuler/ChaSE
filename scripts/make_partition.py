"""Partition CSIGen runs into the train, val and test parts of ChaSE, with one held-out test site per run.

    python scripts/make_partition.py configs/split_example.yaml <runs_dir>/<run> [<runs_dir>/<run> ...] <out_dir>

Then build the sets of the cell with scripts/build_data.py <runs_dir> <out_dir> <data_dir>. The partition reads only
the metadata of the runs, so it takes seconds.
"""

import argparse
from pathlib import Path

import yaml

from chase.partition import make_partition


def main():
    ap = argparse.ArgumentParser(description="Partition CSIGen runs into the parts of ChaSE")
    ap.add_argument("split", type=Path, help="the split config (see configs/split_example.yaml)")
    ap.add_argument("runs", type=Path, nargs="+", help="the CSIGen runs, one folder per city or town")
    ap.add_argument("out_dir", type=Path, help="the partition folder, for scripts/build_data.py")
    a = ap.parse_args()
    make_partition([r.expanduser() for r in a.runs], yaml.safe_load(a.split.read_text()), a.out_dir.expanduser())


if __name__ == "__main__":
    main()
