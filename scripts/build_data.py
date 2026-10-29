"""Build the ChaSE sets of an evaluation cell from CSIGen runs and a partition.

    python scripts/build_data.py <runs_dir> <partition_dir> <out_dir>

For example, with the ChaSE data set, runs/urban, partitions/urban and the data_dir of a config. The channels are
rebuilt from the saved paths on the CPU, about 10 to 20 seconds per CSIGen file.
"""

import argparse
from pathlib import Path

from chase.data import build_cell


def main():
    ap = argparse.ArgumentParser(description="ChaSE sets of an evaluation cell from CSIGen runs and a partition")
    ap.add_argument("runs_dir", type=Path, help="folder with one CSIGen run per city")
    ap.add_argument("partition_dir", type=Path, help="folder with partition.npz and manifest.yaml")
    ap.add_argument("out_dir", type=Path, help="the data_dir of the ChaSE config")
    a = ap.parse_args()
    m = build_cell(a.runs_dir.expanduser(), a.partition_dir.expanduser(), a.out_dir.expanduser())
    print(m["counts"])


if __name__ == "__main__":
    main()
