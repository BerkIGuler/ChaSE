"""Write the ChaSE report of a config from the results of scripts/run_chase.py.

    python scripts/chase_report.py configs/example.yaml [--snr-db 10]

Writes report.json and report.md to the output directory of the run.
"""

import argparse
from pathlib import Path

from chase.report import write_report
from chase.run import load_config


def main():
    ap = argparse.ArgumentParser(description="ChaSE report")
    ap.add_argument("config", type=Path)
    ap.add_argument("--snr-db", type=float, default=None, help="the report of the runs at this SNR")
    a = ap.parse_args()
    print(write_report(load_config(a.config), str(a.config), a.snr_db))


if __name__ == "__main__":
    main()
