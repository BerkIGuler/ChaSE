"""Probe every quantity of a ChaSE config for its models and readouts.

    python scripts/run_chase.py configs/example.yaml [--models raw dummy] [--readouts mean] [--snr-db 10]
"""

import argparse
from pathlib import Path

from chase.run import load_config, run


def main():
    ap = argparse.ArgumentParser(description="ChaSE probes of every model, readout and quantity of a config")
    ap.add_argument("config", type=Path)
    ap.add_argument("--models", nargs="*", default=None, help="only these models of the config")
    ap.add_argument("--readouts", nargs="*", default=None, help="only these readouts")
    ap.add_argument("--snr-db", type=float, default=None,
                    help="probe the channels with AWGN at this SNR (features from <features_dir>/snr<X>)")
    ap.add_argument("--device", default=None, help="torch device (default cuda if available, else cpu)")
    a = ap.parse_args()
    run(load_config(a.config), str(a.config), a.models, a.readouts, a.snr_db, a.device)


if __name__ == "__main__":
    main()
