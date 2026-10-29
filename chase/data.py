"""Build the ChaSE sets of an evaluation cell from CSIGen runs and a partition.

Inputs
- runs_dir/<run>/channel_tx_<i>.npz, CSIGen runs with saved paths and one RX antenna, in the format of CSIGen 0.3.4
  to 0.3.6 (the ChaSE data set). The channels h may be left out, since chase.cfr rebuilds them from the paths.
- partition_dir/partition.npz, with one entry per link of the runs. It gives the run (cities, city), the file (tx),
  the row in the file (row), the user, the part (PARTS) and whether the link goes to the serving site (own_site).
- partition_dir/manifest.yaml (optional), whose "statistics" (the mean power of the pretraining channels) is copied.

The train, val and test sets are the own-site links of the parts of the same name, in partition order. For each set,
out_dir gets <set>.npy, the channels [n, T, N, K] (OFDM symbols, antennas, subcarriers) complex64 as CSIGen computed
them, and <set>_labels.npz. The labels are city, tx, row and user from the partition, and from the metadata of the
CSIGen file los_binary, rx_position, rx_velocity, rx_orientation, tx_orientation (each [n, 3]) and every path_stats
field. manifest.yaml is written last, so that it marks a complete build.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from chase.cfr import load_cfr
from chase.provenance import code_state, sha256
from chase.run import SETS

PARTS = ("train", "val", "test", "excluded", "guard")


def file_labels(meta: dict) -> dict:
    """The labels of every channel of a CSIGen file, from its metadata."""
    n = int(meta["num_valid_channels"])
    lab = {"los_binary": np.asarray(meta["los_binary"]).astype(np.int8),
           "rx_position": np.asarray(meta["rx_positions"], dtype=np.float32).reshape(n, 3),
           "rx_velocity": np.asarray(meta["rx_velocities"], dtype=np.float32).reshape(n, 3),
           "rx_orientation": np.asarray(meta["rx_orientations"], dtype=np.float32).reshape(n, 3),
           "tx_orientation": np.tile(np.asarray(meta["tx_orientation"], dtype=np.float32).reshape(1, 3), (n, 1))}
    for k in meta["path_stats_fields"]:
        lab[k] = np.asarray(meta["path_stats"][k], dtype=np.float32)
    bad = [k for k, v in lab.items() if len(v) != n]
    if bad:
        raise ValueError(f"labels {bad} do not have {n} channels")
    return lab


def build_cell(runs_dir: Path, partition_dir: Path, out_dir: Path) -> dict:
    """Write the train, val and test sets of a cell to out_dir. Returns the manifest."""
    runs_dir, partition_dir, out_dir = Path(runs_dir), Path(partition_dir), Path(out_dir)
    if (out_dir / "manifest.yaml").exists():
        raise FileExistsError(f"{out_dir} has a complete build already. Remove it first.")
    p = dict(np.load(partition_dir / "partition.npz"))
    cities = [str(c) for c in p["cities"]]
    idx = {s: np.flatnonzero((p["part"] == PARTS.index(s)) & p["own_site"]) for s in SETS}
    empty = [s for s in SETS if not len(idx[s])]
    if empty:
        raise ValueError(f"the partition has no own-site links in {empty}")
    out_dir.mkdir(parents=True, exist_ok=True)
    h, lab = {}, {s: {k: p[k][idx[s]] for k in ("city", "tx", "row", "user")} for s in SETS}
    filled = {s: np.zeros(len(idx[s]), bool) for s in SETS}
    fields = None                                             # the label fields, from the first file
    for ci, city in enumerate(cities):
        for tx in np.unique(p["tx"][p["city"] == ci]):
            pick = {s: np.flatnonzero((lab[s]["city"] == ci) & (lab[s]["tx"] == tx)) for s in SETS}
            if not any(len(v) for v in pick.values()):
                continue
            f = runs_dir / city / f"channel_tx_{tx}.npz"
            with np.load(f, allow_pickle=True) as z:
                meta = z["metadata"].item()
            if int(meta["tx_idx"]) != int(tx):
                raise ValueError(f"{f}: the file is of tx {meta['tx_idx']}, the partition expects tx {tx}")
            fl = file_labels(meta)
            fields = sorted(fl) if fields is None else fields
            if sorted(fl) != fields:
                raise ValueError(f"{f}: the label fields {sorted(fl)} differ from those of the first file {fields}")
            full = load_cfr(f)
            if full.shape[1] != 1 or len(full) != int(meta["num_valid_channels"]):
                raise ValueError(f"{f}: {full.shape[1]} RX antennas and {len(full)} channels, ChaSE needs 1 RX "
                                 f"antenna and {meta['num_valid_channels']} channels")
            cfr = full[:, 0, 0].transpose(0, 2, 1, 3)               # [n, T, N, K]
            for s, pos in pick.items():
                if not len(pos):
                    continue
                rows = lab[s]["row"][pos]
                if rows.min() < 0 or rows.max() >= len(cfr):
                    raise ValueError(f"{f}: the partition asks for rows {rows.min()} to {rows.max()}, the file has "
                                     f"{len(cfr)}")
                if s not in h:
                    h[s] = np.lib.format.open_memmap(out_dir / f"{s}.npy", mode="w+", dtype=np.complex64,
                                                     shape=(len(idx[s]), *cfr.shape[1:]))
                h[s][pos] = cfr[rows]
                for k, v in fl.items():
                    lab[s].setdefault(k, np.zeros((len(idx[s]), *v.shape[1:]), v.dtype))[pos] = v[rows]
                filled[s][pos] = True
            print(f"{city} tx {tx}: {', '.join(f'{s} {len(v)}' for s, v in pick.items())}", flush=True)
    missing = {s: int((~v).sum()) for s, v in filled.items() if not v.all()}
    if missing:
        raise ValueError(f"no CSIGen file gave these links (count per set) {missing}")
    for s in SETS:
        h[s].flush()
        np.savez(out_dir / f"{s}_labels.npz", **lab[s])
    pm = partition_dir / "manifest.yaml"
    manifest = {
        "sets": {s: f"part {s}, own-site links, in partition order" for s in SETS},
        "counts": {s: int(len(idx[s])) for s in SETS},
        "channels": "[n, OFDM symbol, TX antenna, subcarrier] complex64, unnormalized",
        "partition_sha256": sha256(partition_dir / "partition.npz"),
        "statistics": yaml.safe_load(pm.read_text()).get("statistics") if pm.exists() else None,
        "labels": ["city", "tx", "row", "user", *fields],
        "labels_sha256": {s: sha256(out_dir / f"{s}_labels.npz") for s in SETS},
        "runs": {c: {"generation_yaml_sha256": sha256(runs_dir / c / "generation.yaml")
                     if (runs_dir / c / "generation.yaml").exists() else None} for c in cities},
        "code": code_state(), "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest
