"""The partition of CSIGen runs into the parts of ChaSE, from the metadata of the runs only.

The split holds out one site of each run (each city or town) as a test cell. Each link (a site-user pair of a run)
gets one part.

- test. The own-site links of the held-out site. A link is own-site when it goes to the site that the user was
  placed for.
- excluded. The other links of the held-out site and all links of its users. They are in no part.
- val. All links of the users in whole square blocks of block_m, in random order, until at least fraction of the
  users of the other sites are in them.
- guard. All links of the users within guard_m (horizontal) of a validation user. They are in no part.
- train. All other links.

ChaSE probes the own-site links of train, val and test. The split config gives the held-out site of each run, the
number of sectors per site and the validation settings (see configs/split_example.yaml). The seed of a run is the
seed of the config plus the place of the run in held_out_site.

Output, in out_dir
- partition.npz, one entry per link of the runs: cities (the run names), parts (PARTS), city, tx, row, user, part and
  own_site.
- manifest.yaml, the split config, the counts per run and part, the runs with the checksum of their generation.yaml,
  and the code state.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from chase.provenance import code_state, sha256

PARTS = ("train", "val", "test", "excluded", "guard")
TRAIN, VAL, TEST, EXCLUDED, GUARD = range(len(PARTS))
GUARD_ELEMENTS = 2 * 10 ** 7            # pairs of users per block of the guard distance computation


def tx_files(run: Path) -> List[Path]:
    """The channel_tx_<i>.npz files of a run, in the order of i."""
    return sorted(Path(run).glob("channel_tx_*.npz"), key=lambda f: int(f.stem.split("_")[-1]))


def val_users(users: np.ndarray, pos: np.ndarray, fraction: float, block_m: float, guard_m: float,
              rng: np.random.Generator):
    """(validation users, guard users) of one run. users [U] are unique ids, pos [U, 2] their positions (m). Whole
    blocks of block_m go to validation, in random order, until at least fraction of the users. At least one block
    stays out of validation. The guard users are the other users within guard_m of a validation user."""
    if not 0 < fraction < 1:
        raise ValueError(f"the validation fraction must be between 0 and 1, not {fraction}")
    cell = np.floor((pos - pos.min(axis=0)) / block_m).astype(np.int64)
    blocks, block_of = np.unique(cell, axis=0, return_inverse=True)
    block_of = block_of.reshape(-1)
    counts = np.bincount(block_of)
    chosen, total = [], 0
    for b in rng.permutation(len(blocks)):
        if total >= fraction * len(users) or len(chosen) == len(blocks) - 1:
            break
        chosen.append(b)
        total += counts[b]
    if not chosen:
        raise ValueError(f"all {len(users)} users lie in one block of {block_m} m, so no block is left for validation. "
                         "Use a smaller block_m.")
    is_val = np.isin(block_of, chosen)
    if guard_m <= 0:
        return users[is_val], users[:0]
    vpos, rest = pos[is_val], pos[~is_val]
    near = np.zeros(len(rest), bool)
    chunk = max(1, GUARD_ELEMENTS // len(vpos))
    for i in range(0, len(rest), chunk):
        d = np.sqrt(((rest[i:i + chunk, None, :] - vpos[None]) ** 2).sum(-1)).min(1)
        near[i:i + chunk] = d < guard_m
    return users[is_val], users[~is_val][near]


def run_partition(run: Path, held_out_site: int, num_sectors: int, val: dict, seed: int) -> Dict[str, np.ndarray]:
    """The part of every link of one run, from the metadata of its files."""
    cols = {k: [] for k in ("tx", "row", "user", "serving_site", "site")}
    positions = {}
    for f in tx_files(run):
        with np.load(f, allow_pickle=True) as z:
            meta = z["metadata"].item()
        if int(meta["num_sectors"]) != num_sectors:
            raise ValueError(f"{f}: {meta['num_sectors']} sectors per site, the split config gives {num_sectors}")
        n = int(meta["num_valid_channels"])
        user = np.array([int(str(nm).split("_")[1]) for nm in np.asarray(meta["rx_names"])], dtype=np.int64)
        pos = np.asarray(meta["rx_positions"], dtype=np.float64).reshape(n, 3)[:, :2]
        positions.update(zip(user.tolist(), map(tuple, pos)))
        cols["tx"].append(np.full(n, int(meta["tx_idx"])))
        cols["row"].append(np.arange(n))
        cols["user"].append(user)
        cols["serving_site"].append(np.asarray(meta["rx_serving_tx"]) // num_sectors)
        cols["site"].append(np.full(n, int(meta["tx_idx"]) // num_sectors))
    if not cols["tx"]:
        raise ValueError(f"{run} has no channel_tx_<i>.npz files")
    c = {k: np.concatenate(v) for k, v in cols.items()}
    own = c["site"] == c["serving_site"]
    new_cell = (c["site"] == held_out_site) | (c["serving_site"] == held_out_site)
    if not (new_cell & own).any():
        raise ValueError(f"{run} has no links of the held-out site {held_out_site}")
    rest_users = np.unique(c["user"][~new_cell])
    rest_pos = np.array([positions[u] for u in rest_users])
    v_users, g_users = val_users(rest_users, rest_pos, float(val["fraction"]), float(val["block_m"]),
                                 float(val["guard_m"]), np.random.default_rng(seed))
    part = np.full(len(own), TRAIN, dtype=np.int8)
    part[np.isin(c["user"], g_users)] = GUARD
    part[np.isin(c["user"], v_users)] = VAL
    part[new_cell] = EXCLUDED
    part[new_cell & own & (c["site"] == held_out_site)] = TEST
    return {"tx": c["tx"], "row": c["row"], "user": c["user"], "part": part, "own_site": own}


def make_partition(runs: List[Path], split: dict, out_dir: Path) -> dict:
    """Write partition.npz and manifest.yaml of the runs to out_dir. Returns the manifest."""
    runs, out_dir = [Path(r) for r in runs], Path(out_dir)
    if len({r.name for r in runs}) < len(runs) or len({r.resolve().parent for r in runs}) > 1:
        raise ValueError("give each run once, all in one folder (chase-build-data reads them from there)")
    if (out_dir / "manifest.yaml").exists():
        raise FileExistsError(f"{out_dir} has a partition already. Remove it first.")
    held = {str(k): int(v) for k, v in split["held_out_site"].items()}
    order = list(held)
    parts, counts, sources = [], {}, {}
    for ci, run in enumerate(runs):
        if run.name not in held:
            raise ValueError(f"{run.name} has no held-out site in the split config")
        # the seed follows the place of the run in the split config, not in the list of runs
        p = run_partition(run, held[run.name], int(split["num_sectors"]), split["val"],
                          int(split["val"]["seed"]) + order.index(run.name))
        p["city"] = np.full(len(p["part"]), ci)
        parts.append(p)
        gen = run / "generation.yaml"
        sources[run.name] = {"generation_yaml_sha256": sha256(gen) if gen.exists() else None}
        counts[run.name] = {"held_out_site": held[run.name], "links": int(len(p["part"])),
                            **{name: int((p["part"] == code).sum()) for code, name in enumerate(PARTS)},
                            **{f"{name}_own_site": int(((p["part"] == code) & p["own_site"]).sum())
                               for code, name in enumerate(PARTS[:3])}}
        print(f"{run.name}: " + ", ".join(f"{k} {v}" for k, v in counts[run.name].items()), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "partition.npz", cities=np.array([r.name for r in runs]), parts=np.array(PARTS),
             **{k: np.concatenate([p[k] for p in parts]) for k in ("city", "tx", "row", "user", "part", "own_site")})
    manifest = {"split": split, "counts": counts, "sources": sources,
                "partition_npz_sha256": sha256(out_dir / "partition.npz"), "code": code_state(),
                "created": datetime.now().isoformat(timespec="seconds")}
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest
