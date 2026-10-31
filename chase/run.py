"""The ChaSE run. For every model and readout of the config, every quantity is probed with the ridge probe and with
each MLP level and seed. The run keeps the test predictions for the report.

Inputs, in data_dir, for each set (train, val, test)
- <set>_labels.npz, the labels of the links (chase.quantities lists the fields), with city and rx_position [n, 3]
  for the tiles.
- <set>.npy, the channels [n, T, U, N, K] complex64 (OFDM symbols, user antennas, base-station antennas,
  subcarriers). The raw baselines read them, and chase.extract encodes them.

The representations of a model are in features_dir/<model>/<readout>/<set>.npy (features_dir/snr<X>/... with an
SNR), one row [D] float per link, in the order of the label files. chase.extract writes them from an encode function,
with a source.json that records how they were made. Extract them again when the data change.

Outputs, in out_dir (out_dir/snr<X> with an SNR), for each model and readout
- <model>/<readout>.json, the scores per quantity and probe, the controls, the config and the code state.
- <model>/<readout>_pred.npz, the test predictions of every probe.

The validation links are split in two halves of whole tiles. Half A fits the settings of each probe (the ridge
penalty, the MLP weight decay and stopping epoch). Half B scores each probe for the choice of the readout and the
probe in the report, so that the larger choice set of the MLPs does not inflate the score that decides between them
and the ridge probe.

A run resumes. It skips the quantities that a result file of the same settings and features already holds. A
resume with another config or code records them under "resumed_with".
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from chase.bootstrap import r2_mean, r2_pooled, tiles
from chase.probes import LARGE, mlp_probe, ridge_probe, shuffled
from chase.provenance import code_state, sha256, write_json
from chase.quantities import ANGLES, QUANTITIES, TARGETS
from chase.raw import mean_power, raw_set

SETS = ("train", "val", "test")
# The settings of the paper. A config overrides any of them.
DEFAULTS = {
    "features_dir": None,
    "min_path_gain_db": None,       # probe only the links with at least this path gain (dB), or all with None
    "mean_power": None,             # the raw baselines divide the channels by its square root, None takes that of train
    "quantities": None,             # None probes all 24
    "levels": {"mlp64": [64], "mlp256": [256], "mlp256x2": [256, 256]},
    "seeds": [0, 1, 2],
    "control_seed": 0,
    "controls": True,
    "val_split_seed": 0,
    "report": {"raw_baselines": None, "n_boot": 1000, "boot_seed": 0},   # None takes every raw model
}


def load_config(path: Path) -> dict:
    """The config of path, with the defaults for the keys it does not set."""
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("data_dir", "out_dir", "models"):
        if key not in cfg:
            raise ValueError(f"{path}: the config needs {key!r}")
    cfg = {**DEFAULTS, **cfg, "report": {**DEFAULTS["report"], **cfg.get("report", {})}}
    cfg["quantities"] = cfg["quantities"] or list(QUANTITIES)
    unknown = set(cfg["quantities"]) - set(QUANTITIES)
    if unknown:
        raise ValueError(f"not ChaSE quantities: {sorted(unknown)}")
    plain = lambda x: isinstance(x, str) and x and "/" not in x and x not in (".", "..")   # noqa: E731
    for name, spec in cfg["models"].items():
        if not plain(name):
            raise ValueError(f"model {name!r}: use a plain name, without '/'")
        if ("raw" in spec) == ("readouts" in spec):
            raise ValueError(f"model {name!r}: give either 'raw' (a raw baseline) or 'readouts'")
        if "readouts" in spec and cfg["features_dir"] is None:
            raise ValueError(f"model {name!r} needs 'features_dir'")
        r = spec.get("readouts")
        if "readouts" in spec and (not isinstance(r, list) or not r or len(set(r)) < len(r)
                                   or not all(plain(x) for x in r)):
            raise ValueError(f"model {name!r}: give a list of one or more readouts, each once, as plain names")
    if cfg["report"]["raw_baselines"] is None:
        cfg["report"]["raw_baselines"] = [m for m, spec in cfg["models"].items() if "raw" in spec]
    return cfg


def readouts_of(spec: dict) -> list:
    return ["raw"] if "raw" in spec else list(spec["readouts"])


def output_dir(cfg: dict, snr_db: float = None) -> Path:
    out = Path(cfg["out_dir"]).expanduser()
    return out if snr_db is None else out / f"snr{snr_db:g}"


def result_paths(out_dir: Path, name: str, readout: str):
    """(result JSON, prediction file) of one model and readout."""
    return out_dir / name / f"{readout}.json", out_dir / name / f"{readout}_pred.npz"


def load_labels(data_dir: Path) -> dict:
    return {s: dict(np.load(Path(data_dir) / f"{s}_labels.npz")) for s in SETS}


def run_settings(cfg: dict, snr_db: float = None) -> dict:
    """Everything except the features that the scores of a run depend on. The report checks it, and a run resumes
    only a result of the same settings."""
    data_dir = Path(cfg["data_dir"]).expanduser()
    return {**{k: cfg[k] for k in ("levels", "seeds", "control_seed", "controls", "val_split_seed",
                                   "min_path_gain_db")},
            "snr_db": snr_db, "labels_sha256": {s: sha256(data_dir / f"{s}_labels.npz") for s in SETS}}


def validation_halves(val_labels: dict, seed: int) -> np.ndarray:
    """True for the validation links of half B (half of the tiles, drawn with seed)."""
    vt = tiles(val_labels)
    n = int(vt.max()) + 1
    return np.isin(vt, np.random.default_rng(seed).permutation(n)[: n // 2])


def score_of(q: str) -> str:
    """The R^2 of a quantity. It is pooled over (sin, cos) for the angles, else the R^2 of its single output."""
    return "pooled" if q in ANGLES else "mean"


def features_path(cfg: dict, name: str, readout: str, snr_db: float = None) -> Path:
    """The directory of the feature files of one model and readout."""
    d = Path(cfg["features_dir"]).expanduser()
    return (d if snr_db is None else d / f"snr{snr_db:g}") / name / readout


def channel_stats(data_dir: Path) -> dict:
    """Size and modification time of each channel file (a checksum of the channels would be slow)."""
    stats = {s: (Path(data_dir) / f"{s}.npy").stat() for s in SETS}
    return {s: {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns} for s, st in stats.items()}


def _features(cfg: dict, name: str, readout: str, snr_db: float, power: float):
    """(fingerprint, loader) of the features of one model and readout. The loader gives {set: [n, D] array}."""
    spec = cfg["models"][name]
    data_dir = Path(cfg["data_dir"]).expanduser()
    if "raw" in spec:
        fp = {"raw": spec["raw"], "mean_power": power, "channels": channel_stats(data_dir)}
        return fp, lambda: {s: raw_set(np.load(data_dir / f"{s}.npy", mmap_mode="r"), spec["raw"], power, i, snr_db)
                            for i, s in enumerate(SETS)}
    d = features_path(cfg, name, readout, snr_db)
    fp = {"files": str(d), "sha256": {s: sha256(d / f"{s}.npy") for s in SETS}}
    return fp, lambda: {s: np.load(d / f"{s}.npy") for s in SETS}


def load_predictions(path: Path):
    """(arrays, identity) of a prediction file."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k != "__identity"}
        identity = json.loads(str(z["__identity"]))
    return arrays, identity


def _save_predictions(path: Path, arrays: dict, identity: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")                   # a crash never leaves a truncated file
    np.savez(tmp, __identity=np.array(json.dumps(identity)), **arrays)
    os.replace(tmp, path)


def _resume(out: Path, pf: Path, identity: dict):
    """(result, predictions) of an earlier run of the same identity, or None."""
    if not out.exists() or not pf.exists():
        return None
    res = json.loads(out.read_text())
    preds, pid = load_predictions(pf)
    if res.get("identity") != identity or pid != identity or any(f"{q}__true" not in preds for q in res["targets"]):
        return None
    return res, preds


def probe_quantity(q, x, labels, covered, half_b, cfg, dev):
    """(result entry, test predictions) of one quantity, with the ridge probe and every MLP level and seed."""
    wide = x["train"].numel() > LARGE              # wide inputs stay on the CPU, with one device copy per probe
    c = (lambda v: v.to(dev, copy=True)) if wide else (lambda v: v)
    sel = {}
    for s in SETS:
        y, ok = TARGETS[q](labels[s])
        sel[s] = (np.flatnonzero(ok & covered[s]), y)
    va = sel["val"][0]
    va_a, va_b, te = va[~half_b[va]], va[half_b[va]], sel["test"][0]
    pick = lambda s, i: (x[s][torch.from_numpy(i).to(x[s].device)],   # noqa: E731
                         torch.from_numpy(sel[s][1][i]).to(dev))
    xtr, ytr = pick("train", sel["train"][0])
    xa, ya = pick("val", va_a)
    xb, yb = pick("val", va_b)
    xt, yt = pick("test", te)
    for part, y in (("train", ytr), ("validation half A", ya), ("validation half B", yb), ("test", yt)):
        if len(y) < 2 or not bool((y != y[0]).any()):
            raise ValueError(f"{q} does not vary on the {part} links ({len(y)} links), so no R^2 is defined. "
                             "Remove it from 'quantities' in the config.")
    xbt, ybt = torch.cat([xb, xt]), torch.cat([yb, yt])   # scored together, split below
    nb = len(va_b)
    pooled = score_of(q) == "pooled"
    r2f = r2_pooled if pooled else r2_mean

    def scores(p):                                 # (score on validation half B, test score) of predictions
        return r2f(sel["val"][1][va_b], p[:nb]), r2f(sel["test"][1][te], p[nb:])

    seed_c = cfg["control_seed"]
    lin = ridge_probe(c(xtr), ytr, c(xa), ya, c(xbt), ybt, return_pred=True, pooled=pooled)
    pl = lin.pop("pred_test")
    lin_val, lin_test = scores(pl)
    lin_ctl = ridge_probe(c(xtr), shuffled(ytr, seed_c), c(xa), ya, c(xt), yt,
                          pooled=pooled)["test"] if cfg["controls"] else None
    preds = {f"{q}__idx": te, f"{q}__true": sel["test"][1][te].astype(np.float32), f"{q}__linear": pl[nb:]}
    entry = {"r2": score_of(q),
             "n": {"train": int(len(sel["train"][0])), "val_fit": int(len(va_a)), "val_select": int(nb),
                   "test": int(len(te))},
             "linear": {"val": lin_val, "test": lin_test, "val_fit": lin["val"], "alpha": lin["alpha"],
                        "control": lin_ctl},
             "mlp": {}}
    for lv, widths in cfg["levels"].items():
        runs = {}
        for sd in cfg["seeds"]:
            r = mlp_probe((c(xtr), ytr, c(xa), ya, c(xbt), ybt), widths, sd, return_pred=True, pooled=pooled)
            p = r.pop("pred_test")
            v, t = scores(p)
            runs[str(sd)] = {"val": v, "test": t, "val_fit": r["val"], "weight_decay": r["weight_decay"]}
            preds[f"{q}__{lv}_s{sd}"] = p[nb:]
        control = None
        if cfg["controls"]:                        # shuffled train and validation targets, first seed
            ctl = (c(xtr), shuffled(ytr, seed_c), c(xa), shuffled(ya, seed_c + 1), c(xt), yt)
            control = mlp_probe(ctl, widths, cfg["seeds"][0], pooled=pooled)["test"]
        entry["mlp"][lv] = {"seeds": runs, "val_mean": float(np.mean([r["val"] for r in runs.values()])),
                            "test_mean": float(np.mean([r["test"] for r in runs.values()])),
                            "control": control}
    return entry, preds


def run(cfg: dict, config_file: str = None, models=None, readouts=None, snr_db: float = None, device=None) -> None:
    """Probe every quantity of the config for the given models and readouts (default all)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    unknown = [m for m in (models or []) if m not in cfg["models"]]
    if unknown:
        raise ValueError(f"not models of the config: {unknown}")
    labels = load_labels(Path(cfg["data_dir"]).expanduser())
    floor = cfg["min_path_gain_db"]
    covered = {s: np.ones(len(labels[s]["path_gain_db"]), bool) if floor is None
               else labels[s]["path_gain_db"] >= floor for s in SETS}
    half_b = validation_halves(labels["val"], cfg["val_split_seed"])
    settings, state = run_settings(cfg, snr_db), code_state()
    out_dir = output_dir(cfg, snr_db)
    out_dir.mkdir(parents=True, exist_ok=True)
    power = cfg["mean_power"]
    if power is None and any("raw" in cfg["models"][m] for m in models or cfg["models"]):
        power = mean_power(np.load(Path(cfg["data_dir"]).expanduser() / "train.npy", mmap_mode="r"))
    power = None if power is None else float(power)
    for name in models or cfg["models"]:
        for readout in readouts_of(cfg["models"][name]):
            if readouts and readout not in readouts:
                continue
            out, pf = result_paths(out_dir, name, readout)
            out.parent.mkdir(exist_ok=True)
            fp, load = _features(cfg, name, readout, snr_db, power)
            identity = {"settings": settings, "features": fp}
            old = _resume(out, pf, identity)
            if old is None:
                res, preds = {"model": name, "readout": readout, "identity": identity, "config": cfg,
                              "config_file": config_file, "code": state,
                              "created": datetime.now().isoformat(timespec="seconds"), "targets": {}}, {}
            else:
                res, preds = old
                resume = {"code": state, "config": cfg, "config_file": config_file}
                first = {k: res[k] for k in resume}
                if resume != first and resume not in res.get("resumed_with", []):
                    res.setdefault("resumed_with", []).append(resume)
            todo = [q for q in cfg["quantities"] if q not in res["targets"]]
            if not todo:
                print(f"{name}/{readout}: complete, skipped", flush=True)
                continue
            x = {s: torch.from_numpy(np.asarray(v, dtype=np.float32)) for s, v in load().items()}
            for s in SETS:
                if len(x[s]) != len(covered[s]):
                    raise ValueError(f"{name}/{readout}/{s}: {len(x[s])} rows, the labels have {len(covered[s])}")
                if not bool(torch.isfinite(x[s]).all()):
                    raise ValueError(f"{name}/{readout}/{s}: the features have NaN or inf values")
            if x["train"].numel() <= LARGE:
                x = {s: v.to(dev) for s, v in x.items()}
            for q in todo:
                entry, p = probe_quantity(q, x, labels, covered, half_b, cfg, dev)
                res["targets"][q] = entry
                preds.update(p)
                _save_predictions(pf, preds, identity)
                write_json(out, res)
                best = max(entry["mlp"].values(), key=lambda v: v["val_mean"], default=None)
                print(f"{name}/{readout} {q:18s} linear {entry['linear']['test']:.3f}"
                      + (f" | best MLP {best['test_mean']:.3f}" if best else ""), flush=True)
