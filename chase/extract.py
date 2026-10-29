"""Write the feature files of a model from a function that encodes channels.

The user gives a function encode(h). Its input h is a batch of channels [B, T, N, K] complex64 on the CPU, in the
units of the data. It does the preprocessing, the encoding and the readouts of its model. It returns
{readout: [B, D] features} as torch tensors or numpy arrays, with at least the readouts of the config.
The same channel must always give the same features.

extract() calls encode on the channels of train, val and test in batches of chase.raw.NOISE_BATCH. With an SNR, it
first adds the same noise that the raw baselines get, so that the lost share compares the model and the raw channel
on the same noisy channels. It writes <features_dir>[/snr<X>]/<model>/<readout>/<set>.npy (float32) and a
source.json with the provenance.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from chase.provenance import code_state, sha256, write_json
from chase.raw import channel_batches
from chase.run import SETS, channel_stats, features_path


def _as_numpy(v, readout: str) -> np.ndarray:
    if torch.is_complex(v) if isinstance(v, torch.Tensor) else np.iscomplexobj(v):
        raise ValueError(f"readout {readout!r}: encode must return real features (split complex ones into real and "
                         "imaginary parts)")
    a = v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"readout {readout!r}: encode must return finite [B, D] features, got shape {a.shape}")
    return a


def _encode_set(encode, channels, set_index, snr_db, readouts, tmp_of) -> None:
    """Encode one set into a temporary file per readout (tmp_of(readout))."""
    out = {}
    try:
        for start, h in channel_batches(channels, set_index, snr_db):
            feats = encode(h.to(torch.complex64))
            if not isinstance(feats, dict) or not set(readouts) <= set(feats):
                got = sorted(feats) if isinstance(feats, dict) else type(feats).__name__
                raise ValueError(f"encode must return a dict with the readouts {readouts}, got {got}")
            for r in readouts:
                a = _as_numpy(feats[r], r)
                if len(a) != len(h):
                    raise ValueError(f"readout {r!r}: {len(a)} rows for a batch of {len(h)} channels")
                if r not in out:                          # the first batch sets D
                    out[r] = np.lib.format.open_memmap(tmp_of(r), mode="w+", dtype=np.float32,
                                                       shape=(len(channels), a.shape[1]))
                elif a.shape[1] != out[r].shape[1]:
                    raise ValueError(f"readout {r!r}: D changed from {out[r].shape[1]} to {a.shape[1]}")
                out[r][start:start + len(a)] = a
        for v in out.values():
            v.flush()
    finally:
        out.clear()                                       # close the memory maps before the files move


def extract(encode, cfg: dict, model: str, snr_db: float = None) -> Path:
    """Write the feature files of model (a model of the config with 'readouts') from encode. Returns the directory
    of the model's features. The new files replace the old ones only after every set is encoded, so that a failure of
    encode leaves the old features as they were. Two extractions of the same model and SNR must not run at once."""
    spec = cfg["models"].get(model)
    if spec is None or "readouts" not in spec:
        raise ValueError(f"{model!r} is not a model of the config with 'readouts'")
    readouts = list(spec["readouts"])
    data_dir = Path(cfg["data_dir"]).expanduser()
    for s in SETS:                                        # check the data before any encoding
        n = len(np.load(data_dir / f"{s}.npy", mmap_mode="r"))
        with np.load(data_dir / f"{s}_labels.npz") as z:
            n_labels = len(z["path_gain_db"])
        if n == 0 or n != n_labels:
            raise ValueError(f"{s}: {n} channels and {n_labels} labels")
    source = {"model": model, "snr_db": snr_db, "data_dir": str(data_dir), "config": cfg,
              "labels_sha256": {s: sha256(data_dir / f"{s}_labels.npz") for s in SETS},
              "channels": channel_stats(data_dir), "code": code_state()}   # the data at the start of the encoding
    json.dumps(source)                                    # fails here, before any encoding
    dirs = {r: features_path(cfg, model, r, snr_db) for r in readouts}
    tmp = {(r, s): dirs[r] / f"{s}.{os.getpid()}.tmp.npy" for r in readouts for s in SETS}
    new = set()                                           # the directories that this extraction creates
    for d in dirs.values():
        new.update(p for p in (d, *d.parents) if not p.exists())
        d.mkdir(parents=True, exist_ok=True)
    done = False
    try:
        with torch.no_grad():
            for i, s in enumerate(SETS):
                channels = np.load(data_dir / f"{s}.npy", mmap_mode="r")
                _encode_set(encode, channels, i, snr_db, readouts, lambda r: tmp[(r, s)])
                print(f"{model} {s}: {len(channels)} channels, readouts {readouts}", flush=True)
        source["created"] = datetime.now().isoformat(timespec="seconds")
        for d in dirs.values():                           # no old provenance next to new files
            (d / "source.json").unlink(missing_ok=True)
        for (r, s), f in tmp.items():
            os.replace(f, dirs[r] / f"{s}.npy")
        for d in dirs.values():
            write_json(d / "source.json", source)
        done = True
    finally:
        for f in tmp.values():
            f.unlink(missing_ok=True)
        if not done:                                      # no empty directories of a failed first extraction
            for d in sorted(new, key=lambda p: len(p.parts), reverse=True):
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
    return dirs[readouts[0]].parent
