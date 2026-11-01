"""Write the feature files of a model from a function that encodes channels.

The user gives a function encode(h). Its input h is a batch of channels [B, T, U, N, K] complex64 (OFDM symbols, user
antennas, base-station antennas, subcarriers) on the CPU, in the units of the data. It does the preprocessing, the
encoding and the readouts of its model. It returns {readout: [B, D] features} as torch tensors or numpy arrays, with at
least the readouts of the config. The same channel must always give the same features, so encode must keep no state
between calls.

extract() calls encode on the channels of train, val and test in batches of chase.raw.NOISE_BATCH. With an SNR, it
adds to each batch the same noise that the raw baselines get, so that the lost share compares the model and the raw
channel on the same noisy channels. Before the extraction, it checks on the first batch of the train set that encode
gives the same vector for the same channel (check_encode). It writes <features_dir>[/snr<X>]/<model>/<readout>/<set>.npy
(float32) and a source.json with the provenance.
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

CHECK_CHANNELS = 8                       # channels of the first batch that encode also gets alone
# Largest difference of the features of the same channel, relative to the largest feature. Measured on an RTX 3060
# with transformer encoders, a correct model differs by at most 1.4e-7 in float32 and 3.1e-4 in bfloat16 (other batch
# sizes), while dropout of 0.01 in training mode differs by at least 1.6e-2 and BatchNorm in training mode by 3.5e-1.
CHECK_RTOL = 3e-3


def _as_numpy(v, readout: str) -> np.ndarray:
    if torch.is_complex(v) if isinstance(v, torch.Tensor) else np.iscomplexobj(v):
        raise ValueError(f"readout {readout!r}: encode must return real features (split complex ones into real and "
                         "imaginary parts)")
    a = v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError(f"readout {readout!r}: encode must return finite [B, D] features with D of 1 or more, got "
                         f"shape {a.shape}")
    return a


def _readouts_of(feats, readouts: list, n: int) -> dict:
    """{readout: [n, D] numpy} of the output of encode for a batch of n channels, after the checks of its form."""
    if not isinstance(feats, dict) or not set(readouts) <= set(feats):
        got = sorted(feats) if isinstance(feats, dict) else type(feats).__name__
        raise ValueError(f"encode must return a dict with the readouts {readouts}, got {got}")
    out = {r: _as_numpy(feats[r], r) for r in readouts}
    for r, a in out.items():
        if len(a) != n:
            raise ValueError(f"readout {r!r}: {len(a)} rows for a batch of {n} channels")
    return out


def check_encode(encode, h: torch.Tensor, readouts: list) -> None:
    """Stop if encode does not give the same vector for the same channel. encode runs on the batch h twice, and on
    its first CHECK_CHANNELS channels alone. The first test finds random steps, such as dropout in training mode. The
    second finds features that depend on the other channels of the batch, such as BatchNorm in training mode."""
    h = h.to(torch.complex64)
    k = min(CHECK_CHANNELS, len(h) // 2)
    a, b = (_readouts_of(encode(h), readouts, len(h)) for _ in range(2))
    c = _readouts_of(encode(h[:k]), readouts, k) if k else None
    if c is None:
        print(f"check of encode: {len(h)} channel in the first batch, so the test of the batch dependence is skipped",
              flush=True)
    for r in readouts:
        tol = CHECK_RTOL * max(float(np.abs(a[r]).max()), np.finfo(np.float32).tiny)
        if np.abs(a[r] - b[r]).max() > tol:
            raise ValueError(f"readout {r!r}: two calls of encode on the same channels give other features. Put the "
                             "model in evaluation mode (model.eval()) and remove random steps from encode.")
        if c is not None and np.abs(a[r][:k] - c[r]).max() > tol:
            raise ValueError(f"readout {r!r}: the features of a channel depend on the other channels of its batch. "
                             "Put the model in evaluation mode, and do not normalize over the batch.")


def _encode_set(encode, channels, set_index, snr_db, readouts, tmp_of) -> None:
    """Encode one set into a temporary file per readout (tmp_of(readout))."""
    out = {}
    try:
        for start, h in channel_batches(channels, set_index, snr_db):
            feats = _readouts_of(encode(h.to(torch.complex64)), readouts, len(h))
            for r, a in feats.items():
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


def extract(encode, cfg: dict, model: str, snr_db: float = None, check: bool = True) -> Path:
    """Write the feature files of model (a model of the config with 'readouts') from encode. Returns the directory
    of the model's features. The new files replace the old ones only after every set is encoded, so that a failure of
    encode leaves the old features as they were. Two extractions of the same model and SNR must not run at once.
    With check, encode is first tested on the first batch of the train set (check_encode)."""
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
    if check:
        with torch.no_grad():
            _, h = next(channel_batches(np.load(data_dir / "train.npy", mmap_mode="r"), 0, snr_db))
            check_encode(encode, h, readouts)
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
