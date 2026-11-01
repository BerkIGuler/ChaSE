"""The feature files of an encode function, with the dummy encoder of scripts/extract_dummy.py."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import chase.run as R
from chase.extract import extract
from chase.report import write_report
from chase.run import features_path, load_config, run

spec = importlib.util.spec_from_file_location("extract_dummy", Path(__file__).parents[1] / "scripts/extract_dummy.py")
extract_dummy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_dummy)
WIDTH, dummy_encoder = extract_dummy.WIDTH, extract_dummy.dummy_encoder


@pytest.fixture
def cfg(config):
    c = load_config(config)
    c["models"]["dummy"] = {"readouts": ["mean", "spatial"]}
    return c


def test_dummy_features_run_and_report(cfg, sizes):
    extract(dummy_encoder(), cfg, "dummy")
    for s, n in sizes.items():
        assert np.load(features_path(cfg, "dummy", "mean") / f"{s}.npy").shape == (n, WIDTH)
        assert np.load(features_path(cfg, "dummy", "spatial") / f"{s}.npy").shape == (n, 4 * WIDTH)
    src = json.loads((features_path(cfg, "dummy", "mean") / "source.json").read_text())
    assert src["model"] == "dummy" and src["snr_db"] is None and set(src["channels"]) == set(sizes)
    assert src["config"]["models"]["dummy"] == {"readouts": ["mean", "spatial"]}
    assert not list(features_path(cfg, "dummy", "mean").glob("*.tmp.npy"))
    cfg["quantities"] = ["path_gain", "distance"]
    run(cfg, models=["raw", "dummy"], device="cpu")
    rep = json.loads(write_report(cfg).read_text())
    assert set(rep["models"]["dummy"]["readouts"]) == {"mean", "spatial"}
    assert set(rep["models"]["dummy"]["lost"]) == {"path_gain", "distance"}


def test_noise_is_fixed_and_in_its_own_directory(cfg):
    extract(dummy_encoder(), cfg, "dummy")
    extract(dummy_encoder(), cfg, "dummy", snr_db=10)
    clean = np.load(features_path(cfg, "dummy", "mean") / "test.npy")
    noisy = np.load(features_path(cfg, "dummy", "mean", 10) / "test.npy")
    assert "snr10" in str(features_path(cfg, "dummy", "mean", 10)) and not np.allclose(clean, noisy)
    extract(dummy_encoder(), cfg, "dummy", snr_db=10)
    assert np.array_equal(noisy, np.load(features_path(cfg, "dummy", "mean", 10) / "test.npy"))


@pytest.mark.parametrize("encode, message", [
    (lambda h: {"mean": h.real.flatten(1)}, "readouts"),
    (lambda h: {"mean": h.real.flatten(1), "temporal": h.real.flatten(1)}, "readouts"),
    (lambda h: h.real.flatten(1), "a dict"),
    (lambda h: {"mean": h.real.flatten(1), "spatial": h.real.flatten(1) * np.nan}, "finite"),
    (lambda h: {"mean": h.real.flatten(1)[:1], "spatial": h.real.flatten(1)}, "rows"),
    (lambda h: {"mean": h.real.flatten(1)[:, :0], "spatial": h.real.flatten(1)}, "D of 1 or more"),
])
def test_encode_errors(cfg, encode, message):
    with pytest.raises(ValueError, match=message):
        extract(encode, cfg, "dummy")


def test_only_models_with_readouts(cfg):
    with pytest.raises(ValueError, match="readouts"):
        extract(dummy_encoder(), cfg, "raw")


def test_failed_extraction_keeps_the_old_features(cfg):
    extract(dummy_encoder(), cfg, "dummy")
    d = features_path(cfg, "dummy", "mean")
    old = {s: np.load(d / f"{s}.npy") for s in ("train", "val", "test")}
    calls = []

    def fails_on_val(h):
        calls.append(len(h))
        if sum(calls) > 300:                                    # after the train set
            raise RuntimeError("out of memory")
        return dummy_encoder(seed=1)(h)
    with pytest.raises(RuntimeError):
        extract(fails_on_val, cfg, "dummy", check=False)               # fails in the val set, not in the check
    for s, v in old.items():
        assert np.array_equal(v, np.load(d / f"{s}.npy"))
    assert not list(d.parent.glob("*/*.tmp.npy"))


def test_failed_first_extraction_leaves_no_directories(cfg):
    def fails(h):
        raise RuntimeError("out of memory")
    with pytest.raises(RuntimeError):
        extract(fails, cfg, "dummy")
    assert not (Path(cfg["features_dir"]).expanduser() / "dummy").exists()
    with pytest.raises(RuntimeError):
        extract(fails, cfg, "dummy", snr_db=10)
    assert not (Path(cfg["features_dir"]).expanduser() / "snr10").exists()


def test_data_are_checked_before_encoding(cfg):
    data = Path(cfg["data_dir"]).expanduser()
    np.save(data / "val.npy", np.load(data / "val.npy")[:-1])
    with pytest.raises(ValueError, match="val: 159 channels and 160 labels"):
        extract(dummy_encoder(), cfg, "dummy")
    assert not (Path(cfg["features_dir"]).expanduser() / "dummy").exists()


def test_complex_features_are_refused(cfg):
    with pytest.raises(ValueError, match="real features"):
        extract(lambda h: {"mean": h.flatten(1), "spatial": h.flatten(1)}, cfg, "dummy")


def test_bfloat16_features_are_accepted(cfg):
    enc = dummy_encoder()
    extract(lambda h: {k: v.bfloat16() for k, v in enc(h).items()}, cfg, "dummy")
    assert np.load(features_path(cfg, "dummy", "mean") / "test.npy").dtype == np.float32


def test_extra_readouts_are_ignored(cfg):
    enc = dummy_encoder()
    extract(lambda h: {**enc(h), "temporal": h.real.flatten(1)}, cfg, "dummy")
    assert not features_path(cfg, "dummy", "temporal").exists()


def test_run_stops_for_features_of_another_snr(cfg, monkeypatch):
    extract(dummy_encoder(), cfg, "dummy")
    extract(dummy_encoder(), cfg, "dummy", snr_db=20)
    for r in ("mean", "spatial"):                                       # clean and 20 dB features in other folders
        shutil.copytree(features_path(cfg, "dummy", r), features_path(cfg, "dummy", r, 10))
        shutil.copytree(features_path(cfg, "dummy", r, 20), features_path(cfg, "dummy", r, 5))
    monkeypatch.setattr(R, "probe_quantity", lambda *a: (_ for _ in ()).throw(AssertionError("probed")))
    with pytest.raises(ValueError, match="at clean, the run is at 10 dB"):
        run(cfg, models=["raw", "dummy"], snr_db=10, device="cpu")     # stops before raw is probed
    with pytest.raises(ValueError, match="at 20 dB, the run is at 5 dB"):
        run(cfg, models=["dummy"], snr_db=5, device="cpu")
    monkeypatch.undo()
    cfg["quantities"] = ["path_gain"]
    extract(dummy_encoder(), cfg, "dummy", snr_db=10)                   # the right features run
    run(cfg, models=["dummy"], snr_db=10.0000001, device="cpu")        # the folder name is snr10 too
    for r in ("mean", "spatial"):                                       # features without a source.json run
        (features_path(cfg, "dummy", r) / "source.json").unlink()
    run(cfg, models=["dummy"], device="cpu")


def _token_model(dropout):
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.Dropout(dropout))


def _encode_with(net):
    def encode(h):
        z = net(torch.cat([h.real, h.imag], -1).flatten(1, 3).float())      # tokens [B, T*U*N, 16]
        return {"mean": z.mean(1), "spatial": z.amax(1)}
    return encode


def test_check_finds_dropout_in_training_mode(cfg):
    net = _token_model(0.5).train()
    with pytest.raises(ValueError, match="evaluation mode"):
        extract(_encode_with(net), cfg, "dummy")
    assert not (Path(cfg["features_dir"]).expanduser() / "dummy").exists()   # nothing written
    extract(_encode_with(net.eval()), cfg, "dummy")                    # the same model in evaluation mode passes


def test_check_finds_features_that_depend_on_the_batch(cfg):
    enc = dummy_encoder()

    def batch_normalized(h):
        return {k: v - v.mean(0) for k, v in enc(h).items()}
    with pytest.raises(ValueError, match="other channels of its batch"):
        extract(batch_normalized, cfg, "dummy")
    extract(batch_normalized, cfg, "dummy", check=False)               # the check can be turned off
