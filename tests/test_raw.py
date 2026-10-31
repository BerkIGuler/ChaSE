import numpy as np
import pytest
import torch

from chase.raw import NOISE_BATCH, add_noise, channel_batches, mean_power, raw_set


def _channels(n=5, seed=0, users=1):
    g = torch.Generator().manual_seed(seed)
    shape = (n, 14, users, 32, 32)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g)) * 1e-6


def test_raw_complex_and_logmag():
    h = _channels()
    mp = mean_power(h.numpy())
    assert np.isclose(mp, float(h.abs().pow(2).mean()), rtol=1e-5)
    x = h[:, 0] / mp ** 0.5
    f = raw_set(h.numpy(), "complex", mp, 0)
    assert f.shape == (5, 2048)
    torch.testing.assert_close(torch.from_numpy(f), torch.cat([x.real.flatten(1), x.imag.flatten(1)], 1))
    f = raw_set(h.numpy(), "logmag", mp, 0)
    torch.testing.assert_close(torch.from_numpy(f), torch.log10(x.abs().pow(2).flatten(1) + 1e-12))
    assert raw_set(h.numpy(), "complex_t", mp, 0).shape == (5, 2 * 14 * 32 * 8)
    assert raw_set(h.numpy(), "logmag_t", mp, 0).shape == (5, 14 * 32 * 8)


def test_raw_with_two_user_antennas():
    h = _channels(users=2).numpy()
    assert raw_set(h, "complex", 1.0, 0).shape == (5, 2 * 2 * 32 * 32)
    assert raw_set(h, "logmag_t", 1.0, 0).shape == (5, 14 * 2 * 32 * 8)


def test_add_noise():
    h = _channels(200)
    y = add_noise(h, 10.0, seed=3)
    snr = h.abs().pow(2).flatten(1).mean(1) / (y - h).abs().pow(2).flatten(1).mean(1)
    assert abs(10 * torch.log10(snr).mean().item() - 10.0) < 0.05
    assert torch.equal(add_noise(h, 10.0, seed=3), y) and not torch.equal(add_noise(h, 10.0, seed=4), y)


def test_noise_is_fixed_per_set_and_channel():
    h = _channels(NOISE_BATCH + 3).numpy()
    a = torch.cat([b for _, b in channel_batches(h, 0, 10.0)])
    b = torch.cat([b for _, b in channel_batches(h, 0, 10.0)])
    c = torch.cat([b for _, b in channel_batches(h, 1, 10.0)])
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(torch.cat([b for _, b in channel_batches(h, 0)]), torch.from_numpy(h))


def test_four_axis_channels_are_refused():
    with pytest.raises(ValueError, match="5 axes"):
        raw_set(np.zeros((3, 14, 32, 32), np.complex64), "complex", 1.0, 0)
