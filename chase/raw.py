"""The raw-channel baselines and the noise that ChaSE adds to the channels.

The raw baselines are the reference of the lost share. They take the channels [n, T, N, K] (OFDM symbols, antennas,
subcarriers), divided by the square root of a mean power, in four variants.

- complex. The real and imaginary parts of the first OFDM symbol (2 N K values).
- logmag. log10 |h|^2 of the first OFDM symbol (N K values).
- complex_t and logmag_t. The same over all OFDM symbols and antennas and every SUBCARRIER_STRIDE-th subcarrier, so
  that the time axis, which motion needs, is kept and the ridge probe stays tractable.

With an SNR, every channel gets complex AWGN at that SNR relative to its own mean power over the grid. The noise of a
channel is fixed by its set and its index, so that every model and raw baseline sees the same noisy channels.
"""

import numpy as np
import torch

RAW_VARIANTS = ("complex", "logmag", "complex_t", "logmag_t")
SUBCARRIER_STRIDE = 4
LOGMAG_FLOOR = 1e-12
# The noise of a channel depends on the batch that holds it, so the batch size is fixed
NOISE_BATCH = 512


def add_noise(h: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    """Complex AWGN at snr_db relative to the mean power of each channel over its grid."""
    g = torch.Generator().manual_seed(int(seed))
    p = h.abs().pow(2).flatten(1).mean(1).view(-1, *([1] * (h.dim() - 1)))
    n = torch.complex(torch.randn(h.shape, generator=g), torch.randn(h.shape, generator=g))
    return h + (n * (p / 10 ** (snr_db / 10) / 2).sqrt()).to(h.dtype)


def noise_seed(set_index: int, snr_db: float, start: int) -> int:
    """The noise seed of the batch that starts at channel start of a set (train 0, val 1, test 2)."""
    return set_index * 1_000_003 + int(round(snr_db * 10)) * 10_007 + start


def channel_batches(channels: np.ndarray, set_index: int, snr_db: float = None):
    """(start, channels [B, T, N, K]) for every batch of NOISE_BATCH channels, with AWGN at snr_db if it is given.
    channels may be a memory map."""
    for i in range(0, len(channels), NOISE_BATCH):
        h = torch.from_numpy(np.array(channels[i:i + NOISE_BATCH]))
        if snr_db is not None:
            h = add_noise(h, snr_db, noise_seed(set_index, snr_db, i))
        yield i, h


def mean_power(channels: np.ndarray) -> float:
    """The mean of |h|^2 over all channels and entries."""
    total = 0.0
    for i in range(0, len(channels), NOISE_BATCH):
        total += float((np.abs(np.asarray(channels[i:i + NOISE_BATCH])).astype(np.float64) ** 2).sum())
    return total / (channels[0].size * len(channels))


def raw_features(h: torch.Tensor, variant: str, scale: float) -> torch.Tensor:
    """Raw-baseline features [B, D] of channels [B, T, N, K]."""
    h = h.to(torch.complex64) * scale
    x = h[..., ::SUBCARRIER_STRIDE] if variant.endswith("_t") else h[:, 0]
    if variant.startswith("logmag"):
        return torch.log10(x.abs().pow(2).flatten(1) + LOGMAG_FLOOR)
    return torch.cat([x.real.flatten(1), x.imag.flatten(1)], 1)


def raw_set(channels: np.ndarray, variant: str, power: float, set_index: int, snr_db: float = None) -> np.ndarray:
    """Raw-baseline features [n, D] float32 of one set of channels."""
    if variant not in RAW_VARIANTS:
        raise ValueError(f"unknown raw variant {variant!r} (one of {RAW_VARIANTS})")
    scale = 1.0 / float(np.sqrt(power))
    return np.concatenate([raw_features(h, variant, scale).numpy()
                           for _, h in channel_batches(channels, set_index, snr_db)])
