"""Write the feature files of a dummy encoder, as an example of how a model enters ChaSE.

    python scripts/extract_dummy.py configs/example.yaml [--model dummy] [--snr-db 10]

The dummy encoder projects the real and imaginary parts of each OFDM symbol and antenna pair with a fixed random matrix.
That gives one token per symbol and antenna pair. Its readouts are the mean over all tokens ("mean") and the mean over
the symbols and user antennas per base-station antenna, concatenated ("spatial"). Replace dummy_encoder with a function
that runs your own model.
"""

import argparse
from pathlib import Path

import torch

from chase.extract import extract
from chase.run import load_config

WIDTH = 16


def dummy_encoder(seed: int = 0):
    """An encode function. Its input is a batch of channels [B, T, U, N, K] complex64."""
    proj = {}

    def encode(h: torch.Tensor) -> dict:
        k = h.shape[-1]
        if k not in proj:
            proj[k] = torch.randn(2 * k, WIDTH, generator=torch.Generator().manual_seed(seed))
        x = torch.cat([h.real, h.imag], -1)                          # [B, T, U, N, 2K]
        x = x / x.flatten(1).std(1).clamp_min(1e-30).view(-1, 1, 1, 1, 1)
        z = x @ proj[k]                                    # [B, T, U, N, WIDTH], a token per symbol and pair
        return {"mean": z.mean((1, 2, 3)), "spatial": z.mean((1, 2)).flatten(1)}
    return encode


def main():
    ap = argparse.ArgumentParser(description="Feature files of a dummy encoder")
    ap.add_argument("config", type=Path)
    ap.add_argument("--model", default="dummy", help="the name of the model in the config")
    ap.add_argument("--snr-db", type=float, default=None, help="encode the channels with AWGN at this SNR")
    a = ap.parse_args()
    print(extract(dummy_encoder(), load_config(a.config), a.model, a.snr_db))


if __name__ == "__main__":
    main()
