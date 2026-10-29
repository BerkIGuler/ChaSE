"""The channel (CFR) of a CSIGen file, rebuilt from its saved paths.

A copy of load_cfr and cfr_from_paths of CSIGen 0.3.6 (src/path_data.py), without the options that change the
numerology, the arrays or the paths. It gives the same array as the h that CSIGen saved, bit for bit. The copy lets
ChaSE rebuild the channels with numpy only, without CSIGen and Sionna.
"""

from pathlib import Path
from typing import Dict

import numpy as np

# The saved arrays that the CFR needs (P is the total number of paths of the file)
CFR_KEYS = ("paths_offsets",    # int64 [num_channels + 1], the paths of channel i are rows offsets[i]:offsets[i + 1]
            "paths_a",          # complex64 [P, num_rx_ant, num_tx_ant], path coefficient per antenna pair
            "paths_tau",        # float32 [P], absolute delay (s)
            "paths_doppler")    # float32 [P], Doppler shift (Hz)
_CFR_MAX_ELEMENTS = 2 ** 24


def subcarrier_offsets(num_subcarriers: int, subcarrier_spacing: float) -> np.ndarray:
    """Baseband subcarrier frequencies (Hz), as Sionna's subcarrier_frequencies."""
    if num_subcarriers % 2 == 0:
        start, limit = -num_subcarriers / 2, num_subcarriers / 2
    else:
        start, limit = -(num_subcarriers - 1) / 2, (num_subcarriers - 1) / 2 + 1
    return np.arange(start, limit, dtype=np.float64) * float(subcarrier_spacing)


def cfr_from_paths(path_data: Dict[str, np.ndarray], carrier_frequency: float, num_subcarriers: int,
                   num_ofdm_symbols: int, subcarrier_spacing: float, normalize_delays: bool = False,
                   normalize: bool = False) -> np.ndarray:
    """The CFR of every channel from its paths, computed in float64.

    h(f, t) = sum_l a_l exp(-j 2 pi (f_c + f) tau_l) exp(j 2 pi nu_l t), with f the subcarrier offsets and
    t = n / subcarrier_spacing, as in Sionna's Paths.cfr. With normalize_delays, the earliest delay of each channel is
    subtracted first. With normalize, each channel is scaled to unit average energy.

    Returns complex64 [num_channels, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, num_subcarriers].
    """
    offsets = path_data["paths_offsets"]
    num_ch = len(offsets) - 1
    _, num_rx_ant, num_tx_ant = path_data["paths_a"].shape
    freqs = subcarrier_offsets(num_subcarriers, subcarrier_spacing)
    times = np.arange(num_ofdm_symbols, dtype=np.float64) * (1 / float(subcarrier_spacing))
    fc = float(carrier_frequency)
    h = np.zeros((num_ch, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, num_subcarriers), dtype=np.complex64)
    counts = np.diff(offsets)

    grid = num_ofdm_symbols * num_subcarriers
    start = 0
    while start < num_ch:
        # Add channels while the padded phase array stays within the bound
        stop, p_max = start + 1, int(counts[start])
        while stop < num_ch and (stop + 1 - start) * max(p_max, int(counts[stop])) * grid <= _CFR_MAX_ELEMENTS:
            p_max = max(p_max, int(counts[stop]))
            stop += 1
        n = stop - start
        h_chunk = np.zeros((n, num_rx_ant, num_tx_ant, num_ofdm_symbols, num_subcarriers), dtype=np.complex128)
        if p_max > 0:
            # Pad the paths of the chunk to [n, p_max], the padded paths have a = 0
            sel = np.arange(offsets[start], offsets[stop])
            row = np.repeat(np.arange(n), counts[start:stop])
            col = sel - np.repeat(offsets[start:stop], counts[start:stop])
            a = np.zeros((n, num_rx_ant, num_tx_ant, p_max), dtype=np.complex128)
            a[row, :, :, col] = path_data["paths_a"][sel]
            tau = np.zeros((n, p_max))
            tau[row, col] = path_data["paths_tau"][sel]
            nu = np.zeros((n, p_max))
            nu[row, col] = path_data["paths_doppler"][sel]
            if normalize_delays:
                first = np.full(n, np.inf)
                np.minimum.at(first, row, tau[row, col])
                tau[row, col] -= first[row]
            a = a * np.exp(-2j * np.pi * fc * tau)[:, None, None, :]
            # A single channel with very many paths is summed in blocks of paths
            block = max(1, _CFR_MAX_ELEMENTS // (n * grid))
            for p0 in range(0, p_max, block):
                p1 = min(p0 + block, p_max)
                phase = (2 * np.pi * nu[:, p0:p1, None, None] * times[None, None, :, None]
                         - 2 * np.pi * freqs[None, None, None, :] * tau[:, p0:p1, None, None])
                h_chunk += np.einsum("rqep,rptf->rqetf", a[..., p0:p1], np.exp(1j * phase))
        if normalize:
            c = np.mean(np.abs(h_chunk) ** 2, axis=(1, 2, 3, 4))
            scale = np.where(c > 0, 1 / np.sqrt(np.where(c > 0, c, 1)), 0)
            h_chunk = h_chunk * scale[:, None, None, None, None]
        h[start:stop, :, 0] = h_chunk
        start = stop
    return h


def load_cfr(npz_path) -> np.ndarray:
    """The CFR of a CSIGen channel_tx_<i>.npz file, rebuilt from its saved paths with the numerology of its run."""
    with np.load(Path(npz_path), allow_pickle=True) as z:
        meta = z["metadata"].item()
        if "paths_offsets" not in z:
            raise ValueError(f"{npz_path} has no saved paths")
        path_data = {k: z[k] for k in CFR_KEYS}
    p = meta["cfr_params"]
    return cfr_from_paths(path_data, carrier_frequency=p["carrier_frequency"], num_subcarriers=p["num_subcarriers"],
                          num_ofdm_symbols=p["num_ofdm_symbols"], subcarrier_spacing=p["subcarrier_spacing"],
                          normalize_delays=p["normalize_delays"], normalize=p["normalize"])
