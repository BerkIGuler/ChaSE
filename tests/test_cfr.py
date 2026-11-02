import numpy as np

from chase import cfr

FC, K, T, SCS = 3.5e9, 8, 3, 30e3


def random_paths(counts, num_rx_ant=2, num_tx_ant=4, seed=0):
    rng = np.random.default_rng(seed)
    p = int(np.sum(counts))
    return {"paths_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            "paths_a": (rng.normal(size=(p, num_rx_ant, num_tx_ant))
                        + 1j * rng.normal(size=(p, num_rx_ant, num_tx_ant))).astype(np.complex64),
            "paths_tau": rng.uniform(1e-7, 1e-6, p).astype(np.float32),
            "paths_doppler": rng.uniform(-100, 100, p).astype(np.float32)}


def test_cfr_matches_the_sum_over_paths():
    counts = [3, 0, 5]
    pd = random_paths(counts)
    h = cfr.cfr_from_paths(pd, FC, K, T, SCS)
    assert h.shape == (3, 2, 1, 4, T, K) and h.dtype == np.complex64
    f = cfr.subcarrier_offsets(K, SCS)
    t = np.arange(T) / SCS
    for i, (s, e) in enumerate(zip(pd["paths_offsets"][:-1], pd["paths_offsets"][1:])):
        a = pd["paths_a"][s:e].astype(np.complex128)
        tau = pd["paths_tau"][s:e].astype(np.float64)[:, None, None]
        nu = pd["paths_doppler"][s:e].astype(np.float64)[:, None, None]
        ph = np.exp(-2j * np.pi * (FC + f[None, None, :]) * tau) * np.exp(2j * np.pi * nu * t[None, :, None])
        ref = np.einsum("pqe,ptf->qetf", a, ph)
        assert np.allclose(h[i, :, 0], ref, rtol=1e-5, atol=1e-6)
    assert not h[1].any()


def test_chunks_do_not_change_the_bits(monkeypatch):
    pd = random_paths([4, 7, 1, 0, 6, 3])
    one_chunk = cfr.cfr_from_paths(pd, FC, K, T, SCS, normalize_delays=True, normalize=True)
    monkeypatch.setattr(cfr, "_CFR_CHUNK_ELEMENTS", 1)                # one channel per chunk
    assert cfr.cfr_from_paths(pd, FC, K, T, SCS, normalize_delays=True, normalize=True).tobytes() == one_chunk.tobytes()


def test_empty_grid_gives_an_empty_cfr():
    assert cfr.cfr_from_paths(random_paths([2, 3]), FC, 0, T, SCS).shape == (2, 2, 1, 4, T, 0)
