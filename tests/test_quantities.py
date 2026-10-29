import numpy as np

from chase.quantities import ANGLES, GROUPS, QUANTITIES, TARGETS, local_angles, rotation

FIELDS = ("distance_m", "path_gain_db", "k_factor_db", "delay_spread_s", "first_delay_s", "dominant_tau_s", "asd_rad",
          "zsd_rad", "asa_rad", "zsa_rad", "dominant_theta_t", "dominant_phi_t", "dominant_theta_r", "dominant_phi_r",
          "dominant_doppler_hz", "max_abs_doppler_hz", "power_frac_los", "power_frac_diffraction",
          "power_frac_specular", "power_frac_refraction", "dominant_power_fraction", "dominant_num_bounces")


def test_every_quantity_has_a_target():
    assert len(QUANTITIES) == 24 == len(set(QUANTITIES)) and set(TARGETS) == set(QUANTITIES)
    assert set(ANGLES) <= set(QUANTITIES) and list(GROUPS) == ["link", "delay", "bs_angles", "motion", "user_angles"]


def test_local_angles():
    assert np.allclose(rotation(np.zeros((1, 3)))[0], np.eye(3))
    th, ph = np.array([np.pi / 2]), np.array([np.pi / 3])
    t2, p2 = local_angles(th, ph, np.array([[np.pi / 3, 0, 0]]))   # array yawed towards the direction, so boresight
    assert np.isclose(t2[0], np.pi / 2) and np.isclose(p2[0], 0, atol=1e-12)
    t3, _ = local_angles(np.array([np.pi / 2 + 0.2]), np.array([0.0]), np.array([[0, 0.2, 0]]))  # downtilt 0.2 rad
    assert np.isclose(t3[0], np.pi / 2)


def test_targets_handle_undefined_values():
    n = 4
    lab = {k: np.zeros(n, np.float32) for k in FIELDS}
    lab.update(rx_velocity=np.zeros((n, 3), np.float32), tx_orientation=np.zeros((n, 3), np.float32),
               rx_orientation=np.zeros((n, 3), np.float32))
    lab["k_factor_db"] = np.array([3, np.nan, np.inf, 1], np.float32)
    lab["distance_m"] = np.array([10, 100, 0, 1000], np.float32)
    y, ok = TARGETS["k_factor"](lab)
    assert ok.tolist() == [True, False, False, True] and np.isfinite(y).all()
    y, ok = TARGETS["distance"](lab)
    assert ok.tolist() == [True, True, False, True] and np.allclose(y[ok, 0], [1, 2, 3])
    y, ok = TARGETS["delay_spread"](lab)                           # zero spread (one path) gives log10(1 ns)
    assert ok.all() and np.allclose(y[:, 0], -9)
    for q in QUANTITIES:
        y, ok = TARGETS[q](lab)
        assert y.shape == (n, 2 if q in ANGLES else 1) and ok.shape == (n,)
