"""The 24 quantities that ChaSE probes and their probe targets.

Each target maps the labels of one set of links to (y [n, k] float64, valid [n] bool). The labels are the path
statistics of CSIGen (0.3.3 or later) and the positions, velocities and orientations of the links. A quantity is
probed only on the links where it is defined.

Quantities that span orders of magnitude are probed in log10. Angles are probed as (sin, cos). Departure angles are
taken in the frame of the array of the sector, which is the frame in which the channel shows them. Arrival angles are
taken in the frame of the user. The global direction is rotated by the inverse of the orientation (Sionna,
R = Rz(a) Ry(b) Rx(c)).

The targets read these label fields.
  distance_m, path_gain_db, k_factor_db, delay_spread_s, first_delay_s, dominant_tau_s, asd_rad, zsd_rad, asa_rad,
  zsa_rad, dominant_theta_t, dominant_phi_t, dominant_theta_r, dominant_phi_r, dominant_doppler_hz,
  max_abs_doppler_hz, power_frac_los, power_frac_diffraction, power_frac_specular, power_frac_refraction,
  dominant_power_fraction, dominant_num_bounces, rx_velocity [n, 3], rx_orientation [n, 3], tx_orientation [n, 3]
"""

from typing import Callable, Dict, Tuple

import numpy as np

GROUPS: Dict[str, list] = {
    "link": ["distance", "path_gain", "k_factor", "los_share", "diffraction_share", "specular_share",
             "refraction_share", "dominant_share", "dominant_bounces"],
    "delay": ["delay_spread", "first_delay", "excess_delay"],
    "bs_angles": ["aod", "zod", "asd", "zsd"],
    "motion": ["doppler", "abs_doppler", "max_doppler", "speed"],
    "user_angles": ["aoa", "zoa", "asa", "zsa"],
}
QUANTITIES = [q for g in GROUPS.values() for q in g]
ANGLES = ("aod", "zod", "aoa", "zoa")                        # probed as (sin, cos), scored with the pooled R^2
DELAY_SPREAD_FLOOR_S = 1e-9                                  # added before log10, since one path has zero spread

Target = Tuple[np.ndarray, np.ndarray]


def rotation(orientation: np.ndarray) -> np.ndarray:
    """[n, 3, 3] rotation matrices of Sionna orientations [n, 3] = (a about z, b about y, c about x)."""
    a, b, c = (orientation[:, i].astype(np.float64) for i in range(3))
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.stack([
        np.stack([ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc], -1),
        np.stack([sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc], -1),
        np.stack([-sb, cb * sc, cb * cc], -1)], -2)


def local_angles(theta: np.ndarray, phi: np.ndarray, orientation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zenith and azimuth (rad) of the global directions (theta, phi) in the frame of the given orientations."""
    th, ph = theta.astype(np.float64), phi.astype(np.float64)
    d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    loc = np.einsum("nji,nj->ni", rotation(orientation), d)            # R^T d
    return np.arccos(np.clip(loc[:, 2], -1, 1)), np.arctan2(loc[:, 1], loc[:, 0])


def _scalar(v: np.ndarray, log: bool = False, floor: float = 0.0) -> Target:
    v = v.astype(np.float64)
    if log:
        v = np.log10(np.maximum(v, 0) + floor) if floor else np.log10(np.where(v > 0, v, np.nan))
    ok = np.isfinite(v)
    return np.where(ok, v, 0)[:, None], ok


def _angle(a: np.ndarray) -> Target:
    ok = np.isfinite(a)
    a = np.where(ok, a, 0)
    return np.stack([np.sin(a), np.cos(a)], 1), ok


def _departure(lab, which):
    th, ph = local_angles(lab["dominant_theta_t"], lab["dominant_phi_t"], lab["tx_orientation"])
    return _angle(ph if which == "azimuth" else th)


def _arrival(lab, which):
    th, ph = local_angles(lab["dominant_theta_r"], lab["dominant_phi_r"], lab["rx_orientation"])
    return _angle(ph if which == "azimuth" else th)


TARGETS: Dict[str, Callable[[Dict[str, np.ndarray]], Target]] = {
    "distance": lambda l: _scalar(l["distance_m"], log=True),
    "path_gain": lambda l: _scalar(l["path_gain_db"]),
    "k_factor": lambda l: _scalar(l["k_factor_db"]),                   # no NaN (no LoS) or inf (LoS only)
    "los_share": lambda l: _scalar(l["power_frac_los"]),
    "diffraction_share": lambda l: _scalar(l["power_frac_diffraction"]),
    "specular_share": lambda l: _scalar(l["power_frac_specular"]),
    "refraction_share": lambda l: _scalar(l["power_frac_refraction"]),
    "dominant_share": lambda l: _scalar(l["dominant_power_fraction"]),
    "dominant_bounces": lambda l: _scalar(l["dominant_num_bounces"]),
    "delay_spread": lambda l: _scalar(l["delay_spread_s"], log=True, floor=DELAY_SPREAD_FLOOR_S),
    "first_delay": lambda l: _scalar(l["first_delay_s"] * 1e6),                           # us
    "excess_delay": lambda l: _scalar((l["dominant_tau_s"] - l["first_delay_s"]) * 1e6),  # us
    "aod": lambda l: _departure(l, "azimuth"),
    "zod": lambda l: _departure(l, "zenith"),
    "asd": lambda l: _scalar(l["asd_rad"]),
    "zsd": lambda l: _scalar(l["zsd_rad"]),
    "doppler": lambda l: _scalar(l["dominant_doppler_hz"]),            # signed
    "abs_doppler": lambda l: _scalar(np.abs(l["dominant_doppler_hz"])),
    "max_doppler": lambda l: _scalar(l["max_abs_doppler_hz"]),
    "speed": lambda l: _scalar(np.linalg.norm(l["rx_velocity"], axis=1)),
    "aoa": lambda l: _arrival(l, "azimuth"),
    "zoa": lambda l: _arrival(l, "zenith"),
    "asa": lambda l: _scalar(l["asa_rad"]),
    "zsa": lambda l: _scalar(l["zsa_rad"]),
}
