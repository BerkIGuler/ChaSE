# Data format

ChaSE reads the data of an evaluation from one folder, the `data_dir` of the config. This document describes the files of that folder, so that you can write them from any simulator or data set. If you use CSIGen, `scripts/build_data.py` writes them for you (step 1 of the README).

## Files

For each of the three sets, `train`, `val` and `test`, the folder holds two files.

| File | Contents |
|---|---|
| `<set>.npy` | The channels of the set |
| `<set>_labels.npz` | The ground truth of each channel of the set |

The rows of the two files of a set belong together. Row i of `<set>.npy` and row i of every array in `<set>_labels.npz` describe the same channel.

The folder can also hold a `manifest.yaml` (see [Manifest](#manifest)). It is optional.

## Channels

`<set>.npy` is one numpy array, saved with `numpy.save`.

| Property | Value |
|---|---|
| Shape | [n, T, U, N, K], with n channels, T OFDM symbols, U user antennas, N base-station antennas and K subcarriers |
| Type | complex64 |
| Units | the complex gain of the channel, without normalization |
| Noise | none, because ChaSE adds the noise itself when you ask for an SNR |

Each channel is the channel of one link between a base-station sector and a user. T, U, N and K must be the same for all channels of the three sets. ChaSE reads `<set>.npy` for the raw baselines and for `chase.extract`. If you write the features of your models in another way and probe no raw baseline, ChaSE does not read it.

## Labels

`<set>_labels.npz` holds one numpy array per field, saved with `numpy.savez`. Each array has one row per channel. Fields that ChaSE does not read are ignored.

### Fields that ChaSE always reads

`city` and `rx_position` must be finite for every link, because ChaSE puts each link into a tile. ChaSE stops if they are not.

| Field | Shape | Unit | Meaning |
|---|---|---|---|
| `city` | [n] integer | | The scene of the link, for example its city. A tile never holds links of two scenes. |
| `rx_position` | [n, 3] float | m | The position of the user (x, y, z). The bootstrap and the two validation halves use x and y to group the links into tiles of 50 m. |
| `path_gain_db` | [n] float | dB | The total power gain of all paths. The coverage floor `min_path_gain_db` keeps only the links with at least this gain. It is also the target of `path_gain`. |

### Fields of the quantities

ChaSE reads the fields of the quantities that the config probes (the `quantities` key, all 24 by default). Before the first probe, it checks that every set has these fields, and it stops with a list of the missing ones. The power of a path is its mean |a|^2 over the antenna pairs. Spreads and shares weight each path by its power. A value that is not finite (NaN or inf) marks a quantity as undefined for that link. ChaSE then leaves the link out of the probes of that quantity. A distance of 0 or less is undefined too, since it has no log10. A negative delay spread counts as 0.

| Quantity | Field | Shape | Unit | Meaning | Target |
|---|---|---|---|---|---|
| `distance` | `distance_m` | [n] | m | 3D distance between the base station and the user | log10 |
| `path_gain` | `path_gain_db` | [n] | dB | total power gain of all paths | value |
| `k_factor` | `k_factor_db` | [n] | dB | power of the LoS path over the power of all other paths. NaN without a LoS path, inf if the LoS path is the only path | value |
| `los_share` | `power_frac_los` | [n] | fraction | share of the power in the LoS path, 0 without one | value |
| `diffraction_share` | `power_frac_diffraction` | [n] | fraction | share of the power in the paths with at least one diffraction | value |
| `specular_share` | `power_frac_specular` | [n] | fraction | share of the power in the paths with at least one specular reflection | value |
| `refraction_share` | `power_frac_refraction` | [n] | fraction | share of the power in the paths with at least one refraction | value |
| `dominant_share` | `dominant_power_fraction` | [n] | fraction | share of the power in the strongest path | value |
| `dominant_bounces` | `dominant_num_bounces` | [n] | count | number of interactions of the strongest path, 0 for a LoS path | value |
| `delay_spread` | `delay_spread_s` | [n] | s | RMS spread of the path delays | log10 of the value plus 1 ns |
| `first_delay` | `first_delay_s` | [n] | s | delay of the earliest path | value in µs |
| `excess_delay` | `dominant_tau_s` and `first_delay_s` | [n] each | s | delay of the strongest path minus the delay of the earliest path | value in µs |
| `aod`, `zod` | `dominant_phi_t`, `dominant_theta_t` and `tx_orientation` | [n], [n] and [n, 3] | rad | azimuth and zenith angles of departure of the strongest path | sine and cosine, in the frame of the base-station array |
| `asd` | `asd_rad` | [n] | rad | RMS azimuth spread of departure, around the circular mean | value |
| `zsd` | `zsd_rad` | [n] | rad | RMS zenith spread of departure, around the arithmetic mean | value |
| `doppler` | `dominant_doppler_hz` | [n] | Hz | signed Doppler shift of the strongest path | value |
| `abs_doppler` | `dominant_doppler_hz` | [n] | Hz | magnitude of the Doppler shift of the strongest path | absolute value |
| `max_doppler` | `max_abs_doppler_hz` | [n] | Hz | largest Doppler magnitude over the paths | value |
| `speed` | `rx_velocity` | [n, 3] | m/s | velocity of the user | norm of the vector |
| `aoa`, `zoa` | `dominant_phi_r`, `dominant_theta_r` and `rx_orientation` | [n], [n] and [n, 3] | rad | azimuth and zenith angles of arrival of the strongest path | sine and cosine, in the frame of the user array |
| `asa` | `asa_rad` | [n] | rad | RMS azimuth spread of arrival, around the circular mean | value |
| `zsa` | `zsa_rad` | [n] | rad | RMS zenith spread of arrival, around the arithmetic mean | value |

### Angles and orientations

The angles of the strongest path are in the global frame, as in Sionna RT.

- The zenith angle theta is measured from the +z axis, from 0 to pi.
- The azimuth angle phi is measured in the x-y plane from the +x axis towards the +y axis.
- An orientation is three angles (a, b, c) in rad. The rotation of the array is R = Rz(a) Ry(b) Rx(c), first a rotation by c about x, then by b about y, then by a about z.

ChaSE turns the global direction of departure into the frame of the base-station array with the inverse of `tx_orientation`, and the direction of arrival into the frame of the user with the inverse of `rx_orientation`. The channel shows the angles in these frames. If your data gives the angles already in the frames of the arrays, set both orientations to zero.

## Manifest

`manifest.yaml` is optional. `scripts/build_data.py` writes it. ChaSE reads only `statistics.mean_power` from it. The other keys are for you and for your `encode` function.

| Key | Meaning |
|---|---|
| `statistics.mean_power` | The mean power of the channels. The raw baselines divide the channels by its square root. If the config leaves `mean_power` empty, ChaSE takes it from here, and without it, from the train channels. |
| `arrays.base_station`, `arrays.user` | The layout of the arrays (CSIGen's `tx_array` and `rx_array`). ChaSE does not read it, but an `encode` function can use it for its readouts. |

## Partition

`scripts/build_data.py` builds the three sets from CSIGen runs and a partition. `scripts/make_partition.py` writes the partition, so you write one yourself only for a split of your own. A partition folder holds `partition.npz`, with one entry per link of the runs.

| Array | Shape | Meaning |
|---|---|---|
| `cities` | [number of runs] string | the folder names of the runs |
| `parts` | [5] string | the names of the parts, `train`, `val`, `test`, `excluded`, `guard` |
| `city` | [links] integer | the run of the link, as an index into `cities` |
| `tx` | [links] integer | the file `channel_tx_<tx>.npz` of the link |
| `row` | [links] integer | the row of the link in that file |
| `user` | [links] integer | the user of the link |
| `part` | [links] integer | the part of the link, as an index into `parts` |
| `own_site` | [links] bool | whether the link goes to the site that its user was placed for |

The train, val and test sets are the links of the part of the same name with `own_site` true, in the order of the partition. A `manifest.yaml` in the partition folder is optional. If it gives `statistics.mean_power`, the builder copies it. Otherwise, the builder computes the mean power over all links of the train part.

## Example

This code writes the files of one set from your own arrays. All label arrays have one row per channel.

```python
import numpy as np

np.save("data/my_cell/train.npy", h.astype(np.complex64))        # [n, T, U, N, K]
np.savez("data/my_cell/train_labels.npz",
         city=city, rx_position=rx_position, path_gain_db=path_gain_db,   # read in every case
         distance_m=distance_m, delay_spread_s=delay_spread_s)           # and the fields of your quantities
```

With only these label fields, set `quantities: [distance, path_gain, delay_spread]` in the config.
