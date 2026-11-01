# ChaSE

**Channel Semantics Evaluation.** ChaSE finds what a wireless channel foundation model (WCFM) knows about the physics of a channel. It probes the frozen representations of a model for 24 physical quantities of the propagation, such as the delay spread, the angles of departure and arrival, and the Doppler shift. The ground truth comes from the ray-traced paths of each channel.

ChaSE does not depend on a model or on a data set. It needs only the representations of a model and channels (train, validation and test) with the ground truth of the 24 quantities. The second part of this README shows the demonstration of ChaSE in our paper, with its data and its models.

## What ChaSE measures

ChaSE probes 24 quantities in five groups.

| Group | Quantities |
|---|---|
| `link` | `distance`, `path_gain`, `k_factor`, `los_share`, `diffraction_share`, `specular_share`, `refraction_share`, `dominant_share`, `dominant_bounces` |
| `delay` | `delay_spread`, `first_delay`, `excess_delay` |
| `bs_angles` | `aod`, `zod`, `asd`, `zsd` (the angles of departure at the base station) |
| `motion` | `doppler`, `abs_doppler`, `max_doppler`, `speed` |
| `user_angles` | `aoa`, `zoa`, `asa`, `zsa` (the angles of arrival at the user) |

The angles of the strongest path are probed as the pair of their sine and cosine. The quantities that span orders of magnitude are probed in log space. Which quantities a channel shows depends on the system. For example, a channel shows the angles at the user only if the user has more than one antenna. The `quantities` key of the config selects the quantities to probe.

For each quantity, ChaSE trains a linear (ridge) probe and MLP probes on the representations of the model. It then splits the quantity into four shares.

- **Available.** A linear probe reads it. This is r_lin, the test R^2 of the ridge probe.
- **Entangled.** Only a nonlinear probe reads it. This is r_best - r_lin, where r_best is the test R^2 of the best probe.
- **Unread.** No probe reads it. This is 1 - r_best.
- **Lost.** A probe on the raw channel reads it, but no probe on the representation does. This is max(0, r_best of the raw channel - r_best of the model).

These shares are not exact information-theoretic quantities. The probes give lower bounds on the information that a representation holds, because a better probe, such as a genie probe, could read more.

The readout and the probe are chosen on validation links. A bootstrap over 50 m tiles of the test links gives a 95% interval for each score, because links in the same area are correlated.

As a control, ChaSE shuffles the targets of the train links in a random order, so that each channel gets the target of another channel. For the MLP probes, it also shuffles the targets of the validation links. It then fits the same probes on the shuffled targets and scores them on the real test targets. A control R^2 near zero shows that a probe cannot fit arbitrary targets, so that its score comes from the representation.

## Requirements

- Python 3.12 or later.
- PyTorch 2.4 or later. A GPU makes the MLP probes much faster, but ChaSE also runs on the CPU.
- numpy and PyYAML.

Install PyTorch for your platform first, then ChaSE.

```bash
git clone https://github.com/BerkIGuler/ChaSE.git
cd ChaSE
pip install -e .            # or pip install -e ".[test]" to run the tests with pytest
```

## Usage

An evaluation has five steps. One config holds all the settings. `configs/example.yaml` shows every key with its default.

**1. Prepare the data.** ChaSE works with any channels and their ray-traced ground truth, if they come in this format. For each of train, val and test, ChaSE reads two files.

- `<set>.npy`, the channels [n, OFDM symbol, user antenna, base-station antenna, subcarrier] complex64.
- `<set>_labels.npz`, the ground truth of each channel, with one array per field and one row per channel. `chase/quantities.py` lists the fields that the probe targets read, such as `delay_spread_s` and `dominant_phi_t`. The bootstrap also reads `city` and `rx_position`.

We generate our data with [CSIGen](https://github.com/BerkIGuler/CSIGen), our open-source tool built on [Sionna RT](https://github.com/NVlabs/sionna-rt). ChaSE does not need CSIGen, but ChaSE reads CSIGen runs directly. With CSIGen runs and a partition of their links into train, val and test, `scripts/build_data.py` writes the two files in one step, with no other tool. The builder also writes a `manifest.yaml`, whose `arrays` give the layout of the base-station and user arrays (rows, columns, polarizations, spacing and the order of the antenna index). An encode function can use this layout for its readouts. `chase/data.py` describes the format of the partition.

**2. Set up the config.** Give the data, the folders and the models.

```yaml
data_dir: data/my_cell                   # the channels and labels of step 1 (<set>.npy, <set>_labels.npz)
features_dir: features/my_cell           # the features of the models, written in step 3
out_dir: results/my_cell                 # the results and the report, written in steps 4 and 5
models:                                  # the models to probe, each with a name of your choice
  raw: {raw: complex}                    # raw baseline, real and imaginary parts of the first OFDM symbol
  raw_logmag: {raw: logmag}              # raw baseline, log10 |h|^2 of the first OFDM symbol
  raw_t: {raw: complex_t}                # raw baseline, as raw over all OFDM symbols and every 4th subcarrier
  raw_logmag_t: {raw: logmag_t}          # raw baseline, as raw_logmag over all OFDM symbols and every 4th subcarrier
  my_model: {readouts: [mean, spatial]}  # your model and the names of its readouts (the keys that encode returns)
```

**3. Write the features of your model.** Give ChaSE a function `encode(h)`. ChaSE calls it on the channels of every set, in batches of 512, and writes the feature files. The function must do these steps.

- **Take the channels as ChaSE gives them.** The input `h` is a CPU tensor [B, OFDM symbol, user antenna, base-station antenna, subcarrier] complex64, in the units of the data, without normalization. With an SNR, the noise is already in `h`.
- **Prepare the input as the model expects it.** For example, scale the power, select OFDM symbols or subcarriers, and cut the channels into tokens. Each model has its own preprocessing, so ChaSE does not do it.
- **Run the frozen model.** Move the input to your device, put the model in evaluation mode (no dropout), and take the representation that you want to probe, for example the tokens of the last encoder layer.
- **Pool the representation into readouts.** Make one vector per channel for each readout of the config. For example, "mean" averages all tokens, and "spatial" averages the tokens of each antenna block and joins the results. The `arrays` in the manifest of step 1 give the antenna layout for such readouts.
- **Return a dict.** The keys are the readout names of the config, and each value is a real, finite [B, D] tensor or numpy array, with the same D in every batch. Extra keys are ignored.

The same channel must always give the same vector, so that the features do not depend on the batch.

```python
from chase.extract import extract
from chase.run import load_config

def encode(h):                    # h: torch complex64 [B, OFDM symbols, user antennas, BS antennas, subcarriers]
    z = my_model(preprocess(h))   # your model, for example tokens [B, n_tokens, D]
    return {"mean": z.mean(1), "spatial": pool_per_antenna(z)}   # [B, D] per readout

cfg = load_config("configs/my_eval.yaml")
# extract reads the channels of data_dir, calls encode on them and writes the features to features_dir
extract(encode, cfg, "my_model")                 # clean channels, to features_dir/my_model/<readout>/
extract(encode, cfg, "my_model", snr_db=10)      # AWGN at 10 dB added first, to features_dir/snr10/my_model/<readout>/
```

`scripts/extract_dummy.py` is a complete example with a dummy encoder. You do not pass the channels and you do not add noise. With `snr_db`, `extract` adds the noise to each batch before it calls `encode`. The noise of a channel is fixed by its set and its index, so every model and the raw baselines of step 4 see the same noisy channels.

**4. Probe.**

```bash
python scripts/run_chase.py configs/my_eval.yaml                # every model and readout
python scripts/run_chase.py configs/my_eval.yaml --snr-db 10    # the features at 10 dB
```

With `--snr-db`, the run reads the features from `features_dir/snr<X>`, adds the same noise to the raw baselines, and writes the results to `out_dir/snr<X>`. A stopped run resumes. It skips the quantities that a result of the same settings and features already holds.

**5. Report.**

```bash
python scripts/chase_report.py configs/my_eval.yaml [--snr-db 10]
```

### Outputs

The results go to `out_dir`, or `out_dir/snr<X>` with an SNR.

| File | Contents |
|---|---|
| `<model>/<readout>.json` | The scores of every quantity and probe (validation and test R^2, settings, controls), the config and the code version |
| `<model>/<readout>_pred.npz` | The test predictions of every probe, for the bootstrap of the report |
| `report.json` | Per model and quantity, r_lin, r_best and their 95% intervals, the levels and the lost share, and per group the means. Also the raw channel per quantity and a summary of the controls |
| `report.md` | Tables of the group means, the lost shares and the controls |

## Demonstration in the paper

Our paper demonstrates ChaSE on one data set and several models. The limits of this demonstration come from its data and its system, not from ChaSE.

**Data.** [CSIGen](https://github.com/BerkIGuler/CSIGen) generated the channels with [Sionna RT](https://github.com/NVlabs/sionna-rt) in eight scenes, four urban (Boston, Chicago, New York, San Francisco), two suburban (Irvine, Palo Alto) and two rural (Agua Dulce, Littlerock). The carrier is at 3.5 GHz, with 32 subcarriers at 30 kHz and 14 OFDM symbols. Each base station sector has a uniform planar array of 4 rows and 8 columns, that is 8 elements horizontally and 4 vertically. Each user has one isotropic antenna. One base station of each scene is a test cell.

Since the user has one antenna, the channels do not show the angles at the user, and the paper reports the other 20 quantities, omitting the 4 user-angle-related ones.

**Models.** Six published WCFMs, all pretrained on the same urban channels (PilotWiMAE, LWM, LWM-Temporal, WiMamba, WiMAE and ContraWiMAE), on clean and noisy channels and on the urban, suburban and rural test cells. Also PilotWiMAE pretrained on data sets that differ in one property at a time. The paper gives the results.

**Get the data.** The data of the demonstration are on Hugging Face at [BerkIGuler/ChaSEDatasets](https://huggingface.co/datasets/BerkIGuler/ChaSEDatasets) (5.6 GB). They hold the ray-traced paths of the eight scenes and one partition each for the urban, suburban and rural test cells. Download them with the `hf` command of `huggingface_hub` (`pip install huggingface_hub`), and build the sets of a cell.

```bash
hf download BerkIGuler/ChaSEDatasets --repo-type dataset --local-dir chase_data
python scripts/build_data.py chase_data/runs/urban chase_data/partitions/urban data/urban
```

The builder rebuilds the channels from the paths. It needs only numpy and the CPU, at about 10 to 20 seconds per file. The built urban, suburban and rural sets take about 17 GB, 5.4 GB and 8.2 GB.

**Settings of the paper.** The probe settings of the paper are the defaults of the config. In your config, also set `min_path_gain_db: -150` and `mean_power` to `statistics.mean_power` of the `manifest.yaml` that the builder writes to `data_dir`. The card of the data set lists these values for each cell.

## Repository layout

| Path | Contents |
|---|---|
| `chase/` | The Python package (quantities, probes, raw baselines, bootstrap, run, report, feature extraction, data builder) |
| `scripts/` | Command-line entry points |
| `configs/` | The example config |
| `tests/` | Tests (`pytest`) |
| `docs/` | `changelog.md` |

## Citation

The paper that introduces ChaSE will be on arXiv soon. Its citation follows here.

## License

This software is licensed under the **MIT License**. See [`LICENSE`](LICENSE). The software is provided **"as is"**, without warranty of any kind.
