# ChaSE

**Channel Semantics Evaluation.** ChaSE finds what a wireless channel foundation model (WCFM) knows about the physics of a channel. It probes the frozen representations of the model for physical quantities of the link, such as the delay spread, the angles of departure and arrival, and the Doppler shift. The ground truth comes from ray-traced paths.

## What ChaSE measures

For each quantity, ChaSE trains a ridge probe and MLP probes on the representations of the model. It then splits the quantity into four shares.

- **Available.** A linear probe reads it. This is r_lin, the test R^2 of the ridge probe.
- **Entangled.** Only a nonlinear probe reads it. This is r_best - r_lin, where r_best is the test R^2 of the best probe.
- **Unread.** No probe reads it. This is 1 - r_best.
- **Lost.** A probe on the raw channel reads it, but no probe on the representation does. This is max(0, r_best of the raw channel - r_best of the model).

The readout and the probe are chosen on validation links. A bootstrap over 50 m tiles of the test links gives a 95% interval for each score, because links in the same area are correlated. Shuffled-target controls show whether a probe can fit arbitrary targets.

ChaSE probes 24 quantities in five groups.

| Group | Quantities |
|---|---|
| `link` | `distance`, `path_gain`, `k_factor`, `los_share`, `diffraction_share`, `specular_share`, `refraction_share`, `dominant_share`, `dominant_bounces` |
| `delay` | `delay_spread`, `first_delay`, `excess_delay` |
| `bs_angles` | `aod`, `zod`, `asd`, `zsd` |
| `motion` | `doppler`, `abs_doppler`, `max_doppler`, `speed` |
| `user_angles` | `aoa`, `zoa`, `asa`, `zsa` |

The angles of the strongest path are probed as the pair of their sine and cosine. The quantities that span orders of magnitude are probed in log10. With one antenna at the user, the channel does not show the angles at the user, so the `user_angles` group is a control.

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

## Data

The evaluation data are on Hugging Face at [BerkIGuler/ChaSEDatasets](https://huggingface.co/datasets/BerkIGuler/ChaSEDatasets) (5.6 GB). They hold the ray-traced paths of eight scenes from [CSIGen](https://github.com/BerkIGuler/CSIGen), and one partition each for an urban, a suburban and a rural test cell. Download them with the `hf` command of `huggingface_hub` (`pip install huggingface_hub`), and build the sets of a cell.

```bash
hf download BerkIGuler/ChaSEDatasets --repo-type dataset --local-dir chase_data
python scripts/build_data.py chase_data/runs/urban chase_data/partitions/urban data/urban
```

The builder rebuilds the channels from the paths, bit for bit as CSIGen computed them. It needs only numpy and the CPU, at about 10 to 20 seconds per file. It writes, for each of train, val and test, the channels `<set>.npy` ([n, OFDM symbol, antenna, subcarrier] complex64) and the labels `<set>_labels.npz`. The built urban, suburban and rural sets take about 17 GB, 5.4 GB and 8.2 GB.

You can also use your own CSIGen data. ChaSE needs only these two files per set (`chase/quantities.py` lists the label fields).

## Usage

An evaluation has four steps. One config holds all the settings. `configs/example.yaml` shows every key with its default. The defaults are the probe settings of the paper. Set `min_path_gain_db` and `mean_power` per cell, as the card of the data set shows.

**1. Set up the config.** Give the data, the folders and the models.

```yaml
data_dir: data/urban
features_dir: features/urban
out_dir: results/urban
models:
  raw: {raw: complex}                    # the raw-channel baselines, computed by ChaSE
  raw_logmag: {raw: logmag}
  raw_t: {raw: complex_t}
  raw_logmag_t: {raw: logmag_t}
  my_model: {readouts: [mean, spatial]}  # your model and its readouts
min_path_gain_db: -150
mean_power: 3.858271428867973e-11        # see the card of the data set for each cell
```

**2. Write the features of your model.** Give ChaSE a function that encodes a batch of channels and returns one vector per channel for each readout. ChaSE calls it on every set and writes the feature files. The function does the preprocessing, the encoding and the pooling of its model.

```python
from chase.extract import extract
from chase.run import load_config

def encode(h):                    # h: torch complex64 [B, OFDM symbols, antennas, subcarriers]
    z = my_model(preprocess(h))   # your model, for example tokens [B, n_tokens, D]
    return {"mean": z.mean(1), "spatial": pool_per_antenna(z)}   # [B, D] per readout

cfg = load_config("configs/my_eval.yaml")
extract(encode, cfg, "my_model")                 # clean channels
extract(encode, cfg, "my_model", snr_db=10)      # channels with AWGN at 10 dB
```

`scripts/extract_dummy.py` is a complete example with a dummy encoder. With an SNR, ChaSE adds the same noise to the channels of every model and of the raw baselines.

**3. Probe.**

```bash
python scripts/run_chase.py configs/my_eval.yaml                # every model and readout
python scripts/run_chase.py configs/my_eval.yaml --snr-db 10    # the features at 10 dB
```

A stopped run resumes. It skips the quantities that a result of the same settings and features already holds.

**4. Report.**

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
