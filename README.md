# ChaSE

**Channel Semantics Evaluation.** ChaSE queries what a wireless channel foundation model (WCFM) knows about the physics of a channel. It probes the frozen representations of a model for 24 physical quantities of the propagation. These quantities include fundamental channel statistics, such as the delay spread and the Doppler shift, and other quantities, such as the distance between the base station and the user, the speed of the user, and the number of interactions of the strongest path.

ChaSE does not depend on a model or on a data set. It needs only representations of channels with the ground truth of the 24 quantities, split into train, validation and test sets. The section [Demonstration in the paper](#demonstration-in-the-paper) shows how our paper applies ChaSE to several models across different data sets.

## What ChaSE measures

ChaSE probes 24 quantities in five groups.

| Group | Quantities |
|---|---|
| `link` | `distance`, `path_gain`, `k_factor`, `los_share`, `diffraction_share`, `specular_share`, `refraction_share`, `dominant_share`, `dominant_bounces` |
| `delay` | `delay_spread`, `first_delay`, `excess_delay` |
| `bs_angles` | `aod`, `zod`, `asd`, `zsd` (the angles of departure at the base station) |
| `motion` | `doppler`, `abs_doppler`, `max_doppler`, `speed` |
| `user_angles` | `aoa`, `zoa`, `asa`, `zsa` (the angles of arrival at the user) |

The angles of the strongest path are probed as the pair of their sine and cosine. The distance and the delay spread span orders of magnitude, so they are probed in log scale. The `quantities` key of the config (step 2 of [Usage](#usage)) selects the quantities to probe.

A model encodes each channel into tokens. A readout pools these tokens into one vector per channel, for example the mean over all tokens. In our demonstration, we use three readouts, which the paper explains. For each quantity, ChaSE trains a linear (ridge) probe and MLP probes of three sizes on the vectors of each readout. It then splits a model's understanding of every quantity into four shares.

- **Available.** A linear probe reads it. This is r_lin, the test R^2 of the ridge probe.
- **Entangled.** Only a nonlinear probe reads it. This is r_best - r_lin, where r_best is the test R^2 of the best probe.
- **Unread.** No probe reads it. This is 1 - r_best.
- **Lost.** A probe on the raw channel reads it, but no probe on the representation does. This is max(0, r_best of the raw channel - r_best of the model's representation). For each quantity, the raw channel is the raw baseline with the best validation score, among the raw baselines of the config (also described in the paper). You list these baselines in the config (step 2 of [Usage](#usage)), and ChaSE computes them from the channels themselves in step 4.

These shares are not exact information-theoretic quantities. The probes give lower bounds on the information that a representation holds, because a better probe (for example, a genie probe) could read more.

The readout and the probe are chosen on the validation links. A bootstrap gives a 95% interval for each score. It draws whole tiles of test links, where a tile holds the links of the users in one square of 50 m by 50 m of a scene, because links in the same area are correlated. The links of users far apart are much less correlated.

As a control, ChaSE shuffles the targets of the train links in a random order, so that each channel gets the target of another channel. For the MLP probes, it also shuffles the targets of the validation links. It then fits the same probes on the shuffled targets and scores them on the real test targets. A control R^2 near zero shows that a probe cannot fit arbitrary targets, so that its score comes from the representation.

## Requirements

- Python 3.12 or later.
- PyTorch 2.4 or later. A GPU makes the MLP probes much faster, but ChaSE also runs on the CPU.
- numpy and PyYAML.

ChaSE uses a GPU or a CPU in two steps, and you choose the device of each step separately.

- **Features (step 3 of [Usage](#usage)).** Your `encode` function runs your model, so it chooses the device. Move the channels and the model to the same device in `encode`.
- **Probes (step 4 of [Usage](#usage)).** `scripts/run_chase.py` trains the probes on the GPU if PyTorch finds one, and on the CPU otherwise. To choose the device yourself, use `--device`, for example `--device cpu` or `--device cuda:1`.

The two steps can use different devices, because step 3 writes the features to files, and step 4 reads them from these files.

Install PyTorch for your platform first, then ChaSE.

```bash
git clone https://github.com/BerkIGuler/ChaSE.git
cd ChaSE
pip install -e .            # add [test], as in pip install -e ".[test]", to also install pytest for the tests in tests/
```

## Usage

An evaluation has five steps. One YAML config file, which you write in step 2, holds all settings of the evaluation. Steps 3, 4 and 5 read it. `configs/example.yaml` is a template for this file. It shows every key with its default, and these defaults are the settings of the paper.

**1. Prepare the data.** ChaSE works with channels from any simulator or data set, if you have the ground truth of each channel. ChaSE needs two files for each of the train, val and test sets.

- `<set>.npy`, the channels [n, OFDM symbol, user antenna, base-station antenna, subcarrier] complex64, without noise. ChaSE adds the noise itself when you ask for an SNR (step 3).
- `<set>_labels.npz`, the ground truth of each channel, with one array per field and one row per channel. `chase/quantities.py` lists the fields that the probe targets read, such as `delay_spread_s` and `dominant_phi_t`. The bootstrap also reads `city` and `rx_position`.

We generate our data with [CSIGen](https://github.com/BerkIGuler/CSIGen), our open-source tool built on [Sionna RT](https://github.com/NVlabs/sionna-rt). ChaSE does not need CSIGen, but ChaSE reads CSIGen runs directly. Two commands turn CSIGen runs into the two files of each set (six files in total) and a `manifest.yaml`.

*Partition the links.* The first command splits the links of your runs into train, val and test.

```bash
python scripts/make_partition.py configs/my_split.yaml runs/city_a runs/city_b partition/my_cell
```

- `configs/my_split.yaml` is your split config. Copy `configs/split_example.yaml`, which explains each setting, and give the held-out site of each of your runs. Sites are counted from 0. Site s holds the files `channel_tx_<i>.npz` with i // num_sectors = s, so with 6 sectors per site, site 1 holds the files 6 to 11.
- `runs/city_a runs/city_b` are the CSIGen runs, one folder per city or town. All runs must be in one folder (here `runs`), because the second command reads them from there.
- `partition/my_cell` is the output folder. It gets `partition.npz`, with the part of every link, and a `manifest.yaml`, with the split config and the counts per run and part.

The command splits the links as in the paper. In each run, it holds out one base-station site as the test cell. The links of this site to its own users are the test set. The other links of this site, and all links of its users to other sites, are left out. It puts the users of whole 50 m blocks of the other sites into validation, and it leaves out the users within 10 m of them. All other links are train links. The command reads only the metadata of the runs, so it takes seconds. It stops if the output folder already has a partition.

*Build the sets.* The second command writes the six files of the cell.

```bash
python scripts/build_data.py runs partition/my_cell data/my_cell
```

- `runs` is the folder that holds the runs of the partition.
- `partition/my_cell` is the partition folder of the first command.
- `data/my_cell` is the output folder. Use it as `data_dir` in the config of step 2.

The command rebuilds the channels from the paths that CSIGen saved, bit for bit as CSIGen computed them. It needs only numpy and the CPU, at about 10 to 20 seconds per file. It keeps the train, val and test links that go to the site of their own user, and it writes their channels and labels. It also writes a `manifest.yaml` with the mean power for the raw baselines and the layout of the base-station and user arrays (rows, columns, polarizations, spacing and the order of the antenna index). An encode function can use this layout for its readouts. The mean power is the mean of |h|^2 over all train links, as in the paper, unless the partition already gives one. The command stops if the output folder already has a complete build.

**2. Set up the config.** Copy `configs/example.yaml`, for example to `configs/my_eval.yaml`, and set the data, the folders and the models. The keys that you leave out take their defaults. The split config of step 1 is a separate file, which only `scripts/make_partition.py` reads.

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

- **Take the channels as ChaSE gives them.** The input `h` is a CPU tensor [B, OFDM symbol, user antenna, base-station antenna, subcarrier] complex64, in the units of the data, without normalization. Do not add noise in `encode`. To evaluate at an SNR, call `extract` with `snr_db`, as in the example below. `extract` then adds the noise to `h` before it calls `encode`.
- **Prepare the input as the model expects it.** For example, scale the power, select OFDM symbols or subcarriers, and cut the channels into tokens. Each model has its own preprocessing, so ChaSE does not do it.
- **Run the frozen model.** Move the input to your device, put the model in evaluation mode (no dropout), and take the representation that you want to probe, for example the tokens of the last encoder layer.
- **Pool the representation into readouts.** Make one vector per channel for each readout of the config. For example, "mean" averages all tokens, and "spatial" averages the tokens of each antenna block and joins the results. The `arrays` in the manifest of step 1 give the antenna layout for such readouts.
- **Return a dict.** The keys are the readout names of the config. Extra keys are ignored. Each value is a real, finite [B, D] tensor or numpy array. B is the number of channels in `h`, with one row per channel in the order of `h`. B is usually 512, but the last batch of a set is smaller. D is the length of the vector of the readout. Each readout can have its own D, for example 128 for "mean" and 1024 for "spatial", but the D of a readout must be the same in every batch.

The same channel must always give the same vector, so that the features do not depend on the batch. `extract` checks this before it writes any file. It calls `encode` twice on the first batch of the train set, and once on up to 8 of its channels alone. If a channel gets another vector, `extract` stops with a message. The usual cause is a model in training mode, with dropout or BatchNorm. `extract(..., check=False)` turns the check off.

```python
from chase.extract import extract
from chase.run import load_config

def encode(h):                    # h: torch complex64 [B, OFDM symbols, user antennas, BS antennas, subcarriers]
    z = my_model(preprocess(h))   # your model and your preprocessing, for example tokens [B, n_tokens, D]
    return {"mean": z.mean(1), "spatial": pool_per_antenna(z)}   # [B, D] per readout, with your pooling

cfg = load_config("configs/my_eval.yaml")
# extract reads the channels of <data_dir>, calls encode on them and writes the features to <features_dir>
extract(encode, cfg, "my_model")                 # clean channels, to <features_dir>/my_model/<readout>/
extract(encode, cfg, "my_model", snr_db=10)      # AWGN at 10 dB added first, to <features_dir>/snr10/my_model/<readout>/
```

`scripts/extract_dummy.py` is a complete example with a dummy encoder. The noise that `extract` adds to a channel depends only on the set and the index of the channel. Thus, every model and the raw baselines of step 4 see exactly the same noisy channels.

**4. Probe.** This step trains and scores all probes for every model, readout and quantity of the config, and writes one result file per model and readout.

```bash
python scripts/run_chase.py configs/my_eval.yaml                # every model and readout
python scripts/run_chase.py configs/my_eval.yaml --snr-db 10    # the features at 10 dB
```

With `--snr-db`, the run reads the features from `<features_dir>/snr<X>`, adds the same noise to the raw baselines, and writes the results to `<out_dir>/snr<X>`, where X is the SNR in dB. It stops if `chase.extract` wrote the features at another SNR. A stopped run resumes. It skips the quantities that a result of the same settings and features already holds.

**5. Report.** This step chooses the readout and the probe of each quantity on the validation links, computes the shares and their 95% intervals, and writes `report.json` and `report.md`.

```bash
python scripts/chase_report.py configs/my_eval.yaml [--snr-db 10]
```

### Outputs

The results go to `<out_dir>`, or to `<out_dir>/snr<X>` with an SNR of X dB, for example `<out_dir>/snr10` for `--snr-db 10`.

| File | Contents |
|---|---|
| `<model>/<readout>.json` | The scores of every quantity and probe (validation and test R^2, settings, controls), the config and the code version |
| `<model>/<readout>_pred.npz` | The test predictions of every probe, for the bootstrap of the report |
| `report.json` | Per model and quantity, r_lin, r_best and their 95% intervals, the four shares, and per group the means. Also the raw channel per quantity and a summary of the controls |
| `report.md` | Tables of the group means, the lost shares and the controls |

## Demonstration in the paper

Our paper demonstrates ChaSE on one data set and several models. The limits of this demonstration come from its data and its system, not from ChaSE.

**Data.** [CSIGen](https://github.com/BerkIGuler/CSIGen) generated the channels with [Sionna RT](https://github.com/NVlabs/sionna-rt) in eight scenes, four urban (Boston, Chicago, New York, San Francisco), two suburban (Irvine, Palo Alto) and two rural (Agua Dulce, Littlerock). The carrier is at 3.5 GHz, with 32 subcarriers at 30 kHz and 14 OFDM symbols. Each base station sector has a uniform planar array of 4 rows and 8 columns, that is 8 elements horizontally and 4 vertically. Each user has one isotropic antenna. One base station of each scene is a test cell.

Since the user has one antenna, the channels do not show the angles at the user, and the paper reports the other 20 quantities, omitting the 4 user-angle-related ones.

**Models.** The paper evaluates six published WCFMs (PilotWiMAE, LWM, LWM-Temporal, WiMamba, WiMAE and ContraWiMAE). All six are pretrained on the same urban channels. The paper evaluates them on clean and noisy channels, and on the urban, suburban and rural test cells. It also evaluates PilotWiMAE pretrained on data sets that differ in one property at a time. The paper gives the results.

**Get the data.** The data of the demonstration are on Hugging Face at [BerkIGuler/ChaSEDatasets](https://huggingface.co/datasets/BerkIGuler/ChaSEDatasets) (5.6 GB). They hold the ray-traced paths of the eight scenes and one partition each for the urban, suburban and rural test cells. The partitions are included, so you skip the first command of step 1 (`scripts/make_partition.py`) and run only the second one, `scripts/build_data.py`. The partitions also hold the mean power of the paper, so the built cells give the raw baselines of the paper. Download the data with the `hf` command of `huggingface_hub` (`pip install huggingface_hub`), and build the sets of a cell.

```bash
hf download BerkIGuler/ChaSEDatasets --repo-type dataset --local-dir chase_data
python scripts/build_data.py chase_data/runs/urban chase_data/partitions/urban data/urban
```

For the suburban and rural cells, replace `urban` with `suburban` or `rural`. The builder rebuilds the channels from the paths. It needs only numpy and the CPU, at about 10 to 20 seconds per file. The built urban, suburban and rural sets take about 17 GB, 5.4 GB and 8.2 GB.

**Settings of the paper.** The settings of the paper are the defaults of the config, which `configs/example.yaml` shows. A config thus needs only the data, the folders and the models. The default `min_path_gain_db` of -150 dB, which we used for the paper, probes only the links with at least this path gain. Set another value for data with another link budget, or `null` to keep every link. When the config leaves `mean_power` empty, the raw baselines take the mean power of the paper from the `manifest.yaml` that the builder writes to `<data_dir>`. Without that manifest, they take the mean power of the train channels. The run prints which one it uses, and the card of the data set lists the values of each cell.

## Repository layout

| Path | Contents |
|---|---|
| `chase/` | The Python package (quantities, probes, raw baselines, bootstrap, run, report, feature extraction, partition, data builder) |
| `scripts/` | Command-line entry points |
| `configs/` | The example config and the example split config |
| `tests/` | Tests (`pytest`) |
| `docs/` | `changelog.md` |

## Citation

The paper that introduces ChaSE will be on arXiv soon. Its citation follows here.

## License

This software is licensed under the **MIT License**. See [`LICENSE`](LICENSE). The software is provided **"as is"**, without warranty of any kind.
