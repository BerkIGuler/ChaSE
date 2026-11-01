# Changelog

All notable changes to ChaSE are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-31

**First release.** ChaSE probes the frozen representations of a wireless channel foundation model for 24 physical quantities of the link, with ground truth from ray-traced paths.

- **Quantities.** 24 quantities in five groups (link, delay, angles at the base station, motion, and angles at the user). The `quantities` key selects the quantities that the channels of a system show. The targets come from the path statistics of CSIGen. Quantities that span orders of magnitude are probed in log10, and the angles as the pair of their sine and cosine. An angle is scored with one R^2 from the squared errors pooled over its sine and cosine.
- **Probes.** A ridge probe with a closed-form fit for 17 penalties from 1e-2 to 1e6, and MLP probes with hidden widths [64], [256] and [256, 256], each with seeds 0, 1 and 2. The MLPs train with AdamW (learning rate 1e-3, mini-batches of 2,048, weight decay 1e-4 or 1e-2) and stop early on validation. The features are standardized with the train statistics. A feature column that is constant on the train links is only centered.
- **Choices on validation.** The validation links are split into two halves of whole 50 m tiles. Half A sets the ridge penalty, the weight decay and the stopping epoch. Half B chooses the readout of r_lin and the readout and probe of r_best. r_best is at least r_lin on the test links.
- **Levels and lost share.** Each quantity gets the available (r_lin), entangled (r_best - r_lin) and unread (1 - r_best) shares, with negative R^2 set to 0. The lost share is max(0, r_best of the raw channel - r_best of the model). The raw channel of a quantity is the raw baseline with the best validation score.
- **Raw baselines.** Four baselines of the raw channel. `complex` and `logmag` take the real and imaginary parts or log10 |h|^2 of the first OFDM symbol. `complex_t` and `logmag_t` take every OFDM symbol and every fourth subcarrier.
- **Uncertainty.** A bootstrap of 1,000 draws of whole 50 m test tiles, shared by every model and quantity, gives 95% intervals of r_lin, r_best, the group means and the lost share. The intervals of differences are paired.
- **Controls.** Every probe is fit again on shuffled train targets (and, for the MLPs, shuffled validation targets). The report summarizes the test R^2 of all controls.
- **Noise.** With `--snr-db`, every channel gets complex AWGN at that SNR relative to its own mean power. The noise of a channel depends only on its set and its index, so every model and raw baseline sees the same noisy channels.
- **Channel format.** The channels of a set are [n, OFDM symbol, user antenna, base-station antenna, subcarrier] complex64, so that systems with several antennas at the user are supported.
- **Model interface.** `chase.extract.extract` runs a user function `encode(h)`, which returns `{readout: [B, D]}`, on the channels of train, val and test, and writes the feature files. `scripts/extract_dummy.py` is an example.
- **Evaluation data.** `scripts/build_data.py` builds the sets of a cell from CSIGen runs and a partition. It rebuilds the channels from the saved paths, bit for bit as CSIGen computed them, with numpy only. Its `manifest.yaml` records the layout of the base-station and user arrays. The runs and partitions of the urban, suburban and rural test cells are on Hugging Face at BerkIGuler/ChaSEDatasets.
- **Run and report.** `scripts/run_chase.py` probes every model, readout and quantity of a config and resumes a stopped run. `scripts/chase_report.py` writes `report.json` and `report.md`. The result files and the report record their config and the version and git commit of the code.
- **Config.** One YAML file per evaluation. The keys are `data_dir`, `features_dir`, `out_dir`, `models`, `min_path_gain_db`, `mean_power`, `quantities`, `levels`, `seeds`, `control_seed`, `controls`, `val_split_seed` and `report` (`raw_baselines`, `n_boot`, `boot_seed`). `configs/example.yaml` shows the defaults.
- **Environment.** Python 3.12 or later, PyTorch 2.4 or later, numpy 2.1 or later and PyYAML 6.0 or later.
