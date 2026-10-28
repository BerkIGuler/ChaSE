# ChaSE

**Channel Semantics Evaluation.** ChaSE finds what a wireless channel foundation model (WCFM) knows about the physics of a channel. It probes the frozen representations of the model for physical quantities of the link, such as the delay spread, the angles of departure and arrival, and the Doppler shift. The ground truth comes from ray-traced paths.

> **Status.** This repository is under construction. The code, the probe targets and the evaluation data will be added before the first release.

## What ChaSE measures

For each quantity, ChaSE trains a linear probe and MLP probes on the representations of the model. It then splits the quantity into four shares.

- **Available.** A linear probe reads it.
- **Entangled.** Only a nonlinear probe reads it.
- **Unread.** No probe reads it.
- **Lost.** A probe on the raw channel reads it, but no probe on the representation does.

The quantities are in five groups (link, delays, departure angles, arrival angles, motion). A bootstrap over spatial tiles of the test links gives the uncertainty, because links in the same area are correlated.

## Requirements

To be written.

## Usage

To be written.

## Repository layout

To be written.

## Citation

To be written when the paper is on arXiv.

## License

This software is licensed under the **MIT License**. See [`LICENSE`](LICENSE). The software is provided **"as is"**, without warranty of any kind.
