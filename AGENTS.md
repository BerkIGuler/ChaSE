# AGENTS.md

This file gives pointers for people and coding agents that use ChaSE or contribute to it. The [README](README.md) explains what ChaSE measures and how to run it. This file does not repeat the README. It tells you where to look and which conventions to follow.

## Where to start

| Task | Read |
|---|---|
| Understand what ChaSE measures | README, sections "What ChaSE measures" and "Demonstration in the paper" |
| Run ChaSE on a model for the first time | `examples/quickstart.ipynb`, then README section "Usage" |
| Write the data files from your own simulator or data set | `docs/data_format.md` |
| Connect your own model | README step 3 and `examples/extract_dummy.py` |
| Find the settings of the config and their defaults | `configs/example.yaml` |
| Find the changes between versions | `docs/changelog.md` |

## Set up and test

ChaSE needs Python 3.12 or later. Install PyTorch for your platform first, then ChaSE with its test extra.

```bash
pip install -e ".[test]"
pytest
```

The tests run on the CPU in under a minute and need no data download. The CI (`.github/workflows/tests.yml`) runs `pytest` on Python 3.12 with the CPU wheel of PyTorch, for every pull request and every push to `main`.

## How the code is organized

| Module | Contents |
|---|---|
| `chase/quantities.py` | The 24 quantities, their groups, the label fields that each one reads (`FIELDS`) and the function that makes its target (`TARGETS`) |
| `chase/probes.py` | The ridge probe and the MLP probes |
| `chase/raw.py` | The raw baselines |
| `chase/extract.py` | `extract`, which calls the `encode` function of a model and writes the feature files |
| `chase/run.py` | The config (`DEFAULTS`, `load_config`) and the run that trains and scores the probes |
| `chase/bootstrap.py` | The tile bootstrap |
| `chase/report.py` | The shares, the intervals and the report |
| `chase/partition.py` | The split of CSIGen runs into train, val and test |
| `chase/data.py`, `chase/cfr.py` | The builder of the sets from CSIGen runs, and the rebuild of the channels from the saved paths |
| `chase/provenance.py` | The version, git commit and checksums that every result file records |
| `chase/cli.py` | The four commands that `pyproject.toml` installs |

## Rules for changes

These rules keep the results of ChaSE comparable between models and between versions.

- **Keep ChaSE independent of models.** A model enters ChaSE only through its `encode` function and the feature files. Do not add code for a specific model to `chase/`.
- **Keep ChaSE independent of the data source.** ChaSE reads the files of `docs/data_format.md`. The CSIGen tools (`chase/partition.py`, `chase/data.py`, `chase/cfr.py`) only write these files. ChaSE does not import CSIGen or Sionna RT.
- **Keep the defaults.** The defaults in `chase/run.py` and `configs/example.yaml` are the settings of the paper. A change of a default changes the scores of every config that leaves out that key. Discuss such a change in an issue first.
- **Keep the scores stable.** If a change can change a score, a level or an interval, say so in the PR and show the size of the change. A refactor or a speed-up must give the same scores.
- **Keep the provenance.** Every result file records its config and the git commit of the code. Keep this when you add a result file.
- **Keep the CFR exact.** `chase/cfr.py` rebuilds the channels bit for bit as CSIGen computes them. `tests/test_cfr.py` checks this.
- **Add a quantity in all places.** A new quantity needs an entry in `GROUPS`, `FIELDS` and `TARGETS` of `chase/quantities.py` (`QUANTITIES` follows from `GROUPS`), a row in the table of `docs/data_format.md`, a row in the table of the README and a test in `tests/test_quantities.py`.

## Code style

- Put module-level constants at the top of each file, after the imports.
- Add a test in `tests/` with each new module or behavior, and run `pytest` before you open a pull request.
- Do not commit data sets, features, predictions or checkpoints. Configs refer to them by path.
- Match the style of the code around your change. Change only the lines that your change needs.

## Pull requests

- Open an issue first for a large change, a new quantity or a change of a default.
- Name the branch `feat/<topic>`, `fix/<topic>` or `docs/<topic>`.
- Write commit messages with a short imperative summary line, then a few lines on what changed and why.
- One pull request covers one topic. Use a short imperative title.
- Fill in `.github/pull_request_template.md` (Why, Where, How). In "How", say how you checked the change, for example "pytest passes".
- Do not change the version in `pyproject.toml` or `docs/changelog.md`. The maintainer makes each release as its own pull request, and writes the changelog from the descriptions of the merged pull requests. Therefore, describe in your pull request what a user of ChaSE can notice, for example a change of the scores, a new option or a new output file.

## Writing style

The docs, docstrings, comments, commit messages and pull requests use simple technical English.

- Write short sentences, with one idea per sentence, in the active voice and the present tense.
- Use simple words and American English.
