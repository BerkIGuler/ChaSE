"""The ChaSE report, from the results of the run. The readouts and probes are chosen on validation links.

- Per model and quantity, the readout of r_lin and the readout and probe of r_best are chosen on validation half B.
  The report gives their test R^2 with a 95% interval and the levels (available, entangled, unread). r_best is at
  least r_lin, in the estimate and in every bootstrap draw. A choice on validation can score lower on test, so r_best
  takes the larger test R^2 of the chosen probe and the ridge probe of r_lin, as in the paper.
- Per model and group, the means of r_lin, r_best and the levels, and the interval of the mean of r_best.
- The raw channel. Per quantity, the raw baseline with the best validation score of r_best, and the one with the best
  validation score of r_lin.
- The lost share per model and quantity, max(0, r_best of the raw channel - r_best of the model), with a paired 95%
  interval of the difference.
- The shuffled-target controls of every fit.

Negative R^2 values count as 0 in the group means and the differences, in the estimate and in every draw. MLP scores
are means over the probe seeds. The intervals come from one bootstrap of whole tiles of the test links, shared by
every model and quantity, so that differences between models and means over a group are paired.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from chase.bootstrap import ci95, levels, r2_draws, tile_draws, tiles
from chase.provenance import code_state
from chase.quantities import ANGLES, GROUPS
from chase.run import SETS, load_predictions, output_dir, readouts_of, result_paths, run_settings, score_of

SHARES = ("r_lin", "r_best", "available", "entangled", "unread")


def choose(entries: dict, q: str) -> dict:
    """The choices of quantity q on validation half B, over the readouts in entries. r_lin takes the readout with the
    best ridge probe. r_best takes the best readout and probe (ridge or an MLP level). If the ridge probe of r_lin
    scores higher on test than the probe chosen for r_best, it gives r_best. The choice on validation stays as
    readout_val and probe_val."""
    lin = {r: e["targets"][q]["linear"] for r, e in entries.items()}
    r_lin = max(lin, key=lambda r: lin[r]["val"])
    cands = [(lin[r]["val"], r, "linear", lin[r]["test"]) for r in lin]
    for r, e in entries.items():
        for lv, m in e["targets"][q]["mlp"].items():
            cands.append((m["val_mean"], r, lv, m["test_mean"]))
    v, rb, pb, tb = max(cands)
    out = {"readout_lin": r_lin, "r_lin": lin[r_lin]["test"], "val_lin": lin[r_lin]["val"],
           "readout_best": rb, "probe_best": pb, "r_best": tb, "val_best": v, "readout_val": rb, "probe_val": pb}
    if lin[r_lin]["test"] > tb:
        out.update(readout_best=r_lin, probe_best="linear", r_best=lin[r_lin]["test"])
    return out


def _z(x):
    """R^2 with negative values set to 0."""
    return np.clip(x, 0.0, None)


def make_report(cfg: dict, config_file: str = None, snr_db: float = None) -> dict:
    data_dir = Path(cfg["data_dir"]).expanduser()
    res_dir = output_dir(cfg, snr_db)
    rcfg = cfg["report"]
    quantities = cfg["quantities"]
    settings = run_settings(cfg, snr_db)
    lab = dict(np.load(data_dir / f"{SETS[2]}_labels.npz"))
    gtile = tiles(lab)
    draws = tile_draws(gtile, int(rcfg["n_boot"]), int(rcfg["boot_seed"]))
    report = {"config": cfg, "config_file": config_file, "code": code_state(), "snr_db": snr_db,
              "created": datetime.now().isoformat(timespec="seconds"), "quantities": quantities,
              "groups": GROUPS, "n_tiles": int(draws.shape[1]), "models": {}}

    # the results and predictions of the settings of the config
    entries, preds, incomplete = {}, {}, {}
    for model, spec in cfg["models"].items():
        ent = {}
        for r in readouts_of(spec):
            f, pf = result_paths(res_dir, model, r)
            if not f.exists() or not pf.exists():
                continue
            e = json.loads(f.read_text())
            arrays, pid = load_predictions(pf)
            if e["identity"]["settings"] != settings or pid != e["identity"]:
                print(f"{model}/{r}: a run of other settings, left out", flush=True)
                continue
            ent[r], preds[(model, r)] = e, arrays
        missing = [q for q in quantities if len(ent) < len(readouts_of(spec))
                   or any(q not in e["targets"] or e["targets"][q]["r2"] != score_of(q) for e in ent.values())]
        if missing:
            incomplete[model] = missing
            print(f"{model}: incomplete, without {missing}", flush=True)
        if ent:
            entries[model] = (ent, [q for q in quantities if q not in missing])
    report["incomplete"] = incomplete

    cache = {}

    def boot(model, readout, probe, q):
        """R^2 in every bootstrap draw. For an MLP level, the mean over the seeds in each draw."""
        key = (model, readout, probe, q)
        if key not in cache:
            p = preds[(model, readout)]
            arrs = [p[f"{q}__linear"]] if probe == "linear" else [p[f"{q}__{probe}_s{s}"] for s in cfg["seeds"]]
            t = gtile[p[f"{q}__idx"]]
            cache[key] = np.mean([r2_draws(p[f"{q}__true"], a, t, draws, q in ANGLES) for a in arrs], 0)
        return cache[key]

    def best_draws(model, c, q):
        """r_best in every draw, the larger of the probe chosen on validation and the ridge probe of r_lin."""
        return np.fmax(boot(model, c["readout_val"], c["probe_val"], q), boot(model, c["readout_lin"], "linear", q))

    # per model and quantity, the choices, intervals and levels, and the group means
    sel = {}
    for model, (ent, qs_done) in entries.items():
        per_q = {}
        for q in qs_done:
            c = choose(ent, q)
            c["ci_lin"] = ci95(boot(model, c["readout_lin"], "linear", q))
            c["ci_best"] = ci95(best_draws(model, c, q))
            c.update(levels(c["r_lin"], c["r_best"]))
            per_q[q] = sel[(model, q)] = c
        groups = {}
        for g, qs in GROUPS.items():
            if all(q in per_q for q in qs):
                groups[g] = {k: float(np.mean([max(per_q[q][k], 0.0) for q in qs])) for k in SHARES}
                groups[g]["ci_r_best"] = ci95(np.mean([_z(best_draws(model, per_q[q], q)) for q in qs], 0))
        report["models"][model] = {"readouts": list(ent), "quantities": per_q, "groups": groups}

    # the raw channel is the raw baseline chosen on validation per quantity, for r_best and for r_lin
    raws = [m for m in rcfg["raw_baselines"] if m in entries]
    refs = {q: max((sel[(m, q)]["val_best"], m) for m in raws if (m, q) in sel)[1]
            for q in quantities if any((m, q) in sel for m in raws)}
    lin_refs = {q: max((sel[(m, q)]["val_lin"], m) for m in raws if (m, q) in sel)[1] for q in refs}
    report["raw_channel"] = {q: {"model_best": refs[q], "r_best": sel[(refs[q], q)]["r_best"],
                                 "model_lin": lin_refs[q], "r_lin": sel[(lin_refs[q], q)]["r_lin"]} for q in refs}
    report["raw_channel_groups"] = {
        g: {"r_best": float(np.mean([max(sel[(refs[q], q)]["r_best"], 0.0) for q in qs])),
            "ci_r_best": ci95(np.mean([_z(best_draws(refs[q], sel[(refs[q], q)], q)) for q in qs], 0)),
            "r_lin": float(np.mean([max(sel[(lin_refs[q], q)]["r_lin"], 0.0) for q in qs]))}
        for g, qs in GROUPS.items() if all(q in refs for q in qs)}

    # the lost share of each model against the raw channel
    for model in entries:
        if model in rcfg["raw_baselines"]:
            continue
        lost = {}
        for q, ref in refs.items():
            if (model, q) not in sel:
                continue
            c, cr = sel[(model, q)], sel[(ref, q)]
            diff = max(cr["r_best"], 0.0) - max(c["r_best"], 0.0)
            d = _z(best_draws(ref, cr, q)) - _z(best_draws(model, c, q))
            lost[q] = {"lost": max(0.0, diff), "diff": diff, "ci_diff": ci95(d)}
        report["models"][model]["lost"] = lost

    # the shuffled-target controls of every fit
    ctl = {"linear": [], **{lv: [] for lv in cfg["levels"]}}
    for ent, _ in entries.values():
        for e in ent.values():
            for t in e["targets"].values():
                if t["linear"]["control"] is not None:
                    ctl["linear"].append(t["linear"]["control"])
                for lv, m in t["mlp"].items():
                    if m["control"] is not None:
                        ctl[lv].append(m["control"])
    report["controls"] = {k: {"n": len(v), **dict(zip(("min", "median", "q95", "max"),
                                                      map(float, np.quantile(v, [0, .5, .95, 1]))))}
                          for k, v in ctl.items() if v}
    report["complete"] = not incomplete and set(entries) == set(cfg["models"])
    return report


def markdown(report: dict) -> str:
    """Summary tables of the report, with the group means, the raw channel, the lost shares and the controls."""
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    gnames = list(GROUPS)
    L = [f"# ChaSE report ({report['created']})", "",
         "Each cell gives r_lin / r_best, the mean over the quantities of the group. The test R^2 is that of the "
         "choice on validation, with r_best at least r_lin.",
         f"Complete: {report['complete']}. Intervals from a bootstrap of {report['n_tiles']} test tiles.", ""]
    if report["snr_db"] is not None:
        L += [f"SNR {report['snr_db']:g} dB.", ""]
    L += ["## Group means", "", "| Model | " + " | ".join(gnames) + " |", "|---" * (len(gnames) + 1) + "|"]
    rows = [(m, v["groups"]) for m, v in report["models"].items()]
    rows.append(("raw channel", report["raw_channel_groups"]))
    for m, g in rows:
        L.append(f"| {m} | " + " | ".join(f"{f2(g[k]['r_lin'])} / {f2(g[k]['r_best'])}" if k in g else "-"
                                          for k in gnames) + " |")
    L += ["", "## Raw channel per quantity (baseline of r_best, chosen on validation)", "",
          ", ".join(f"{q} {v['model_best']} ({f2(v['r_best'])})" for q, v in report["raw_channel"].items()), "",
          "## Lost share, max(0, r_best(raw) - r_best) [95% interval of r_best(raw) - r_best]", ""]
    lost_rows = {m: v["lost"] for m, v in report["models"].items() if v.get("lost")}
    for g, qs in GROUPS.items():
        L += [f"### {g}", "", "| Model | " + " | ".join(qs) + " |", "|---" * (len(qs) + 1) + "|"]
        for m, lost in lost_rows.items():
            L.append(f"| {m} | " + " | ".join(
                f"{f2(lost[q]['lost'])} [{f2(lost[q]['ci_diff'][0])}, {f2(lost[q]['ci_diff'][1])}]"
                if q in lost and lost[q]["ci_diff"][0] is not None else "-" for q in qs) + " |")
        L.append("")
    L += ["## Shuffled-target controls (test R^2 of every fit)", "",
          "| Probe | n | min | median | 95% | max |", "|---|---|---|---|---|---|"]
    for k, c in report["controls"].items():
        L.append(f"| {k} | {c['n']} | " + " | ".join(f"{c[x]:.3f}" for x in ("min", "median", "q95", "max")) + " |")
    return "\n".join(L) + "\n"


def write_report(cfg: dict, config_file: str = None, snr_db: float = None) -> Path:
    """Write report.json and report.md to the output directory of the run. Returns the path of the JSON."""
    report = make_report(cfg, config_file, snr_db)
    out = output_dir(cfg, snr_db) / "report.json"
    out.write_text(json.dumps(report, indent=1))
    out.with_suffix(".md").write_text(markdown(report))
    return out
