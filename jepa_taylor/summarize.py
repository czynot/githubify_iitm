"""Aggregate results/*.json -> results.md (mean ± std over seeds)."""
import glob
import json
import os
import statistics as st
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["direct", "piwsl", "jepa", "conservative", "jepa_noJEPA", "jepa_noTaylor",
         "piwsl_fwd", "jepa_fwd"]


def get(r, path):
    for k in path.split("/"):
        if not isinstance(r, dict) or k not in r:
            return None
        r = r[k]
    return r


def fmt(vals, digits=3, sci=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    f = (lambda x: f"{x:.2e}") if sci else (lambda x: f"{x:.{digits}f}")
    if len(vals) == 1:
        return f(vals[0])
    return f"{f(st.mean(vals))} ± {f(st.stdev(vals))}"


def main():
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        r = json.load(open(p))
        runs[r["config"]].append(r)
    cfgs = [c for c in ORDER if c in runs] + [c for c in runs if c not in ORDER]
    L = []
    ref = None
    try:
        import run, evaluate as ev
        ref = ev.reference_md(run.load_data(1000)["test"])
    except Exception as e:  # noqa
        print("reference MD failed:", e)

    L.append("### Accuracy and curl (test set, 748 LJ13 configurations)\n")
    L.append("| config | seeds | E MAE / atom | F MAE (inference force) | F MAE (−∇E of energy head) | curl ‖J−Jᵀ‖/2‖J‖ |")
    L.append("|---|---|---|---|---|---|")
    for c in cfgs:
        R = runs[c]
        L.append(f"| {c} | {len(R)} | {fmt([get(r,'E_mae_per_atom') for r in R], 4)} | "
                 f"{fmt([get(r,'F_mae') for r in R])} | {fmt([get(r,'F_mae_gradE') for r in R])} | "
                 f"{fmt([get(r,'curl_asym') for r in R], 4)} |")

    L.append("\n### Molecular dynamics (32 NVE trajectories, T₀=0.2, dt=0.005τ, 20k steps = 100τ)\n")
    L.append("Drift = slope of the **true** LJ total energy per atom per τ (mean over surviving "
             "trajectories). MTS8 = RESPA with direct forces every step and a "
             "(−∇E − F_direct) correction every 8 steps.\n")
    L.append("| config | NVE survive | NVE drift | NVE T_end | MTS8 survive | MTS8 drift | MTS8 T_end |")
    L.append("|---|---|---|---|---|---|---|")
    if ref:
        L.append(f"| exact LJ (reference) | {ref['alive_frac']:.2f} | {ref['drift_per_atom_per_tau']:.2e} | "
                 f"{ref['T_end']:.3f} | – | – | – |")
    for c in cfgs:
        R = runs[c]
        L.append(f"| {c} | {fmt([get(r,'md/nve/alive_frac') for r in R], 2)} | "
                 f"{fmt([get(r,'md/nve/drift_per_atom_per_tau') for r in R], sci=True)} | "
                 f"{fmt([get(r,'md/nve/T_end') for r in R])} | "
                 f"{fmt([get(r,'md/mts8/alive_frac') for r in R], 2)} | "
                 f"{fmt([get(r,'md/mts8/drift_per_atom_per_tau') for r in R], sci=True)} | "
                 f"{fmt([get(r,'md/mts8/T_end') for r in R])} |")

    jc = [c for c in cfgs if get(runs[c][0], "F_mae_latent") is not None]
    if jc:
        L.append("\n### Predictor as a latent response model\n")
        L.append("MAE of the energy change ΔE = E(S+δr) − E(S) for random Gaussian δr of "
                 "per-coordinate std δ, predicted (a) through the predictor P(h, δr) and "
                 "(b) by re-encoding S+δr. `latent rel err` = ‖P(h,δr)−h − (enc(S+δr)−h)‖/‖enc(S+δr)−h‖. "
                 "`F latent` = −∂E(P(h,δr))/∂δr at δr=0 (backprop through predictor only).\n")
        L.append("| config | δ | mean |ΔE_true| | ΔE MAE predictor | ΔE MAE re-encode | latent rel err |")
        L.append("|---|---|---|---|---|---|")
        for c in jc:
            R = runs[c]
            for d in ("0.01", "0.03", "0.1"):
                L.append(f"| {c} | {d} | {fmt([get(r,f'dE_true_mean_abs@{d}') for r in R])} | "
                         f"{fmt([get(r,f'dE_mae_pred@{d}') for r in R])} | "
                         f"{fmt([get(r,f'dE_mae_reenc@{d}') for r in R])} | "
                         f"{fmt([get(r,f'latent_rel_err@{d}') for r in R])} |")
        L.append("\n| config | F MAE latent force | F MAE direct head |")
        L.append("|---|---|---|")
        for c in jc:
            R = runs[c]
            L.append(f"| {c} | {fmt([get(r,'F_mae_latent') for r in R])} | {fmt([get(r,'F_mae') for r in R])} |")

    L.append("\n### Cost (float64, batch of 32 structures, 1 CPU thread, ms)\n")
    L.append("| config | train time (s) | direct force | conservative force (backprop) | 8 displacements via predictor | 8 displacements via re-encoding |")
    L.append("|---|---|---|---|---|---|")
    for c in cfgs:
        R = runs[c]
        L.append(f"| {c} | {fmt([get(r,'train_time_s') for r in R], 0)} | "
                 f"{fmt([get(r,'ms_direct_force') for r in R], 1)} | "
                 f"{fmt([get(r,'ms_conservative_force') for r in R], 1)} | "
                 f"{fmt([get(r,'ms_8_displacements_predictor') for r in R], 1)} | "
                 f"{fmt([get(r,'ms_8_displacements_reencode') for r in R], 1)} |")
    out = "\n".join(L) + "\n"
    open(os.path.join(HERE, "results.md"), "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
