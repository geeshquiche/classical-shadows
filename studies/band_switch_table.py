#!/usr/bin/env python3
"""Recompute the Table 1 GP coverage under both band constructions.

The data path is byte-identical to routes3_final.py MODE=table (same fixed-basis protocol, same
cell seeds, same matched-count series, same gp_fit) -- only the conditional route is skipped,
because the band choice does not touch it.  The GP RMSE therefore reproduces the stored table
exactly, which is the check that this script is a faithful re-run and not a new experiment.

  latent      : z*sqrt(Var[f])            -- interval for the trajectory, which is what is scored
  predictive  : z*sqrt(Var[f] + se^2)     -- interval for a fresh noisy measurement (as published)
"""
import os, sys, csv, time as _time, warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))
os.environ["MODE"] = "table"
import numpy as np, qutip as qt, torch
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
_HERE = os.path.dirname(os.path.abspath(__file__)); _PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path: sys.path.append(_p)
from Synthetic_Error_Uncertainty_Check import (make_cell_seed, build_hamiltonian,
    build_initial_state, pauli_string_operator, observable_support, generate_measurement_df)
from final_config_coverage import gp_fit
from routes3_final import matched_series, Z95

QUBIT, OBSERVABLES, SHADOW = 2, ["XI", "ZZ"], 200
SEEDS = list(range(10, 30))
TRUE_T, PRED_T, N_TIMES = 400, 100, 100

def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(0.0, 2*np.pi, TRUE_T)
    target = np.linspace(0.0, 2*np.pi, PRED_T)
    observed = np.linspace(0, TRUE_T-1, N_TIMES, dtype=int)
    states = qt.mesolve(build_hamiltonian("tfim", QUBIT), build_initial_state(QUBIT), tlist, []).states
    rows, per = [], {}
    for obs in OBSERVABLES:
        op = pauli_string_operator(obs, QUBIT); support = observable_support(obs)
        truth = np.interp(target, tlist, np.real(np.asarray(qt.expect(op, states), dtype=float)))
        for seed in SEEDS:
            ds = make_cell_seed(seed, obs, SHADOW, N_TIMES, 0)
            torch.manual_seed(ds)
            mdf = generate_measurement_df(states=states, t_grid=tlist, observed_indices=observed,
                                          n=QUBIT, shadow_size=SHADOW, seed=ds)
            ot, y, se = matched_series(mdf, support)
            mean, std, se_tgt = gp_fit(ot, y, se, target, "empirical")
            err = np.abs(mean - truth)
            rmse = float(np.sqrt(np.mean((mean - truth)**2)))
            cl = float(np.mean(err <= Z95*std))
            cp = float(np.mean(err <= Z95*np.sqrt(std**2 + se_tgt**2)))
            wl = float(np.mean(2*Z95*std)); wp = float(np.mean(2*Z95*np.sqrt(std**2 + se_tgt**2)))
            per.setdefault(obs, {})[seed] = (rmse, cl, cp, wl, wp)
            rows.append({"observable": obs, "seed": seed, "rmse": round(rmse, 5),
                         "cov_latent": round(cl, 4), "cov_predictive": round(cp, 4),
                         "width_latent": round(wl, 5), "width_predictive": round(wp, 5)})
            print(f"  {obs} seed={seed} rmse={rmse:.4f} lat={cl:.3f} pred={cp:.3f} "
                  f"({_time.perf_counter()-t0:.0f}s)", flush=True)
    n = len(SEEDS); summ = []
    print("\n==== TABLE 1 GP ROW, BOTH BANDS (20 seeds, 100 times x 200 shadows) ====", flush=True)
    for obs in OBSERVABLES:
        a = np.array([per[obs][s] for s in SEEDS])
        m = a.mean(0); e = a.std(0, ddof=1)/np.sqrt(n)
        summ.append({"observable": obs, "n_seeds": n, "rmse_mean": round(float(m[0]), 4),
                     "rmse_se": round(float(e[0]), 4),
                     "cov_latent_mean": round(float(m[1]), 4), "cov_latent_se": round(float(e[1]), 4),
                     "cov_pred_mean": round(float(m[2]), 4), "cov_pred_se": round(float(e[2]), 4),
                     "width_latent": round(float(m[3]), 4), "width_pred": round(float(m[4]), 4)})
        print(f"  {obs:2s} rmse={m[0]:.4f}+/-{e[0]:.4f}   latent cov={m[1]:.3f}+/-{e[1]:.3f} "
              f"w={m[3]:.4f}   predictive cov={m[2]:.3f}+/-{e[2]:.3f} w={m[4]:.4f}", flush=True)
    wall = _time.perf_counter() - t0
    hdr = (f"# Table 1 GP row under latent vs predictive bands; identical data path to "
           f"routes3_final MODE=table ({N_TIMES} times, {SHADOW} shadows, seeds 10..29, "
           f"fixed-basis, matched-count, empirical noise); wall={wall:.0f}s")
    for fn, data in [("band_switch_table.csv", rows), ("band_switch_table_summary.csv", summ)]:
        with open(os.path.join(_HERE, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); f.write(hdr+"\n")
            w.writeheader(); w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved band_switch_table{{,_summary}}.csv", flush=True)

if __name__ == "__main__":
    main()
