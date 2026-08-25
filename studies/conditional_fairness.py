#!/usr/bin/env python3
"""Is the conditional route losing to the raw estimator for a fixable reason?

The route replaces measured outcomes with model-sampled ones, so model error is injected where the
raw estimator simply uses the data. Two candidate causes for it trailing raw:
  (a) Monte-Carlo noise: the reported estimate averages only `prediction_resamples` reconstructions.
  (b) Under-training: fixed capacity (40 inducing points, 80 iterations) per classifier.
This sweeps both at the route-table configuration (2q TFIM, 100 times x 200 shadows), reporting the
conditional RMSE against the raw and GP values from the same datasets.
"""
import os, sys, csv, time as _time, warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))
import numpy as np, qutip as qt, torch
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
_HERE = os.path.dirname(os.path.abspath(__file__)); _PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path: sys.path.append(_p)
from Synthetic_Error_Uncertainty_Check import (Config, make_cell_seed, build_hamiltonian,
    build_initial_state, pauli_string_operator, observable_support, generate_measurement_df,
    run_conditional_model, summarize_prediction_samples)
from final_config_coverage import gp_fit
from routes3_final import matched_series

QUICK = os.environ.get("QUICK", "0") == "1"
OBS = os.environ.get("OBS", "XI")
if QUICK:
    TRUE_T, PRED_T, N_TIMES, SHADOW, SEEDS = 200, 40, 30, 40, [10]
    GRID = [(40, 80, 20), (40, 80, 60)]
else:
    TRUE_T, PRED_T, N_TIMES, SHADOW, SEEDS = 400, 100, 100, 200, [10, 20, 30, 40, 50, 60, 70, 80]
    GRID = [(40, 80, 20), (40, 80, 100), (80, 320, 20), (80, 320, 100)]   # (inducing, iters, resamples)

def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(0, 2*np.pi, TRUE_T); pred = np.linspace(0, 2*np.pi, PRED_T)
    observed = np.linspace(0, TRUE_T-1, N_TIMES, dtype=int)
    states = qt.mesolve(build_hamiltonian("tfim", 2), build_initial_state(2), tlist, []).states
    op = pauli_string_operator(OBS, 2); support = observable_support(OBS)
    truth = np.interp(pred, tlist, np.real(np.asarray(qt.expect(op, states), dtype=float)))
    rows, per = [], {}
    for seed in SEEDS:
        ds = make_cell_seed(seed, OBS, SHADOW, N_TIMES, 0)
        torch.manual_seed(ds)
        mdf = generate_measurement_df(states=states, t_grid=tlist, observed_indices=observed,
                                      n=2, shadow_size=SHADOW, seed=ds)
        ot, y, se = matched_series(mdf, support); keep = ~np.isnan(y)
        raw = float(np.sqrt(np.mean((np.interp(pred, ot[keep], y[keep]) - truth) ** 2)))
        gp = float(np.sqrt(np.mean((gp_fit(ot, y, se, pred, "empirical")[0] - truth) ** 2)))
        per.setdefault("raw", {})[seed] = raw; per.setdefault("gp", {})[seed] = gp
        for (ind, iters, res) in GRID:
            cfg = Config(base_seed=seed, qubit_num=2, observable_string=OBS, dynamics_name="tfim",
                         num_true_time_points=TRUE_T, prediction_time_count=PRED_T,
                         shadow_size_grid=(SHADOW,), num_time_index_grid=(N_TIMES,),
                         num_dataset_repeats=1, prediction_resamples=res, gp_kernel="matern32",
                         conditional_training_iter=iters, num_inducing=ind, gp_lr=0.05,
                         use_matched_basis_estimator_for_pauli_strings=True, independent_seeds=True)
            torch.manual_seed(ds)
            sm = run_conditional_model(mdf, pred, cfg, SHADOW, op)
            m, _pm, _ps = summarize_prediction_samples(sm, truth)
            key = f"cond_i{ind}_it{iters}_r{res}"
            per.setdefault(key, {})[seed] = float(m["rmse"])
            rows.append({"observable": OBS, "seed": seed, "inducing": ind, "iters": iters,
                         "resamples": res, "cond_rmse": round(float(m["rmse"]), 5),
                         "cond_coverage": round(float(m["coverage_95pct"]), 4),
                         "raw_rmse": round(raw, 5), "gp_rmse": round(gp, 5)})
        print(f"  seed {seed} done ({_time.perf_counter()-t0:.0f}s)", flush=True)
    n = len(SEEDS)
    print(f"\n==== CONDITIONAL FAIRNESS SWEEP ({OBS}, {n} seeds) ====", flush=True)
    summ = []
    for k, d in per.items():
        a = np.array([d[s] for s in SEEDS])
        summ.append({"arm": k, "rmse_mean": round(float(a.mean()), 5),
                     "rmse_se": round(float(a.std(ddof=1)/np.sqrt(n)), 5)})
        print(f"  {k:24s} rmse={a.mean():.4f} +/- {a.std(ddof=1)/np.sqrt(n):.4f}", flush=True)
    for fn, data in [(f"conditional_fairness_{OBS}.csv", rows),
                     (f"conditional_fairness_{OBS}_summary.csv", summ)]:
        with open(os.path.join(_HERE, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(f"# conditional capacity/resample sweep vs raw and GP on identical data; "
                    f"{n} seeds; wall={_time.perf_counter()-t0:.0f}s\n"); w.writeheader(); w.writerows(data)
    print(f"\nwall={_time.perf_counter()-t0:.0f}s -> saved conditional_fairness_{OBS}.csv", flush=True)

if __name__ == "__main__": main()
