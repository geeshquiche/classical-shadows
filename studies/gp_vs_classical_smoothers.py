#!/usr/bin/env python3
"""What does the GP buy over a classical smoother?  Spline/Savitzky-Golay baseline.

The report recommends GP observable regression but never compares it to a non-Bayesian
smoother.  This script runs, on the SAME per-time raw shadow series y(t_i) and at the exact
route-comparison configuration of Table 1 (2q TFIM, 100 obs times x 200 shadows, fixed-basis
protocol, seeds 10..80, matern32), the following curve estimators:

  raw          : linear interpolation of y(t_i)                     (Table 1 "raw" row)
  gp_matern    : BoTorch GP posterior mean, matern32                (Table 1 "matrix-GP" row)
  gp_rbf       : BoTorch GP posterior mean, RBF                     (kernel check)
  spline_gcv   : cubic smoothing spline, GCV-chosen penalty (scipy make_smoothing_spline)
  savgol_w9/15/25 : Savitzky-Golay, polyorder 3, window 9/15/25 (all three reported so the
                    baseline gets its best case, not a hand-picked window)

Data generation and seeding replicate estimator_comparison/compare_estimators.py cell-for-cell
(make_cell_seed, torch seeded per cell), so raw and gp_matern double as a replication check of
the published Table 1 numbers (raw 0.092/0.170, matrix-GP 0.056/0.202 for XI/ZZ).

Only the GP provides a predictive band; the classical smoothers are point estimators.  The
comparison is therefore about the accuracy margin, with calibration a separate axis.

Run:  python gp_vs_classical_smoothers.py          (full: 8 seeds)
      QUICK=1 python gp_vs_classical_smoothers.py  (1 seed, small config, smoke)
"""
import os
import sys
import csv
import time as _time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt
import torch
from scipy.interpolate import make_smoothing_spline
from scipy.signal import savgol_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT,):
    if _p not in sys.path:
        sys.path.append(_p)

from Synthetic_Error_Uncertainty_Check import (
    make_cell_seed,
    build_hamiltonian,
    build_initial_state,
    pauli_string_operator,
    generate_measurement_df,
)
from classical_shadow_matrix import construct_classical_shadow_matrices_by_time
from bayesian_matrix_inference_botorch import infer_observable_from_shadow_with_botorch

QUICK = os.environ.get("QUICK", "0") == "1"
QUBIT_NUM = 2
DYNAMICS = "tfim"
OBSERVABLES = ["XI", "ZZ"]
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
if QUICK:
    NUM_TRUE, PRED_T, N_TIMES, SHADOWS, SEEDS = 200, 40, 30, 40, [10]
else:
    NUM_TRUE, PRED_T, N_TIMES, SHADOWS, SEEDS = 400, 100, 100, 200, list(range(10, 81, 10))
SAVGOL_WINDOWS = [9, 15, 25]
METHODS = (["raw", "gp_matern", "gp_rbf", "spline_gcv"]
           + [f"savgol_w{w}" for w in SAVGOL_WINDOWS])


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, NUM_TRUE)
    prediction_times = np.linspace(TIME_MIN, TIME_MAX, PRED_T)
    observed = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
    psi0 = build_initial_state(QUBIT_NUM)
    states = qt.mesolve(build_hamiltonian(DYNAMICS, QUBIT_NUM), psi0, tlist, []).states

    print(f"QUICK={QUICK}  {N_TIMES} obs times x {SHADOWS} shadows, seeds={SEEDS}, "
          f"methods={METHODS}\n", flush=True)

    rows = []
    scores = {}  # (obs, method) -> {seed: rmse}
    for obs in OBSERVABLES:
        operator = pauli_string_operator(obs, QUBIT_NUM)
        true_full = np.real(np.asarray(qt.expect(operator, states), dtype=float))
        truth = np.interp(prediction_times, tlist, true_full)
        op_np = operator.full()

        for seed in SEEDS:
            data_seed = make_cell_seed(seed, obs, SHADOWS, N_TIMES, 0)
            torch.manual_seed(data_seed)
            mdf = generate_measurement_df(
                states=states, t_grid=tlist, observed_indices=observed,
                n=QUBIT_NUM, shadow_size=SHADOWS, seed=data_seed)
            _, mtimes, shadow_mats = construct_classical_shadow_matrices_by_time(mdf, QUBIT_NUM)

            res_m = infer_observable_from_shadow_with_botorch(
                observations=shadow_mats, operator=op_np, time_index=mtimes,
                target_time_index=prediction_times, kernel="matern32", credible_mass=0.95)
            res_r = infer_observable_from_shadow_with_botorch(
                observations=shadow_mats, operator=op_np, time_index=mtimes,
                target_time_index=prediction_times, kernel="rbf", credible_mass=0.95)
            y = np.real(np.asarray(res_m["observable_estimates"], dtype=complex))

            curves = {
                "raw": np.interp(prediction_times, mtimes, y),
                "gp_matern": np.real(np.asarray(res_m["posterior_mean"], dtype=complex)),
                "gp_rbf": np.real(np.asarray(res_r["posterior_mean"], dtype=complex)),
                "spline_gcv": make_smoothing_spline(mtimes, y)(prediction_times),
            }
            for w in SAVGOL_WINDOWS:
                if w < len(y):
                    sm = savgol_filter(y, window_length=w, polyorder=3)
                    curves[f"savgol_w{w}"] = np.interp(prediction_times, mtimes, sm)
            for method, curve in curves.items():
                rmse = float(np.sqrt(np.nanmean((curve - truth) ** 2)))
                scores.setdefault((obs, method), {})[seed] = rmse
            print(f"  {obs} seed {seed} done ({_time.perf_counter()-t0:.0f}s)  "
                  + "  ".join(f"{m}={scores[(obs, m)][seed]:.3f}"
                              for m in curves), flush=True)

    ns = len(SEEDS)
    print("\n==== SUMMARY (paired diffs vs gp_matern; Table 1 check: raw 0.092/0.170, "
          "matrix 0.056/0.202) ====", flush=True)
    for obs in OBSERVABLES:
        base = np.array([scores[(obs, "gp_matern")][s] for s in SEEDS])
        for method in METHODS:
            if (obs, method) not in scores:
                continue
            arr = np.array([scores[(obs, method)][s] for s in SEEDS])
            d = arr - base
            se = arr.std(ddof=1) / np.sqrt(ns) if ns > 1 else np.nan
            d_se = d.std(ddof=1) / np.sqrt(ns) if ns > 1 else np.nan
            rows.append({"observable": obs, "method": method,
                         "rmse_mean": round(float(arr.mean()), 4),
                         "rmse_se": round(float(se), 4),
                         "paired_diff_vs_gp_matern": round(float(d.mean()), 4),
                         "paired_diff_se": round(float(d_se), 4)})
            print(f"  {obs:2s} {method:11s} rmse={arr.mean():.4f} +/- {se:.4f}   "
                  f"diff_vs_gp={d.mean():+.4f} +/- {d_se:.4f}", flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# GP vs classical smoothers on identical per-time raw series; Table 1 config "
              f"(2q {DYNAMICS}, {N_TIMES} times x {SHADOWS} shadows, fixed-basis, "
              f"{ns} seeds {SEEDS[0]}..{SEEDS[-1]}, paired); wall={wall:.0f}s")
    with open(os.path.join(_HERE, "gp_vs_classical_smoothers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        f.write(header + "\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwall={wall:.0f}s -> saved gp_vs_classical_smoothers.csv", flush=True)


if __name__ == "__main__":
    main()
