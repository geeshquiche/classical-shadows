#!/usr/bin/env python3
"""Three-route comparison under the consistent (matched-count) estimator, house seed counts.

Routes (independent-marginal binned per 2026-08-20 decision):
  raw          : matched-count per-time estimates, linear interpolation
  gp           : final-method GP (matched-count series, smoothed SE, empirical noise, Matern-3/2)
  conditional  : autoregressive classifier route (mechanics unchanged), with its internal
                 resampling estimator ALSO set to matched-basis normalisation for consistency

Protocol: FIXED bases per shot index (the conditional route requires it; the matched-count
normalisation makes raw/gp insensitive to the frozen draw). All routes see identical data per
seed. Errors are reported absolute and RELATIVE to the true signal scale,
rRMSE = RMSE / sqrt(mean(truth^2)), so fractions compare consistently across observables and
studies (Vageesh's convention, 2026-08-21).

Modes (env MODE):
  table       (default): 2q TFIM, 100x200, seeds 10..29 (20), XI + ZZ      -> Table 1
  robustness  : 2q TFIM ZZ, budgets {40,120,240} shadows, seeds 10..24 (15) -> robustness fig
  fourq       : 4q TFIM chain, 100x200, seeds 10..19 (10), XIII + ZZII      -> consistency check

Run:  MODE=table python routes3_final.py        (QUICK=1 for smoke)
"""
import os
import sys
import csv
import time as _time
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt
import torch
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT,):
    if _p not in sys.path:
        sys.path.append(_p)

from Synthetic_Error_Uncertainty_Check import (
    Config,
    make_cell_seed,
    build_hamiltonian,
    build_initial_state,
    pauli_string_operator,
    observable_support,
    generate_measurement_df,
    run_conditional_model,
    summarize_prediction_samples,
)
from final_config_coverage import gp_fit

Z95 = 1.959963984540054
QUICK = os.environ.get("QUICK", "0") == "1"
MODE = os.environ.get("MODE", "table")
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi

if MODE == "table":
    QUBIT, OBSERVABLES, SHADOWS_LIST = 2, ["XI", "ZZ"], [200]
    SEEDS = [10] if QUICK else list(range(10, 30))
elif MODE == "robustness":
    QUBIT, OBSERVABLES, SHADOWS_LIST = 2, ["ZZ"], [40, 120, 240]
    SEEDS = [10] if QUICK else list(range(10, 25))
elif MODE == "fourq":
    QUBIT, OBSERVABLES, SHADOWS_LIST = 4, ["XIII", "ZZII"], [200]
    SEEDS = [10] if QUICK else list(range(10, 20))
else:
    raise SystemExit(f"unknown MODE={MODE}")

if QUICK:
    TRUE_T, PRED_T, N_TIMES = 200, 40, 30
    SHADOWS_LIST = [40] if MODE != "robustness" else [40]
    COND_ITER, RESAMPLES = 30, 8
else:
    TRUE_T, PRED_T, N_TIMES = 400, 100, 100
    COND_ITER, RESAMPLES = 80, 20


def matched_series(mdf, support):
    """Matched-count per-time estimates + Laplace-smoothed SE from a fixed-basis mdf."""
    times = np.sort(mdf["time"].unique())
    y = np.full(len(times), np.nan)
    se = np.full(len(times), np.nan)
    for ti, t in enumerate(times):
        block = mdf[mdf["time"] == t]
        prods = []
        for _sid, shot in block.groupby("shadow_id"):
            row = {int(r.qubit): (str(r.pauli), int(r.outcome))
                   for r in shot.itertuples(index=False)}
            if all(row[q][0] == lab for q, lab in support):
                p = 1.0
                for q, _lab in support:
                    p *= row[q][1]
                prods.append(p)
        m = len(prods)
        if m > 0:
            arr = np.asarray(prods)
            y[ti] = arr.mean()
            p_s = (np.sum(arr > 0) + 1.0) / (m + 2.0)
            se[ti] = np.sqrt(4.0 * p_s * (1.0 - p_s) / m)
    return times, y, se


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    prediction_times = np.linspace(TIME_MIN, TIME_MAX, PRED_T)
    observed = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
    psi0 = build_initial_state(QUBIT)
    states = qt.mesolve(build_hamiltonian("tfim", QUBIT), psi0, tlist, []).states

    print(f"MODE={MODE} QUICK={QUICK}  {N_TIMES} times, shadows={SHADOWS_LIST}, "
          f"{len(SEEDS)} seeds, qubits={QUBIT}\n", flush=True)

    rows, per_seed = [], {}
    for obs in OBSERVABLES:
        operator = pauli_string_operator(obs, QUBIT)
        support = observable_support(obs)
        true_full = np.real(np.asarray(qt.expect(operator, states), dtype=float))
        truth = np.interp(prediction_times, tlist, true_full)
        truth_rms = float(np.sqrt(np.mean(truth ** 2)))

        for shadow in SHADOWS_LIST:
            cfg = Config(
                base_seed=SEEDS[0], qubit_num=QUBIT, observable_string=obs,
                dynamics_name="tfim", num_true_time_points=TRUE_T,
                prediction_time_count=PRED_T, shadow_size_grid=(shadow,),
                num_time_index_grid=(N_TIMES,), num_dataset_repeats=1,
                prediction_resamples=RESAMPLES, gp_kernel="matern32",
                conditional_training_iter=COND_ITER, num_inducing=40, gp_lr=0.05,
                use_matched_basis_estimator_for_pauli_strings=True,
                independent_seeds=True)
            for seed in SEEDS:
                data_seed = make_cell_seed(seed, obs, shadow, N_TIMES, 0)
                torch.manual_seed(data_seed)
                mdf = generate_measurement_df(
                    states=states, t_grid=tlist, observed_indices=observed,
                    n=QUBIT, shadow_size=shadow, seed=data_seed)

                obs_times, y, se = matched_series(mdf, support)
                keep = ~np.isnan(y)
                out = {}
                raw_curve = np.interp(prediction_times, obs_times[keep], y[keep])
                out["raw"] = (float(np.sqrt(np.mean((raw_curve - truth) ** 2))), np.nan)
                mean, std, se_tgt = gp_fit(obs_times, y, se, prediction_times, "empirical")
                err = np.abs(mean - truth)
                out["gp"] = (float(np.sqrt(np.mean((mean - truth) ** 2))),
                             float(np.mean(err <= Z95 * np.sqrt(std ** 2 + se_tgt ** 2))))
                try:
                    sm = run_conditional_model(mdf, prediction_times, cfg, shadow, operator)
                    metrics, _pm, _ps = summarize_prediction_samples(sm, truth)
                    out["conditional"] = (float(metrics["rmse"]),
                                          float(metrics["coverage_95pct"]))
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] conditional failed {obs} shadow={shadow} seed={seed}: {e}",
                          flush=True)
                    out["conditional"] = (np.nan, np.nan)

                for route, (rmse, cov) in out.items():
                    per_seed.setdefault((obs, shadow, route), {})[seed] = (rmse, cov)
                    rows.append({"observable": obs, "shadow": shadow, "route": route,
                                 "seed": seed, "rmse": round(rmse, 5),
                                 "rrmse": round(rmse / truth_rms, 5),
                                 "coverage_95": (round(cov, 4) if np.isfinite(cov) else "")})
                print(f"  {obs} shadow={shadow} seed={seed} done "
                      f"({_time.perf_counter()-t0:.0f}s)  "
                      + "  ".join(f"{r}={v[0]:.3f}" for r, v in out.items()), flush=True)

    ns = len(SEEDS)
    print("\n==== SUMMARY (rRMSE = RMSE / RMS(truth)) ====", flush=True)
    sum_rows = []
    for obs in OBSERVABLES:
        operator = pauli_string_operator(obs, QUBIT)
        true_full = np.real(np.asarray(qt.expect(operator, states), dtype=float))
        truth = np.interp(prediction_times, tlist, true_full)
        truth_rms = float(np.sqrt(np.mean(truth ** 2)))
        for shadow in SHADOWS_LIST:
            for route in ["raw", "gp", "conditional"]:
                arr = np.array([per_seed[(obs, shadow, route)][s] for s in SEEDS])
                r = arr[:, 0][np.isfinite(arr[:, 0])]
                c = arr[:, 1][np.isfinite(arr[:, 1])]
                r_se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan
                sum_rows.append({
                    "observable": obs, "shadow": shadow, "route": route, "n_seeds": len(r),
                    "rmse_mean": round(float(r.mean()), 4), "rmse_se": round(float(r_se), 4),
                    "rrmse_mean": round(float(r.mean() / truth_rms), 4),
                    "coverage_mean": (round(float(c.mean()), 4) if len(c) else "")})
                print(f"  {obs:4s} shadow={shadow:3d} {route:11s} "
                      f"rmse={r.mean():.4f}+/-{r_se:.4f}  rrmse={r.mean()/truth_rms:.3f}  "
                      f"cov={(c.mean() if len(c) else float('nan')):.3f}  (n={len(r)})",
                      flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# 3-route comparison, consistent matched-count estimator; MODE={MODE}; "
              f"{N_TIMES} times, shadows={SHADOWS_LIST}, {ns} seeds {SEEDS[0]}..{SEEDS[-1]}, "
              f"fixed-basis protocol (conditional requirement), paired data per seed; "
              f"conditional internal estimator matched-basis; wall={wall:.0f}s")
    for fname, data in [(f"routes3_{MODE}.csv", rows),
                        (f"routes3_{MODE}_summary.csv", sum_rows)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(header + "\n")
            w.writeheader()
            w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved routes3_{MODE}{{,_summary}}.csv", flush=True)


if __name__ == "__main__":
    main()
