#!/usr/bin/env python3
"""Fixed-basis vs re-randomised-basis: does the measurement protocol drive undercoverage?

The report's coverage analysis (calibration_and_adaptive/test_pipeline_coverage.py: XI 0.77 /
ZZ 0.89 fit-inclusive vs nominal 0.95; matrix-GP route, 20 times x 60 shadows, matern32) uses
the FIXED-BASIS protocol: generate_repeated_measurement_df draws one Pauli setting per shadow_id
and reuses it at every observed time.  The report lists the resulting cross-time correlation as
a plausible undercoverage contributor but no ablation was run.  This script runs the SAME
pipeline under both protocols:

  fixed        : settings drawn once per shadow_id, reused at every time (report protocol)
  rerandomised : fresh settings drawn per (time, shadow_id) (standard iid shadow protocol)

Same seed list for both arms (paired at seed level), same GP route, same band constructions.
Reports empirical 95% coverage (shot-only and fit-inclusive) and GP-mean RMSE per arm, plus the
per-seed paired difference (rerandomised - fixed) with its SE.

Note: torch is seeded per (protocol-independent) cell so both arms share GP-init randomness;
the fixed arm is therefore an independent replication of the fixed-basis analysis.

Run:  python coverage_basis_ablation.py          (full: 20 seeds, ~2x the original run)
      QUICK=1 python coverage_basis_ablation.py  (2 seeds smoke)
"""
import os
import sys
import csv
import random
import time as _time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import pandas as pd
import qutip as qt
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_RHO = os.path.join(_PARENT, "rho_reconstruction")
_CAL = os.path.join(_PARENT, "calibration_and_adaptive")
for _p in (_PARENT, _RHO, _CAL):
    if _p not in sys.path:
        sys.path.append(_p)

import conditional_rho as cr
from bayesian_matrix_inference_botorch import infer_observable_from_shadow_with_botorch
from test_pipeline_coverage import exact_curve, shot_standard_error, OPS, Z95

QUICK = os.environ.get("QUICK", "0") == "1"
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
TRUE_T, PRED_T = 400, 60
N_TIMES, SHADOWS = 20, 60
SEEDS = [10, 11] if QUICK else list(range(10, 30))
KERNEL = "matern32"
SUBSYS = (0, 1)
PROTOCOLS = ["fixed", "rerandomised"]


def generate_rerandomised_measurement_df(states, t_grid, observed_indices, total_qubits,
                                         shadow_size, shots_per_setting=1, seed=None):
    """Like cr.generate_repeated_measurement_df but redraws every Pauli setting at every time."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    records = []
    for idx in observed_indices:
        state_t = states[int(idx)]
        rho_t = state_t if state_t.isoper else state_t * state_t.dag()
        t = float(t_grid[int(idx)])
        settings = [cr.create_random_pauli_obs(total_qubits) for _ in range(shadow_size)]
        for shadow_id, setting in enumerate(settings):
            for shot_repeat in range(int(shots_per_setting)):
                measurement = cr.pauli_measurement(rho_t, setting, total_qubits)
                for qubit, (label, outcome) in enumerate(measurement):
                    records.append({"time": t, "shadow_id": int(shadow_id),
                                    "shot_repeat": int(shot_repeat), "qubit": int(qubit),
                                    "pauli": str(label), "outcome": int(outcome)})
    return pd.DataFrame(records)


GENERATORS = {"fixed": cr.generate_repeated_measurement_df,
              "rerandomised": generate_rerandomised_measurement_df}


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target_times = np.linspace(TIME_MIN, TIME_MAX, PRED_T)
    states = qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2),
                        tlist, []).states
    observed = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)

    print(f"QUICK={QUICK}  matrix-GP route, {N_TIMES} obs times x {SHADOWS} shadows, "
          f"{len(SEEDS)} seeds, protocols={PROTOCOLS}\n", flush=True)

    rows = []
    # per_seed[(obs, protocol)][seed] = (cov_shot, cov_fit, rmse)
    per_seed = {}
    for name, op in OPS.items():
        truth = exact_curve(states, tlist, op, target_times)
        for protocol in PROTOCOLS:
            gen = GENERATORS[protocol]
            for seed in SEEDS:
                torch.manual_seed(seed)   # protocol-independent: both arms share init RNG
                mdf = gen(states, tlist, observed, 2, SHADOWS, shots_per_setting=1, seed=seed)
                obs_times, shadow_mats = cr.averaged_subsystem_shadow_matrices(mdf, SUBSYS)
                res = infer_observable_from_shadow_with_botorch(
                    observations=shadow_mats, operator=op, time_index=obs_times,
                    target_time_index=target_times, kernel=KERNEL, credible_mass=0.95)
                mean = np.real(np.asarray(res["posterior_mean"], dtype=complex))
                fit_var = np.real(np.asarray(res["posterior_variance"], dtype=float))
                se_obs = shot_standard_error(mdf, op, obs_times)
                se_tgt = np.interp(target_times, obs_times, se_obs)
                err = np.abs(mean - truth)
                cov_shot = float(np.mean(err <= Z95 * se_tgt))
                cov_fit = float(np.mean(err <= Z95 * np.sqrt(fit_var + se_tgt ** 2)))
                rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
                per_seed.setdefault((name, protocol), {})[seed] = (cov_shot, cov_fit, rmse)
                rows.append({"observable": name, "protocol": protocol, "seed": seed,
                             "coverage_shot_only": round(cov_shot, 4),
                             "coverage_fit_inclusive": round(cov_fit, 4),
                             "rmse": round(rmse, 5)})
            arr = np.array([per_seed[(name, protocol)][s] for s in SEEDS])
            print(f"  {name:2s} {protocol:12s}  cov_shot={arr[:,0].mean():.3f}  "
                  f"cov_fit={arr[:,1].mean():.3f}  rmse={arr[:,2].mean():.4f}  "
                  f"({_time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- summary with paired differences (rerandomised - fixed), SE over seeds ----
    print("\n==== SUMMARY (target coverage 0.95) ====", flush=True)
    summary_rows = []
    ns = len(SEEDS)
    for name in OPS:
        fx = np.array([per_seed[(name, "fixed")][s] for s in SEEDS])
        rr = np.array([per_seed[(name, "rerandomised")][s] for s in SEEDS])
        for mi, metric in enumerate(["coverage_shot_only", "coverage_fit_inclusive", "rmse"]):
            d = rr[:, mi] - fx[:, mi]
            d_se = d.std(ddof=1) / np.sqrt(ns) if ns > 1 else np.nan
            fx_se = fx[:, mi].std(ddof=1) / np.sqrt(ns) if ns > 1 else np.nan
            rr_se = rr[:, mi].std(ddof=1) / np.sqrt(ns) if ns > 1 else np.nan
            summary_rows.append({
                "observable": name, "metric": metric,
                "fixed_mean": round(float(fx[:, mi].mean()), 4),
                "fixed_se": round(float(fx_se), 4),
                "rerand_mean": round(float(rr[:, mi].mean()), 4),
                "rerand_se": round(float(rr_se), 4),
                "paired_diff": round(float(d.mean()), 4),
                "paired_diff_se": round(float(d_se), 4)})
            print(f"  {name:2s} {metric:24s} fixed={fx[:, mi].mean():.4f}  "
                  f"rerand={rr[:, mi].mean():.4f}  diff={d.mean():+.4f} +/- {d_se:.4f}",
                  flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# coverage basis-protocol ablation; matrix-GP route, {N_TIMES} times x "
              f"{SHADOWS} shadows, kernel={KERNEL}, {ns} seeds {SEEDS[0]}..{SEEDS[-1]}, "
              f"paired at seed level; wall={wall:.0f}s")
    with open(os.path.join(_HERE, "coverage_basis_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        f.write(header + "\n")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(_HERE, "coverage_basis_ablation_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        f.write(header + "\n")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nwall={wall:.0f}s -> saved coverage_basis_ablation{{,_summary}}.csv", flush=True)


if __name__ == "__main__":
    main()
