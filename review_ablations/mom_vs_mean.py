#!/usr/bin/env python3
"""Median-of-means vs plain mean: does the estimator Appendix B derives matter in practice?

Appendix B explains why the HKP sample-complexity guarantee needs median-of-means (MoM): the
single-shot shadow estimator is heavy-tailed (a spike at 0 with rare +/-3^k excursions), so the
plain mean concentrates slowly.  The pipeline nevertheless uses the plain sample mean
throughout, and the small-shot correlator estimates are visibly unstable.  This script closes
the loop with two paired experiments on the 2-qubit TFIM:

  Leg A (estimator level, no GP): per observed time draw N snapshots, estimate <O> by
      plain mean, MoM(K=5), MoM(K=10) from the SAME snapshots; N in {60, 100, 300, 1000};
      report per-seed RMSE over times, worst-time |error|, and the paired difference vs mean.
      Also reports the measured excess kurtosis of the single-shot distribution (App. B ties
      the MoM motivation to exactly this heavy-tailedness).

  Leg B (downstream GP): feed the per-time series from each estimator into the house scalar
      GP route (BoTorch, matern32 - the observable-regression route) at 20 times x
      {60, 100} shots and compare reconstruction RMSE on the target grid, paired per seed.
      Implementation: the series is passed as 1x1 "shadow matrices" with operator [[1.0]],
      which reduces the matrix route exactly to GP regression on the series.

Snapshots here are drawn with a fresh random basis per snapshot (the iid protocol assumed by
the HKP analysis; the per-element studies use the same convention).

Run:  python mom_vs_mean.py          (full: 30 seeds leg A, 20 seeds leg B)
      QUICK=1 python mom_vs_mean.py  (small smoke)
"""
import os
import sys
import csv
import random
import time as _time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt
import torch
from scipy.stats import kurtosis

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_RHO = os.path.join(_PARENT, "rho_reconstruction")
for _p in (_PARENT, _RHO):
    if _p not in sys.path:
        sys.path.append(_p)

import conditional_rho as cr
from Synthetic_Error_Uncertainty_Check import make_cell_seed
from bayesian_matrix_inference_botorch import infer_observable_from_shadow_with_botorch

QUICK = os.environ.get("QUICK", "0") == "1"
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
TRUE_T, N_TIMES, PRED_T = 400, 20, 60
QUBITS = 2
SUPPORTS = {"XI": [(0, "X")], "ZZ": [(0, "Z"), (1, "Z")]}
if QUICK:
    N_GRID_A, SEEDS_A = [60, 100], [10, 11]
    N_GRID_B, SEEDS_B = [60], [10, 11]
else:
    N_GRID_A, SEEDS_A = [60, 100, 300, 1000], list(range(10, 40))
    N_GRID_B, SEEDS_B = [60, 100], list(range(10, 30))
ESTIMATORS = ["mean", "mom5", "mom10"]
KERNEL = "matern32"


def single_shot_values(rho_t, n_qubits, n_shots):
    """Single-shot shadow estimates of every observable in SUPPORTS from the SAME snapshots."""
    vals = {name: np.zeros(n_shots) for name in SUPPORTS}
    for i in range(n_shots):
        setting = cr.create_random_pauli_obs(n_qubits)
        meas = cr.pauli_measurement(rho_t, setting, n_qubits)
        for name, support in SUPPORTS.items():
            if all(meas[q][0] == lab for q, lab in support):
                prod = 1.0
                for q, _lab in support:
                    prod *= meas[q][1]
                vals[name][i] = (3.0 ** len(support)) * prod
    return vals


def median_of_means(v, k):
    m = len(v) // k
    if m < 1:
        return float(np.median(v))
    return float(np.median(v[: m * k].reshape(k, m).mean(axis=1)))


def estimate(v, estimator):
    if estimator == "mean":
        return float(v.mean())
    if estimator == "mom5":
        return median_of_means(v, 5)
    if estimator == "mom10":
        return median_of_means(v, 10)
    raise ValueError(estimator)


def gp_on_series(times, y, target, seed):
    """House scalar GP route via the 1x1-matrix reduction of the matrix-GP code path."""
    torch.manual_seed(seed)
    res = infer_observable_from_shadow_with_botorch(
        observations=np.asarray(y, dtype=complex).reshape(-1, 1, 1),
        operator=np.array([[1.0]], dtype=complex),
        time_index=np.asarray(times, dtype=float),
        target_time_index=target, kernel=KERNEL, credible_mass=0.95)
    return np.real(np.asarray(res["posterior_mean"], dtype=complex))


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target_times = np.linspace(TIME_MIN, TIME_MAX, PRED_T)
    states = qt.mesolve(cr.build_ising_hamiltonian(QUBITS), cr.build_plus_initial_state(QUBITS),
                        tlist, []).states
    observed = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
    obs_times = tlist[observed]
    # exact curves via qutip expectation on the full state (2-qubit system = full subsystem)
    P = {"I": qt.qeye(2), "X": qt.sigmax(), "Z": qt.sigmaz()}
    truth_obs, truth_tgt = {}, {}
    for name in SUPPORTS:
        op = qt.tensor([P[c] for c in name])
        full = np.real(np.asarray(qt.expect(op, states), dtype=float))
        truth_obs[name] = np.interp(obs_times, tlist, full)
        truth_tgt[name] = np.interp(target_times, tlist, full)

    print(f"QUICK={QUICK}  {N_TIMES} obs times; legA N={N_GRID_A} seeds={len(SEEDS_A)}; "
          f"legB N={N_GRID_B} seeds={len(SEEDS_B)}\n", flush=True)

    # ---------------- Leg A: estimator level ----------------
    rows_a = []
    kurt_pool = {name: [] for name in SUPPORTS}
    # scores[(obs, N, est)][seed] = (rmse_over_times, max_abs_err)
    scores = {}
    for N in N_GRID_A:
        for seed in SEEDS_A:
            cell = make_cell_seed(seed, "ZZ", N, N_TIMES, 0)
            np.random.seed(cell)
            random.seed(cell)
            est_series = {(name, est): np.zeros(N_TIMES) for name in SUPPORTS for est in ESTIMATORS}
            for ti, idx in enumerate(observed):
                st = states[int(idx)]
                rho_t = st if st.isoper else st * st.dag()
                vals = single_shot_values(rho_t, QUBITS, N)
                for name in SUPPORTS:
                    if N == max(N_GRID_A):
                        kurt_pool[name].append(vals[name])
                    for est in ESTIMATORS:
                        est_series[(name, est)][ti] = estimate(vals[name], est)
            for name in SUPPORTS:
                for est in ESTIMATORS:
                    err = est_series[(name, est)] - truth_obs[name]
                    scores.setdefault((name, N, est), {})[seed] = (
                        float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err))))
        line = "  ".join(
            f"{name}/{est}={np.mean([scores[(name, N, est)][s][0] for s in SEEDS_A]):.3f}"
            for name in SUPPORTS for est in ESTIMATORS)
        print(f"  legA N={N:4d} done ({_time.perf_counter()-t0:.0f}s)  {line}", flush=True)

    ns_a = len(SEEDS_A)
    for name in SUPPORTS:
        for N in N_GRID_A:
            base = np.array([scores[(name, N, "mean")][s][0] for s in SEEDS_A])
            for est in ESTIMATORS:
                arr = np.array([scores[(name, N, est)][s] for s in SEEDS_A])
                d = arr[:, 0] - base
                rows_a.append({
                    "observable": name, "N": N, "estimator": est,
                    "rmse_mean": round(float(arr[:, 0].mean()), 4),
                    "rmse_se": round(float(arr[:, 0].std(ddof=1) / np.sqrt(ns_a)), 4),
                    "maxerr_mean": round(float(arr[:, 1].mean()), 4),
                    "maxerr_se": round(float(arr[:, 1].std(ddof=1) / np.sqrt(ns_a)), 4),
                    "paired_diff_vs_mean": round(float(d.mean()), 4),
                    "paired_diff_se": round(float(d.std(ddof=1) / np.sqrt(ns_a)), 4)
                    if est != "mean" else 0.0})

    for name in SUPPORTS:
        pooled = np.concatenate(kurt_pool[name]) if kurt_pool[name] else np.zeros(1)
        k = float(kurtosis(pooled, fisher=True))
        print(f"  single-shot excess kurtosis ({name}, N={max(N_GRID_A)} pool): {k:.2f}",
              flush=True)
        rows_a.append({"observable": name, "N": max(N_GRID_A), "estimator": "kurtosis",
                       "rmse_mean": round(k, 3), "rmse_se": "", "maxerr_mean": "",
                       "maxerr_se": "", "paired_diff_vs_mean": "", "paired_diff_se": ""})

    # ---------------- Leg B: downstream GP ----------------
    rows_b = []
    gp_scores = {}
    for N in N_GRID_B:
        for seed in SEEDS_B:
            cell = make_cell_seed(seed, "ZZ", N, N_TIMES, 1)   # repeat_id=1: distinct from leg A
            np.random.seed(cell)
            random.seed(cell)
            series = {(name, est): np.zeros(N_TIMES) for name in SUPPORTS for est in ("mean", "mom5")}
            for ti, idx in enumerate(observed):
                st = states[int(idx)]
                rho_t = st if st.isoper else st * st.dag()
                vals = single_shot_values(rho_t, QUBITS, N)
                for name in SUPPORTS:
                    series[(name, "mean")][ti] = estimate(vals[name], "mean")
                    series[(name, "mom5")][ti] = estimate(vals[name], "mom5")
            for name in SUPPORTS:
                for est in ("mean", "mom5"):
                    mean_curve = gp_on_series(obs_times, series[(name, est)], target_times, cell)
                    rmse = float(np.sqrt(np.mean((mean_curve - truth_tgt[name]) ** 2)))
                    gp_scores.setdefault((name, N, est), {})[seed] = rmse
        line = "  ".join(
            f"{name}/{est}={np.mean(list(gp_scores[(name, N, est)].values())):.3f}"
            for name in SUPPORTS for est in ("mean", "mom5"))
        print(f"  legB N={N:4d} done ({_time.perf_counter()-t0:.0f}s)  {line}", flush=True)

    ns_b = len(SEEDS_B)
    for name in SUPPORTS:
        for N in N_GRID_B:
            base = np.array([gp_scores[(name, N, "mean")][s] for s in SEEDS_B])
            for est in ("mean", "mom5"):
                arr = np.array([gp_scores[(name, N, est)][s] for s in SEEDS_B])
                d = arr - base
                rows_b.append({
                    "observable": name, "N": N, "estimator": est,
                    "gp_rmse_mean": round(float(arr.mean()), 4),
                    "gp_rmse_se": round(float(arr.std(ddof=1) / np.sqrt(ns_b)), 4),
                    "paired_diff_vs_mean": round(float(d.mean()), 4),
                    "paired_diff_se": round(float(d.std(ddof=1) / np.sqrt(ns_b)), 4)
                    if est != "mean" else 0.0})

    wall = _time.perf_counter() - t0
    header_a = (f"# MoM vs mean, estimator level; 2q TFIM, {N_TIMES} times, fresh basis per "
                f"snapshot; {ns_a} seeds; paired on identical snapshots; wall={wall:.0f}s")
    header_b = (f"# MoM vs mean, downstream scalar GP ({KERNEL}); 2q TFIM, {N_TIMES} times; "
                f"{ns_b} seeds; paired on identical snapshots; wall={wall:.0f}s")
    with open(os.path.join(_HERE, "mom_vs_mean_estimator.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
        f.write(header_a + "\n")
        w.writeheader()
        w.writerows(rows_a)
    with open(os.path.join(_HERE, "mom_vs_mean_gp.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
        f.write(header_b + "\n")
        w.writeheader()
        w.writerows(rows_b)

    print("\n==== SUMMARY ====", flush=True)
    for r in rows_a:
        if r["estimator"] != "kurtosis":
            print(f"  legA {r['observable']:2s} N={r['N']:4d} {r['estimator']:5s} "
                  f"rmse={r['rmse_mean']} +/- {r['rmse_se']}  "
                  f"diff_vs_mean={r['paired_diff_vs_mean']} +/- {r['paired_diff_se']}", flush=True)
    for r in rows_b:
        print(f"  legB {r['observable']:2s} N={r['N']:4d} {r['estimator']:5s} "
              f"gp_rmse={r['gp_rmse_mean']} +/- {r['gp_rmse_se']}  "
              f"diff_vs_mean={r['paired_diff_vs_mean']} +/- {r['paired_diff_se']}", flush=True)
    print(f"\nwall={wall:.0f}s -> saved mom_vs_mean_{{estimator,gp}}.csv", flush=True)


if __name__ == "__main__":
    main()
