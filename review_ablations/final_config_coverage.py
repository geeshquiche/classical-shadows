#!/usr/bin/env python3
"""The repaired pipeline, all pieces together: which combination is the final method?

Grid over the two identified repairs plus Vageesh's matched-count normalisation, everything
scored on coverage AND RMSE, 20 paired seeds, XI and ZZ, empirical-noise GP throughout
(sklearn Matern-3/2, per-point alpha = measured shot SE^2 -- the repaired noise treatment):

  protocol      : fixed        (bases drawn once, report default)
                  rerandomised (fresh bases per time)
  normalisation : expected     (3^k x mean over ALL snapshots; divide by expected match rate)
                  matched      (mean of outcome products over the snapshots that actually
                                matched, i.e. divide by the realised count m -- Hajek estimator)

Prediction: 'matched' should repair the correlator's frozen-scale bias WITHOUT changing the
protocol, because the bias exists only through the expected-rate normalisation; and
rerandomised+either should also be calibrated.  The winner becomes the report's final
configuration.  Guard: times with m=0 are dropped from the GP's training set.

Run:  python final_config_coverage.py          (full: 20 seeds)
      QUICK=1 python final_config_coverage.py  (2 seeds smoke)
"""
import os
import sys
import csv
import time as _time
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_RHO = os.path.join(_PARENT, "rho_reconstruction")
_CAL = os.path.join(_PARENT, "calibration_and_adaptive")
for _p in (_PARENT, _RHO, _CAL):
    if _p not in sys.path:
        sys.path.append(_p)

import conditional_rho as cr
from test_pipeline_coverage import exact_curve, OPS, Z95
from coverage_basis_ablation import generate_rerandomised_measurement_df

QUICK = os.environ.get("QUICK", "0") == "1"
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
# Config env-overridable (2026-08-21 consolidation): NT/NS/NP, e.g. NT=100 NS=200 NP=100 for
# the A-standard. Defaults preserve the original 20x60 run.
TRUE_T = 400
PRED_T = int(os.environ.get("NP", 60))
N_TIMES, SHADOWS = int(os.environ.get("NT", 20)), int(os.environ.get("NS", 60))
SEEDS = [10, 11] if QUICK else list(range(10, 30))
SUPPORTS = {"XI": [(0, "X")], "ZZ": [(0, "Z"), (1, "Z")]}
PROTOCOLS = {"fixed": cr.generate_repeated_measurement_df,
             "rerandomised": generate_rerandomised_measurement_df}
NORMALISATIONS = ["expected", "matched"]


def per_time_series(mdf, support):
    """Per observed time: expected-rate estimate, matched-count estimate, and their SEs."""
    k = len(support)
    times = np.sort(mdf["time"].unique())
    y_exp = np.zeros(len(times)); se_exp = np.zeros(len(times))
    y_mat = np.full(len(times), np.nan); se_mat = np.full(len(times), np.nan)
    for ti, t in enumerate(times):
        block = mdf[mdf["time"] == t]
        singles, matched_products = [], []
        for _sid, shot in block.groupby(["shadow_id", "shot_repeat"]):
            row = {int(r.qubit): (str(r.pauli), int(r.outcome)) for r in shot.itertuples(index=False)}
            if all(row[q][0] == lab for q, lab in support):
                prod = 1.0
                for q, _lab in support:
                    prod *= row[q][1]
                matched_products.append(prod)
                singles.append((3.0 ** k) * prod)
            else:
                singles.append(0.0)
        singles = np.asarray(singles)
        y_exp[ti] = singles.mean()
        se_exp[ti] = singles.std(ddof=1) / np.sqrt(len(singles))
        m = len(matched_products)
        if m > 0:
            mp = np.asarray(matched_products)
            y_mat[ti] = mp.mean()
            # Laplace-smoothed binomial SE: with m ~ 6 products in {-1,+1}, the raw sample std is
            # often exactly 0 (all outcomes agree), which tells the GP the point is noise-free and
            # breaks the fit.  Smoothing keeps the SE data-driven but never degenerate.
            p_smooth = (np.sum(mp > 0) + 1.0) / (m + 2.0)
            se_mat[ti] = np.sqrt(4.0 * p_smooth * (1.0 - p_smooth) / m)
    return times, {"expected": (y_exp, se_exp), "matched": (y_mat, se_mat)}


def gp_fit(obs_times, y, se, target_times, noise):
    """noise='empirical': per-point alpha = measured SE^2 (final method).
    noise='fitted': WhiteKernel fitted by marginal likelihood (the studied alternative)."""
    keep = ~np.isnan(y)
    base = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.05, 20.0), nu=1.5)
    if noise == "empirical":
        gp = GaussianProcessRegressor(kernel=base,
                                      alpha=np.clip(se[keep], 1e-6, None) ** 2,
                                      normalize_y=False, n_restarts_optimizer=2)
    else:
        from sklearn.gaussian_process.kernels import WhiteKernel
        gp = GaussianProcessRegressor(kernel=base + WhiteKernel(1e-2, (1e-8, 1e2)),
                                      normalize_y=False, n_restarts_optimizer=2)
    gp.fit(obs_times[keep].reshape(-1, 1), y[keep])
    mean, std = gp.predict(target_times.reshape(-1, 1), return_std=True)
    se_tgt = np.interp(target_times, obs_times[keep], se[keep])
    return mean, std, se_tgt


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target_times = np.linspace(TIME_MIN, TIME_MAX, PRED_T)
    states = qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2),
                        tlist, []).states
    observed = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
    truth = {name: exact_curve(states, tlist, op, target_times) for name, op in OPS.items()}

    print(f"QUICK={QUICK}  {N_TIMES} times x {SHADOWS} shadows, {len(SEEDS)} seeds, "
          f"empnoise GP, protocols x normalisations grid\n", flush=True)

    rows, per_seed = [], {}
    for protocol, gen in PROTOCOLS.items():
        for seed in SEEDS:
            mdf = gen(states, tlist, observed, 2, SHADOWS, shots_per_setting=1, seed=seed)
            for name in SUPPORTS:
                obs_times, series = per_time_series(mdf, SUPPORTS[name])
                for norm in NORMALISATIONS:
                    for noise in ["empirical", "fitted"]:
                        y, se = series[norm]
                        mean, std, se_tgt = gp_fit(obs_times, y, se, target_times, noise)
                        err = np.abs(mean - truth[name])
                        cov = float(np.mean(err <= Z95 * np.sqrt(std ** 2 + se_tgt ** 2)))
                        rmse = float(np.sqrt(np.mean((mean - truth[name]) ** 2)))
                        per_seed.setdefault((name, protocol, norm, noise), {})[seed] = (cov, rmse)
                        rows.append({"observable": name, "protocol": protocol,
                                     "normalisation": norm, "noise": noise, "seed": seed,
                                     "coverage": round(cov, 4), "rmse": round(rmse, 5)})
        print(f"  protocol={protocol} done ({_time.perf_counter()-t0:.0f}s)", flush=True)

    ns = len(SEEDS)
    print("\n==== SUMMARY (empnoise GP everywhere; old defaults for reference: "
          "XI 0.78/0.293, ZZ 0.89/0.396) ====", flush=True)
    sum_rows = []
    for name in SUPPORTS:
        for protocol in PROTOCOLS:
            for norm in NORMALISATIONS:
                for noise in ["empirical", "fitted"]:
                    arr = np.array([per_seed[(name, protocol, norm, noise)][s] for s in SEEDS])
                    cse = arr[:, 0].std(ddof=1) / np.sqrt(ns)
                    rse = arr[:, 1].std(ddof=1) / np.sqrt(ns)
                    sum_rows.append({"observable": name, "protocol": protocol,
                                     "normalisation": norm, "noise": noise,
                                     "coverage_mean": round(float(arr[:, 0].mean()), 4),
                                     "coverage_se": round(float(cse), 4),
                                     "rmse_mean": round(float(arr[:, 1].mean()), 4),
                                     "rmse_se": round(float(rse), 4)})
                    print(f"  {name:2s} {protocol:12s} {norm:8s} {noise:9s} "
                          f"cov={arr[:, 0].mean():.3f}+/-{cse:.3f}  "
                          f"rmse={arr[:, 1].mean():.4f}+/-{rse:.4f}", flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# final-configuration grid: protocol x normalisation, empnoise Matern-3/2 GP, "
              f"{N_TIMES}x{SHADOWS}, {ns} seeds {SEEDS[0]}..{SEEDS[-1]}, paired; wall={wall:.0f}s")
    tag = f"{N_TIMES}x{SHADOWS}"
    for fname, data in [(f"final_config_coverage_{tag}.csv", rows),
                        (f"final_config_coverage_{tag}_summary.csv", sum_rows)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(header + "\n")
            w.writeheader()
            w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved final_config_coverage{{,_summary}}.csv", flush=True)


if __name__ == "__main__":
    main()
