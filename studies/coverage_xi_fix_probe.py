#!/usr/bin/env python3
"""Two follow-ups to coverage_basis_ablation.py, on the report's FIXED-basis protocol.

(1) MECHANISM CHECK (no GP): under the fixed protocol each seed freezes which snapshots have
    bases matching the observable support, and how many (m).  The raw estimator normalises by
    the EXPECTED match rate, so a seed with m below/above expectation scales its WHOLE curve by
    ~m/E[m] at every time; smoothing cannot remove an error shared by all points.  Per seed we
    record m and the whole-curve scale factor (LS slope of raw estimates onto truth) and report
    the correlation, plus the correlation of m with the seed's baseline coverage from
    coverage_basis_ablation.csv (fixed arm).  Prediction: strong for ZZ (E[m]=60/9, relative
    sd ~35%), weaker for XI (E[m]=20, relative sd ~18%).

(2) NOISE-TREATMENT PROBE: the fitted-noise coverage bands come from a marginal-likelihood
    GP whose fitted noise under-shoots the shadow variance (measured ratio 0.64).  Here the GP
    observation noise is FIXED per point to the measured shot variance (sklearn alpha=se_t^2,
    Matern-3/2, only amplitude+lengthscale fitted) -- the empirical-noise treatment that wins in
    the per-element study -- against an otherwise-identical sklearn GP with FITTED WhiteKernel
    noise as the control.  Bands use the report's fit-inclusive construction
    z*sqrt(posterior_var + se^2).  Question: does supplying the noise restore the local-observable
    undercoverage that basis re-randomisation did not?

Run:  python coverage_xi_fix_probe.py          (full: 20 seeds, ~3 min)
      QUICK=1 python coverage_xi_fix_probe.py  (2 seeds smoke)
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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
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
from test_pipeline_coverage import exact_curve, shot_standard_error, OPS, Z95

QUICK = os.environ.get("QUICK", "0") == "1"
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
TRUE_T, PRED_T = 400, 60
N_TIMES, SHADOWS = 20, 60
SEEDS = [10, 11] if QUICK else list(range(10, 30))
SUBSYS = (0, 1)
SUPPORTS = {"XI": [(0, "X")], "ZZ": [(0, "Z"), (1, "Z")]}
ARMS = ["empnoise", "fitnoise"]


def matched_count(mdf, support):
    """Number of shadow_ids whose (fixed) basis matches the support on every needed qubit."""
    ok = None
    for q, lab in support:
        col = (mdf[mdf["qubit"] == q].drop_duplicates("shadow_id")
               .set_index("shadow_id")["pauli"] == lab)
        ok = col if ok is None else (ok & col)
    return int(ok.sum())


def sklearn_fit(obs_times, y, se, target_times, fixed_noise):
    base = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.05, 20.0), nu=1.5)
    if fixed_noise:
        gp = GaussianProcessRegressor(kernel=base, alpha=np.clip(se, 1e-6, None) ** 2,
                                      normalize_y=False, n_restarts_optimizer=2)
    else:
        gp = GaussianProcessRegressor(kernel=base + WhiteKernel(1e-2, (1e-8, 1e2)),
                                      normalize_y=False, n_restarts_optimizer=2)
    gp.fit(obs_times.reshape(-1, 1), y)
    mean, std = gp.predict(target_times.reshape(-1, 1), return_std=True)
    return mean, std


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target_times = np.linspace(TIME_MIN, TIME_MAX, PRED_T)
    states = qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2),
                        tlist, []).states
    observed = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
    truth = {name: exact_curve(states, tlist, op, target_times) for name, op in OPS.items()}

    # baseline per-seed fixed-arm coverage from the ablation CSV (if present)
    base_cov = {}
    abl = os.path.join(_HERE, "coverage_basis_ablation.csv")
    if os.path.exists(abl):
        with open(abl) as f:
            next(f)  # provenance comment
            for row in csv.DictReader(f):
                if row["protocol"] == "fixed":
                    base_cov[(row["observable"], int(row["seed"]))] = float(
                        row["coverage_fit_inclusive"])

    print(f"QUICK={QUICK}  fixed-basis protocol, {N_TIMES} times x {SHADOWS} shadows, "
          f"{len(SEEDS)} seeds, arms={ARMS}\n", flush=True)

    rows, mech_rows = [], []
    per_seed = {}
    for seed in SEEDS:
        mdf = cr.generate_repeated_measurement_df(states, tlist, observed, 2, SHADOWS,
                                                  shots_per_setting=1, seed=seed)
        obs_times, shadow_mats = cr.averaged_subsystem_shadow_matrices(mdf, SUBSYS)
        for name, op in OPS.items():
            y = np.array([np.trace(op @ m).real for m in shadow_mats])
            se_obs = shot_standard_error(mdf, op, obs_times)
            se_tgt = np.interp(target_times, obs_times, se_obs)
            tr = truth[name]
            truth_obs = np.interp(obs_times, target_times, tr)
            m = matched_count(mdf, SUPPORTS[name])
            slope = float(np.sum(y * truth_obs) / np.sum(truth_obs ** 2))
            mech_rows.append({"observable": name, "seed": seed, "matched_m": m,
                              "scale_slope": round(slope, 4),
                              "baseline_cov": base_cov.get((name, seed), "")})
            for arm in ARMS:
                mean, std = sklearn_fit(obs_times, y, se_obs, target_times,
                                        fixed_noise=(arm == "empnoise"))
                err = np.abs(mean - tr)
                band = Z95 * np.sqrt(std ** 2 + se_tgt ** 2)
                cov = float(np.mean(err <= band))
                rmse = float(np.sqrt(np.mean((mean - tr) ** 2)))
                per_seed.setdefault((name, arm), {})[seed] = (cov, rmse)
                rows.append({"observable": name, "arm": arm, "seed": seed,
                             "coverage_fit_inclusive": round(cov, 4), "rmse": round(rmse, 5)})
        print(f"  seed {seed} done ({_time.perf_counter()-t0:.0f}s)", flush=True)

    ns = len(SEEDS)
    print("\n==== PROBE SUMMARY ====", flush=True)
    sum_rows = []
    for name in OPS:
        for arm in ARMS:
            arr = np.array([per_seed[(name, arm)][s] for s in SEEDS])
            cov_se = arr[:, 0].std(ddof=1) / np.sqrt(ns)
            rmse_se = arr[:, 1].std(ddof=1) / np.sqrt(ns)
            sum_rows.append({"observable": name, "arm": arm,
                             "coverage_mean": round(float(arr[:, 0].mean()), 4),
                             "coverage_se": round(float(cov_se), 4),
                             "rmse_mean": round(float(arr[:, 1].mean()), 4),
                             "rmse_se": round(float(rmse_se), 4)})
            print(f"  {name:2s} {arm:9s} coverage={arr[:, 0].mean():.3f}+/-{cov_se:.3f}  "
                  f"rmse={arr[:, 1].mean():.4f}+/-{rmse_se:.4f}", flush=True)

    print("\n==== MECHANISM CHECK (fixed protocol) ====", flush=True)
    for name, support in SUPPORTS.items():
        ms = np.array([r["matched_m"] for r in mech_rows if r["observable"] == name], float)
        sl = np.array([r["scale_slope"] for r in mech_rows if r["observable"] == name], float)
        e_m = SHADOWS * (1.0 / 3.0) ** len(support)
        c_ms = float(np.corrcoef(ms, sl)[0, 1]) if len(ms) > 2 else np.nan
        line = (f"  {name:2s} E[m]={e_m:.1f}  m: mean={ms.mean():.1f} sd={ms.std(ddof=1):.1f}  "
                f"corr(m, scale)={c_ms:+.2f}")
        cov_vals = [r["baseline_cov"] for r in mech_rows
                    if r["observable"] == name and r["baseline_cov"] != ""]
        if len(cov_vals) > 2:
            cv = np.array(cov_vals, float)
            mm = np.array([r["matched_m"] for r in mech_rows
                           if r["observable"] == name and r["baseline_cov"] != ""], float)
            dev = np.abs(mm / e_m - 1.0)   # coverage should fall with |scale error|, either sign
            line += f"  corr(|m/E[m]-1|, baseline cov)={float(np.corrcoef(dev, cv)[0, 1]):+.2f}"
        print(line, flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# noise-treatment probe + mechanism check; fixed-basis protocol, {N_TIMES}x{SHADOWS}, "
              f"Matern-3/2 sklearn, {ns} seeds {SEEDS[0]}..{SEEDS[-1]}; "
              f"empnoise=per-point alpha=se^2, fitnoise=WhiteKernel control; wall={wall:.0f}s")
    for fname, data in [("coverage_xi_fix_probe.csv", rows),
                        ("coverage_xi_fix_probe_summary.csv", sum_rows),
                        ("coverage_mechanism_check.csv", mech_rows)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(header + "\n")
            w.writeheader()
            w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved probe + mechanism CSVs", flush=True)


if __name__ == "__main__":
    main()
