#!/usr/bin/env python3
"""Headline single-observable reconstruction numbers under the FINAL configuration.

Regenerates the Section 5.1 opening comparison (raw estimator vs GP reconstruction) at the
100-observed-times x 200-shadows configuration, with the final method throughout: fresh bases
per time, matched-count (Hajek) normalisation with Laplace-smoothed SE, and the GP observation
noise fixed per point to the measured SE (Matern-3/2).  Reports RMSE for raw and GP and the
GP band's empirical coverage, both observables, 20 paired seeds.

Run:  python final_headline_recon.py          (full)
      QUICK=1 python final_headline_recon.py  (2 seeds, small config)
"""
import os
import sys
import csv
import time as _time
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt
from scipy.interpolate import make_smoothing_spline
from scipy.signal import savgol_filter
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
from final_config_coverage import per_time_series, gp_fit, SUPPORTS

QUICK = os.environ.get("QUICK", "0") == "1"
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
if QUICK:
    TRUE_T, PRED_T, N_TIMES, SHADOWS, SEEDS = 200, 40, 30, 40, [10, 11]
else:
    TRUE_T, PRED_T, N_TIMES, SHADOWS, SEEDS = 400, 100, 100, 200, list(range(10, 30))


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target_times = np.linspace(TIME_MIN, TIME_MAX, PRED_T)
    states = qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2),
                        tlist, []).states
    observed = np.linspace(0, len(tlist) - 1, N_TIMES, dtype=int)
    truth = {name: exact_curve(states, tlist, op, target_times) for name, op in OPS.items()}

    print(f"QUICK={QUICK}  final method at {N_TIMES} times x {SHADOWS} shadows, "
          f"{len(SEEDS)} seeds\n", flush=True)

    rows, per_seed = [], {}
    for seed in SEEDS:
        mdf = generate_rerandomised_measurement_df(states, tlist, observed, 2, SHADOWS,
                                                   shots_per_setting=1, seed=seed)
        for name in SUPPORTS:
            obs_times, series = per_time_series(mdf, SUPPORTS[name])
            y, se = series["matched"]
            keep = ~np.isnan(y)
            raw_curve = np.interp(target_times, obs_times[keep], y[keep])
            raw_rmse = float(np.sqrt(np.mean((raw_curve - truth[name]) ** 2)))
            mean, std, se_tgt = gp_fit(obs_times, y, se, target_times, "empirical")
            gp_rmse = float(np.sqrt(np.mean((mean - truth[name]) ** 2)))
            err = np.abs(mean - truth[name])
            cov = float(np.mean(err <= Z95 * np.sqrt(std ** 2 + se_tgt ** 2)))
            # classical smoothers on the SAME final-method series (spline: GCV penalty)
            spl = make_smoothing_spline(obs_times[keep], y[keep])(target_times)
            spl_rmse = float(np.sqrt(np.mean((spl - truth[name]) ** 2)))
            w = min(15, (np.sum(keep) // 2) * 2 - 1)
            sav = np.interp(target_times, obs_times[keep],
                            savgol_filter(y[keep], window_length=w, polyorder=3))
            sav_rmse = float(np.sqrt(np.mean((sav - truth[name]) ** 2)))
            per_seed.setdefault(name, {})[seed] = (raw_rmse, gp_rmse, cov, spl_rmse, sav_rmse)
            rows.append({"observable": name, "seed": seed, "raw_rmse": round(raw_rmse, 5),
                         "gp_rmse": round(gp_rmse, 5), "gp_coverage": round(cov, 4),
                         "spline_rmse": round(spl_rmse, 5), "savgol_rmse": round(sav_rmse, 5)})
        print(f"  seed {seed} done ({_time.perf_counter()-t0:.0f}s)", flush=True)

    ns = len(SEEDS)
    print("\n==== SUMMARY ====", flush=True)
    sum_rows = []
    for name in SUPPORTS:
        arr = np.array([per_seed[name][s] for s in SEEDS])
        d = arr[:, 1] - arr[:, 0]   # gp - raw, paired
        d_spl = arr[:, 1] - arr[:, 3]   # gp - spline, paired
        stats = {
            "observable": name,
            "raw_rmse_mean": round(float(arr[:, 0].mean()), 4),
            "raw_rmse_se": round(float(arr[:, 0].std(ddof=1) / np.sqrt(ns)), 4),
            "gp_rmse_mean": round(float(arr[:, 1].mean()), 4),
            "gp_rmse_se": round(float(arr[:, 1].std(ddof=1) / np.sqrt(ns)), 4),
            "gp_coverage_mean": round(float(arr[:, 2].mean()), 4),
            "gp_coverage_se": round(float(arr[:, 2].std(ddof=1) / np.sqrt(ns)), 4),
            "spline_rmse_mean": round(float(arr[:, 3].mean()), 4),
            "spline_rmse_se": round(float(arr[:, 3].std(ddof=1) / np.sqrt(ns)), 4),
            "savgol_rmse_mean": round(float(arr[:, 4].mean()), 4),
            "savgol_rmse_se": round(float(arr[:, 4].std(ddof=1) / np.sqrt(ns)), 4),
            "paired_gp_minus_raw": round(float(d.mean()), 4),
            "paired_se": round(float(d.std(ddof=1) / np.sqrt(ns)), 4),
            "paired_gp_minus_spline": round(float(d_spl.mean()), 4),
            "paired_spline_se": round(float(d_spl.std(ddof=1) / np.sqrt(ns)), 4)}
        sum_rows.append(stats)
        print(f"  {name:2s} raw={stats['raw_rmse_mean']}+/-{stats['raw_rmse_se']}  "
              f"gp={stats['gp_rmse_mean']}+/-{stats['gp_rmse_se']}  "
              f"cov={stats['gp_coverage_mean']}+/-{stats['gp_coverage_se']}  "
              f"spline={stats['spline_rmse_mean']}+/-{stats['spline_rmse_se']}  "
              f"savgol={stats['savgol_rmse_mean']}+/-{stats['savgol_rmse_se']}  "
              f"gp-spline={stats['paired_gp_minus_spline']}+/-{stats['paired_spline_se']}", flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# headline recon, FINAL method (fresh bases/time, matched-count norm, smoothed SE, "
              f"empirical-noise Matern-3/2); {N_TIMES}x{SHADOWS}, {ns} seeds {SEEDS[0]}..{SEEDS[-1]}, "
              f"paired; wall={wall:.0f}s")
    for fname, data in [("final_headline_recon.csv", rows),
                        ("final_headline_recon_summary.csv", sum_rows)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(header + "\n")
            w.writeheader()
            w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved final_headline_recon{{,_summary}}.csv", flush=True)


if __name__ == "__main__":
    main()
