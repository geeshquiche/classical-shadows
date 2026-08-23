#!/usr/bin/env python3
"""Budget allocation (1x500 vs 5x100) under the final matched-count estimator.

Mirrors the report's Section 5.3 design: at each of 500 observed times a budget of 500 shadows
per time is spent either as one dataset (1x500) or five independent datasets of 100 whose GP
reconstructions are averaged (5x100). Both observables; noise treatment axis {empirical, fitted}
preserved, since the study's point is that the apparent breadth advantage depends on it.

Sampling uses the exact matched-count distribution directly: for a k-local Pauli string the
number of aligned snapshots is Binomial(N, 3^-k) and, given alignment, the outcome product is
+/-1 with mean <O>(t) -- statistically identical to full snapshot simulation for this estimator.
Truth enters only sampling and final metrics.

Run:  python budget_final.py     (QUICK=1 smoke)
"""
import os
import sys
import csv
import time as _time
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))

import numpy as np
import qutip as qt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT,):
    if _p not in sys.path:
        sys.path.append(_p)

from Synthetic_Error_Uncertainty_Check import (make_cell_seed, build_hamiltonian,
                                               build_initial_state, pauli_string_operator)
from final_config_coverage import gp_fit

QUICK = os.environ.get("QUICK", "0") == "1"
TIME_MIN, TIME_MAX = 0.0, 2.0 * np.pi
if QUICK:
    TRUE_T, N_TIMES, TARGET_T, SEEDS = 120, 60, 80, [10, 11]
else:
    TRUE_T, N_TIMES, TARGET_T, SEEDS = 500, 500, 200, list(range(10, 30))
SUPPORT_K = {"XI": 1, "ZZ": 2}
ALLOCS = [("1x500", 1, 500), ("5x100", 5, 100)]
NOISES = ["empirical", "fitted"]


def matched_series_direct(rng, truth_obs, k, n_shots):
    """Matched-count series sampled exactly: m ~ Bin(N, 3^-k); n+ ~ Bin(m, (1+<O>)/2)."""
    T = len(truth_obs)
    y = np.full(T, np.nan)
    se = np.full(T, np.nan)
    p_align = 3.0 ** (-k)
    for t in range(T):
        m = rng.binomial(n_shots, p_align)
        if m == 0:
            continue
        n_plus = rng.binomial(m, np.clip((1.0 + truth_obs[t]) / 2.0, 0, 1))
        y[t] = (2.0 * n_plus - m) / m
        p_s = (n_plus + 1.0) / (m + 2.0)
        se[t] = np.sqrt(4.0 * p_s * (1.0 - p_s) / m)
    return y, se


def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(TIME_MIN, TIME_MAX, TRUE_T)
    target = np.linspace(TIME_MIN, TIME_MAX, TARGET_T)
    obs_times = tlist[np.linspace(0, TRUE_T - 1, N_TIMES, dtype=int)]
    states = qt.mesolve(build_hamiltonian("tfim", 2), build_initial_state(2), tlist, []).states

    print(f"QUICK={QUICK}  {N_TIMES} times, allocations {[a[0] for a in ALLOCS]}, "
          f"{len(SEEDS)} seeds, noises={NOISES}\n", flush=True)

    rows, per_seed = [], {}
    for obs, k in SUPPORT_K.items():
        operator = pauli_string_operator(obs, 2)
        full = np.real(np.asarray(qt.expect(operator, states), dtype=float))
        truth_obs = np.interp(obs_times, tlist, full)
        truth_tgt = np.interp(target, tlist, full)
        rms = float(np.sqrt(np.mean(truth_tgt ** 2))) or 1.0
        for seed in SEEDS:
            rng = np.random.default_rng(make_cell_seed(seed, obs, 500, N_TIMES, 3))
            for name, reps, n_shots in ALLOCS:
                series = [matched_series_direct(rng, truth_obs, k, n_shots)
                          for _ in range(reps)]
                for noise in NOISES:
                    means = []
                    for y, se in series:
                        mean, _s, _st = gp_fit(obs_times, y, se, target, noise)
                        means.append(mean)
                    avg = np.mean(means, axis=0)
                    rmse = float(np.sqrt(np.mean((avg - truth_tgt) ** 2)))
                    per_seed.setdefault((obs, name, noise), {})[seed] = rmse
                    rows.append({"observable": obs, "allocation": name, "noise": noise,
                                 "seed": seed, "rmse": round(rmse, 5),
                                 "rrmse": round(rmse / rms, 5)})
            if seed % 5 == 0:
                print(f"  {obs} seed {seed} done ({_time.perf_counter()-t0:.0f}s)", flush=True)

    ns = len(SEEDS)
    print("\n==== SUMMARY (gap% = 100*(5x100 - 1x500)/1x500, paired) ====", flush=True)
    sum_rows = []
    for obs in SUPPORT_K:
        for noise in NOISES:
            a = np.array([per_seed[(obs, "1x500", noise)][s] for s in SEEDS])
            b = np.array([per_seed[(obs, "5x100", noise)][s] for s in SEEDS])
            d = b - a
            gap = 100 * d.mean() / a.mean()
            gap_se = 100 * (d.std(ddof=1) / np.sqrt(ns)) / a.mean()
            sum_rows.append({"observable": obs, "noise": noise,
                             "rmse_1x500": round(float(a.mean()), 5),
                             "se_1x500": round(float(a.std(ddof=1) / np.sqrt(ns)), 5),
                             "rmse_5x100": round(float(b.mean()), 5),
                             "se_5x100": round(float(b.std(ddof=1) / np.sqrt(ns)), 5),
                             "gap_pct": round(float(gap), 2),
                             "gap_se_pct": round(float(gap_se), 2)})
            print(f"  {obs:2s} {noise:9s} 1x500={a.mean():.5f}  5x100={b.mean():.5f}  "
                  f"gap={gap:+.1f}% +/- {gap_se:.1f}%", flush=True)

    wall = _time.perf_counter() - t0
    header = (f"# budget allocation under final matched-count estimator; {N_TIMES} times, "
              f"budget 500/time as 1x500 vs 5x100 (averaged); {ns} seeds; exact matched-count "
              f"sampling; wall={wall:.0f}s")
    for fname, data in [("budget_final.csv", rows), ("budget_final_summary.csv", sum_rows)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(header + "\n")
            w.writeheader()
            w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved budget_final{{,_summary}}.csv", flush=True)


if __name__ == "__main__":
    main()
