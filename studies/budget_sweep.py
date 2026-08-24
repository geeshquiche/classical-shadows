#!/usr/bin/env python3
"""Budget allocation across TOTAL budgets: does 'one deep dataset beats several shallow ones' hold
at every budget?  Per-time budget B in {200, 500, 1000, 2000}; allocations 1xB, 2x(B/2), 5x(B/5)
(reconstructions averaged); noise {empirical, fitted}; 20 seeds; 500 observed times; exact
matched-count sampling (see budget_final.py).  Run: python budget_sweep.py (QUICK=1 smoke)."""
import os, sys, csv, time as _time, warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))
import numpy as np, qutip as qt
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
_HERE = os.path.dirname(os.path.abspath(__file__)); _PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path: sys.path.append(_p)
from Synthetic_Error_Uncertainty_Check import make_cell_seed, build_hamiltonian, build_initial_state, pauli_string_operator
from final_config_coverage import gp_fit
from budget_final import matched_series_direct, SUPPORT_K
QUICK = os.environ.get("QUICK", "0") == "1"
if QUICK: TRUE_T, N_TIMES, TARGET_T, SEEDS, BUDGETS = 120, 60, 80, [10, 11], [200, 500]
else:     TRUE_T, N_TIMES, TARGET_T, SEEDS, BUDGETS = 500, 500, 200, list(range(10, 30)), [200, 500, 1000, 2000]
ALLOCS = [("1x", 1), ("2x", 2), ("5x", 5)]; NOISES = ["empirical", "fitted"]
def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(0, 2*np.pi, TRUE_T); target = np.linspace(0, 2*np.pi, TARGET_T)
    obs_times = tlist[np.linspace(0, TRUE_T-1, N_TIMES, dtype=int)]
    states = qt.mesolve(build_hamiltonian("tfim", 2), build_initial_state(2), tlist, []).states
    rows, per = [], {}
    for obs, k in SUPPORT_K.items():
        full = np.real(np.asarray(qt.expect(pauli_string_operator(obs, 2), states), dtype=float))
        truth_obs = np.interp(obs_times, tlist, full); truth_tgt = np.interp(target, tlist, full)
        for B in BUDGETS:
            for seed in SEEDS:
                cell = make_cell_seed(seed, obs, B, N_TIMES, 5)
                np.random.seed(cell)
                rng = np.random.default_rng(cell)
                for name, reps in ALLOCS:
                    series = [matched_series_direct(rng, truth_obs, k, B // reps) for _ in range(reps)]
                    for noise in NOISES:
                        avg = np.mean([gp_fit(obs_times, y, se, target, noise)[0] for y, se in series], axis=0)
                        rmse = float(np.sqrt(np.mean((avg - truth_tgt) ** 2)))
                        per.setdefault((obs, B, name, noise), {})[seed] = rmse
                        rows.append({"observable": obs, "budget_per_time": B, "allocation": name, "noise": noise,
                                     "seed": seed, "rmse": round(rmse, 5)})
            print(f"  {obs} B={B} done ({_time.perf_counter()-t0:.0f}s)", flush=True)
    ns = len(SEEDS); summ = []
    print("\n==== SUMMARY (gap% vs 1x, paired) ====", flush=True)
    for obs in SUPPORT_K:
        for B in BUDGETS:
            for noise in NOISES:
                base = np.array([per[(obs, B, "1x", noise)][s] for s in SEEDS])
                for name, _ in ALLOCS:
                    arr = np.array([per[(obs, B, name, noise)][s] for s in SEEDS]); d = arr - base
                    summ.append({"observable": obs, "budget_per_time": B, "allocation": name, "noise": noise,
                                 "rmse_mean": round(float(arr.mean()), 5), "rmse_se": round(float(arr.std(ddof=1)/np.sqrt(ns)), 5),
                                 "gap_pct_vs_1x": round(float(100*d.mean()/base.mean()), 2),
                                 "gap_se_pct": round(float(100*(d.std(ddof=1)/np.sqrt(ns))/base.mean()), 2)})
                    print(f"  {obs} B={B:5d} {name} {noise:9s} rmse={arr.mean():.5f}  gap={100*d.mean()/base.mean():+.1f}%", flush=True)
    wall = _time.perf_counter() - t0
    header = f"# budget sweep: per-time budget x allocation x noise; {N_TIMES} times; {ns} seeds; exact matched sampling; wall={wall:.0f}s"
    for fname, data in [("budget_sweep.csv", rows), ("budget_sweep_summary.csv", summ)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); f.write(header + "\n"); w.writeheader(); w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved budget_sweep{{,_summary}}.csv", flush=True)
if __name__ == "__main__": main()
