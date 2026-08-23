#!/usr/bin/env python3
"""GP (final method) vs classical smoothers across the sampling-density ladder.
Densities (times x shadows) 20x60 .. 150x300, fresh bases, matched-count series; GP empirical-noise
Matern-3/2 vs GCV cubic spline vs Savitzky-Golay (w=15, poly 3); 20 paired seeds; both observables.
Run: python smoothers_ladder.py (QUICK=1 smoke)."""
import os, sys, csv, time as _time, warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))
import numpy as np, qutip as qt
from scipy.interpolate import make_smoothing_spline
from scipy.signal import savgol_filter
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
_HERE = os.path.dirname(os.path.abspath(__file__)); _PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, os.path.join(_PARENT, "rho_reconstruction"), os.path.join(_PARENT, "calibration_and_adaptive"), _HERE):
    if _p not in sys.path: sys.path.append(_p)
import conditional_rho as cr
from test_pipeline_coverage import exact_curve, OPS
from coverage_basis_ablation import generate_rerandomised_measurement_df
from final_config_coverage import per_time_series, gp_fit, SUPPORTS
from Synthetic_Error_Uncertainty_Check import make_cell_seed
QUICK = os.environ.get("QUICK", "0") == "1"
LADDER = [(20, 60)] if QUICK else [(20, 60), (40, 100), (60, 150), (100, 200), (150, 300)]
SEEDS = [10, 11] if QUICK else list(range(10, 30))
def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(0, 2*np.pi, 400); target = np.linspace(0, 2*np.pi, 100)
    states = qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2), tlist, []).states
    truth = {n: exact_curve(states, tlist, op, target) for n, op in OPS.items()}
    rows, per = [], {}
    for (nt, ns_) in LADDER:
        observed = np.linspace(0, len(tlist)-1, nt, dtype=int)
        for seed in SEEDS:
            cell = make_cell_seed(seed, "ZZ", ns_, nt, 6)
            np.random.seed(cell)
            mdf = generate_rerandomised_measurement_df(states, tlist, observed, 2, ns_, shots_per_setting=1, seed=cell)
            for name in SUPPORTS:
                obs_times, series = per_time_series(mdf, SUPPORTS[name]); y, se = series["matched"]; keep = ~np.isnan(y)
                curves = {"gp": gp_fit(obs_times, y, se, target, "empirical")[0],
                          "spline_gcv": make_smoothing_spline(obs_times[keep], y[keep])(target)}
                w = min(15, (keep.sum()//2)*2 - 1)
                curves["savgol"] = np.interp(target, obs_times[keep], savgol_filter(y[keep], window_length=w, polyorder=3))
                for m, c in curves.items():
                    r = float(np.sqrt(np.mean((c - truth[name])**2))); per.setdefault((name, nt, ns_, m), {})[seed] = r
                    rows.append({"observable": name, "times": nt, "shadows": ns_, "method": m, "seed": seed, "rmse": round(r, 5)})
        print(f"  {nt}x{ns_} done ({_time.perf_counter()-t0:.0f}s)", flush=True)
    n = len(SEEDS); summ = []
    print("\n==== SUMMARY (paired diff vs gp) ====", flush=True)
    for name in SUPPORTS:
        for (nt, ns_) in LADDER:
            base = np.array([per[(name, nt, ns_, "gp")][s] for s in SEEDS])
            for m in ("gp", "spline_gcv", "savgol"):
                arr = np.array([per[(name, nt, ns_, m)][s] for s in SEEDS]); d = arr - base
                summ.append({"observable": name, "times": nt, "shadows": ns_, "method": m, "rmse_mean": round(float(arr.mean()), 5),
                             "rmse_se": round(float(arr.std(ddof=1)/np.sqrt(n)), 5), "diff_vs_gp": round(float(d.mean()), 5),
                             "diff_se": round(float(d.std(ddof=1)/np.sqrt(n)), 5)})
                print(f"  {name} {nt}x{ns_} {m:10s} rmse={arr.mean():.4f}  diff={d.mean():+.4f}+/-{d.std(ddof=1)/np.sqrt(n):.4f}", flush=True)
    wall = _time.perf_counter() - t0
    header = f"# smoothers vs GP across density ladder; matched series, fresh bases; {n} seeds; wall={wall:.0f}s"
    for fname, data in [("smoothers_ladder.csv", rows), ("smoothers_ladder_summary.csv", summ)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); f.write(header + "\n"); w.writeheader(); w.writerows(data)
    print(f"\nwall={wall:.0f}s -> saved smoothers_ladder{{,_summary}}.csv", flush=True)
if __name__ == "__main__": main()
