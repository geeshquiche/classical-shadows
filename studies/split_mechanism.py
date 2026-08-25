#!/usr/bin/env python3
"""Why does splitting a fixed budget hurt under measured noise?

Supervisor hypothesis (meeting, 2026-08-24): the split arm "applies the noise term twice", once
per replica and again when combining, inflating the noise estimate.

Alternative mechanism: no noise is added at combination. Each replica has B/k shots, so its measured
per-point noise is ~sqrt(k) larger, so its GP smooths harder; every replica is over-smoothed in the
SAME direction, and averaging cannot remove a shared bias (the same logic as the frozen-count bias).

Decisive test: decompose each arm's error across seeds into bias (systematic, common to all seeds)
and variance (seed-to-seed scatter), and record the fitted lengthscale and the amplitude attenuation
of the reconstruction. Over-smoothing predicts: split arm is BIAS-dominated, with longer fitted
lengthscale and lower amplitude. Double-counted noise would predict inflated bands, not a biased mean.

B = 500 per time, allocations 1x500 / 5x100, both noise treatments, 20 seeds, 500 observed times.
"""
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

QUICK = os.environ.get("QUICK", "0") == "1"
if QUICK: TRUE_T, N_TIMES, TARGET_T, SEEDS, B = 120, 60, 80, [10, 11], 500
else:     TRUE_T, N_TIMES, TARGET_T, SEEDS, B = 500, 500, 200, list(range(10, 30)), 500
ALLOCS = [("1x500", 1), ("5x100", 5)]; NOISES = ["empirical", "fitted"]

def fit_with_ls(obs_times, y, se, target, noise):
    """Same fit as gp_fit, but also return the fitted lengthscale."""
    keep = ~np.isnan(y)
    base = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.05, 20.0), nu=1.5)
    if noise == "empirical":
        gp = GaussianProcessRegressor(kernel=base, alpha=np.clip(se[keep], 1e-6, None) ** 2,
                                      normalize_y=False, n_restarts_optimizer=2)
    else:
        gp = GaussianProcessRegressor(kernel=base + WhiteKernel(1e-2, (1e-8, 1e2)),
                                      normalize_y=False, n_restarts_optimizer=2)
    gp.fit(obs_times[keep].reshape(-1, 1), y[keep])
    mean = gp.predict(target.reshape(-1, 1))
    ls = float(gp.kernel_.k1.k2.length_scale if noise == "fitted" else gp.kernel_.k2.length_scale)
    return mean, ls

def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(0, 2*np.pi, TRUE_T); target = np.linspace(0, 2*np.pi, TARGET_T)
    obs_times = tlist[np.linspace(0, TRUE_T-1, N_TIMES, dtype=int)]
    states = qt.mesolve(build_hamiltonian("tfim", 2), build_initial_state(2), tlist, []).states
    rows, curves = [], {}
    for obs, k in SUPPORT_K.items():
        full = np.real(np.asarray(qt.expect(pauli_string_operator(obs, 2), states), dtype=float))
        truth_obs = np.interp(obs_times, tlist, full); truth = np.interp(target, tlist, full)
        amp_true = truth.max() - truth.min()
        for seed in SEEDS:
            rng = np.random.default_rng(make_cell_seed(seed, obs, B, N_TIMES, 9))
            np.random.seed(make_cell_seed(seed, obs, B, N_TIMES, 9))
            for name, reps in ALLOCS:
                series = [matched_series_direct(rng, truth_obs, k, B // reps) for _ in range(reps)]
                for noise in NOISES:
                    fits = [fit_with_ls(obs_times, y, se, target, noise) for y, se in series]
                    avg = np.mean([f[0] for f in fits], axis=0)
                    ls = float(np.mean([f[1] for f in fits]))
                    curves.setdefault((obs, name, noise), []).append(avg)
                    rows.append({"observable": obs, "allocation": name, "noise": noise, "seed": seed,
                                 "rmse": round(float(np.sqrt(np.mean((avg-truth)**2))), 5),
                                 "lengthscale": round(ls, 4),
                                 "amplitude_ratio": round(float((avg.max()-avg.min())/amp_true), 4)})
        print(f"  {obs} done ({_time.perf_counter()-t0:.0f}s)", flush=True)
    print("\n==== BIAS / VARIANCE DECOMPOSITION ====", flush=True)
    summ = []
    for obs, k in SUPPORT_K.items():
        full = np.real(np.asarray(qt.expect(pauli_string_operator(obs, 2), states), dtype=float))
        truth = np.interp(target, tlist, full)
        for name, _ in ALLOCS:
            for noise in NOISES:
                C_ = np.array(curves[(obs, name, noise)])
                bias2 = float(np.mean((C_.mean(axis=0) - truth) ** 2))
                var = float(np.mean(C_.var(axis=0, ddof=1)))
                sel = [r for r in rows if r["observable"] == obs and r["allocation"] == name and r["noise"] == noise]
                ls = float(np.mean([r["lengthscale"] for r in sel]))
                ar = float(np.mean([r["amplitude_ratio"] for r in sel]))
                summ.append({"observable": obs, "allocation": name, "noise": noise,
                             "rmse": round(float(np.sqrt(bias2 + var)), 5),
                             "bias2": round(bias2, 7), "variance": round(var, 7),
                             "bias_share_pct": round(100*bias2/(bias2+var), 1),
                             "mean_lengthscale": round(ls, 4), "amplitude_ratio": round(ar, 4)})
                print(f"  {obs:2s} {name:6s} {noise:9s} rmse={np.sqrt(bias2+var):.4f}  "
                      f"bias={100*bias2/(bias2+var):5.1f}% of MSE  lengthscale={ls:.3f}  "
                      f"amplitude={ar:.3f} of truth", flush=True)
    for fname, data in [("split_mechanism.csv", rows), ("split_mechanism_summary.csv", summ)]:
        with open(os.path.join(_HERE, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(f"# split-allocation mechanism test; B={B}/time, {N_TIMES} times, {len(SEEDS)} seeds; "
                    f"wall={_time.perf_counter()-t0:.0f}s\n"); w.writeheader(); w.writerows(data)
    print(f"\nwall={_time.perf_counter()-t0:.0f}s -> saved split_mechanism{{,_summary}}.csv", flush=True)

if __name__ == "__main__": main()
