#!/usr/bin/env python3
"""Why is empirical coverage 1.00 rather than 0.95, and can the band be tightened?

The band currently used is z*sqrt(Var[f] + se^2): the GP's posterior variance of the LATENT
function plus the observation-noise variance. But the quantity the band is checked against is the
exact trajectory f(t), not a new noisy measurement of it. The observation-noise term therefore
widens the band beyond what covering f requires, which would explain coverage pinned at 1.00.

This compares three constructions at two sampling densities, 20 seeds, both observables:
  latent      : z*sqrt(Var[f])                  -- the interval for the trajectory itself
  predictive  : z*sqrt(Var[f] + se^2)           -- the interval for a new measurement (current)
  latent_infl : z*sqrt(Var[f]) with the fitted amplitude, as a control
Reports empirical coverage and mean band width for each.
"""
import os, sys, csv, time as _time, warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplcfg"))
import numpy as np, qutip as qt
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
_HERE = os.path.dirname(os.path.abspath(__file__)); _PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, os.path.join(_PARENT, "rho_reconstruction"),
           os.path.join(_PARENT, "calibration_and_adaptive"), _HERE):
    if _p not in sys.path: sys.path.append(_p)
import conditional_rho as cr
from test_pipeline_coverage import exact_curve, OPS, Z95
from coverage_basis_ablation import generate_rerandomised_measurement_df
from final_config_coverage import per_time_series, gp_fit, SUPPORTS
from Synthetic_Error_Uncertainty_Check import make_cell_seed

QUICK = os.environ.get("QUICK", "0") == "1"
CONFIGS = [(20, 60)] if QUICK else [(20, 60), (100, 200)]
SEEDS = [10, 11] if QUICK else list(range(10, 30))
TRUE_T, PRED_T = 400, 100

def main():
    t0 = _time.perf_counter()
    tlist = np.linspace(0, 2*np.pi, TRUE_T); target = np.linspace(0, 2*np.pi, PRED_T)
    states = qt.mesolve(cr.build_ising_hamiltonian(2), cr.build_plus_initial_state(2), tlist, []).states
    truth = {n: exact_curve(states, tlist, op, target) for n, op in OPS.items()}
    rows, per = [], {}
    for (nt, ns_) in CONFIGS:
        observed = np.linspace(0, TRUE_T-1, nt, dtype=int)
        for seed in SEEDS:
            cell = make_cell_seed(seed, "ZZ", ns_, nt, 7)
            np.random.seed(cell)
            mdf = generate_rerandomised_measurement_df(states, tlist, observed, 2, ns_,
                                                       shots_per_setting=1, seed=cell)
            for name in SUPPORTS:
                obs_times, series = per_time_series(mdf, SUPPORTS[name])
                y, se = series["matched"]
                mean, std, se_tgt = gp_fit(obs_times, y, se, target, "empirical")
                err = np.abs(mean - truth[name])
                for lab, band in [("latent", Z95*std), ("predictive", Z95*np.sqrt(std**2 + se_tgt**2))]:
                    cov = float(np.mean(err <= band)); width = float(np.mean(2*band))
                    per.setdefault((nt, ns_, name, lab), {})[seed] = (cov, width)
                    rows.append({"times": nt, "shadows": ns_, "observable": name, "band": lab,
                                 "seed": seed, "coverage": round(cov, 4), "mean_width": round(width, 5)})
        print(f"  {nt}x{ns_} done ({_time.perf_counter()-t0:.0f}s)", flush=True)
    n = len(SEEDS); summ = []
    print("\n==== BAND CONSTRUCTION (target coverage 0.95) ====", flush=True)
    for (nt, ns_, name, lab), d in sorted(per.items()):
        a = np.array([d[s] for s in SEEDS])
        summ.append({"times": nt, "shadows": ns_, "observable": name, "band": lab,
                     "coverage_mean": round(float(a[:,0].mean()), 4),
                     "coverage_se": round(float(a[:,0].std(ddof=1)/np.sqrt(n)), 4),
                     "width_mean": round(float(a[:,1].mean()), 5)})
        print(f"  {nt:3d}x{ns_:3d} {name:2s} {lab:10s} coverage={a[:,0].mean():.3f}+/-{a[:,0].std(ddof=1)/np.sqrt(n):.3f}"
              f"  mean width={a[:,1].mean():.4f}", flush=True)
    for fn, data in [("band_construction.csv", rows), ("band_construction_summary.csv", summ)]:
        with open(os.path.join(_HERE, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            f.write(f"# band construction comparison; {n} seeds; wall={_time.perf_counter()-t0:.0f}s\n")
            w.writeheader(); w.writerows(data)
    print(f"\nwall={_time.perf_counter()-t0:.0f}s -> saved band_construction{{,_summary}}.csv", flush=True)

if __name__ == "__main__": main()
